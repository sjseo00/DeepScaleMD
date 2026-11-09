"""The per-layer exchange op `dsmd::halo_exchange`, for models that exchange node features between their message-passing layers
(metadata `layer_exchange`).

Why: a model with L message-passing layers needs ghost atoms L cutoffs deep when every rank computes all its layers alone
(docs/mlip_integration_design.md 5.1), and the work on those ghosts is what keeps a domain-decomposed run from speeding up. If,
after every layer, the owners send the new features of their atoms to the ranks that hold copies of them as ghosts, the ghosts
need to be only one cutoff deep: the ghost atoms never run a layer themselves, they only receive.

    h_owned  [n_owned, F]   the features this rank computed for its own atoms
    halo_exchange(h_owned, n_ghost) -> [n_owned + n_ghost, F]
        rows [0, n_owned): h_owned unchanged;  rows [n_owned, ...): the features of the owner of each ghost atom

The op is differentiable and its backward is the adjoint: the gradient of every ghost row is added to the row of its owner (the
owner's rank when the owner is on another rank). Forces then come out of one ordinary backward pass of the owned energy sum,
which crosses the ranks through these ops.

Three places implement the op, with the same schema `dsmd::halo_exchange(Tensor feats, int n_ghost) -> Tensor`:
  * this file: the Python version, used while a model is built, scripted and checked (and by the tests). It asks a *transport*
    for the data movement; the default transport has no ghosts (n_ghost = 0 is the identity) and refuses anything else. Inside
    dsMD (a --backend pytorch model folder, pytorch_runner.py) the op is already there, registered by the C++ version: this file then
    registers nothing, and the model calls dsMD's op with dsMD's transport.
  * src/HaloExchangeOp.cpp: the C++ version the MD program registers; the saved TorchScript model calls it by name.
  * the transport behind the C++ op is the MD's halo exchange of dsDecompX (Decomp::forward_exchange / reverse_exchange).

The second op, `dsmd::allreduce_sum(Tensor x) -> Tensor`, is for the few numbers a model sums over the whole system (so3lr: the
charge sum that fixes the total charge, the denominators of its charge and spin embeddings): the element-wise sum of x over all
ranks, the same on every rank; its backward is the same sum of the incoming gradients. The transport does it (sum_over_ranks); in
one process without ranks it is a copy. In dsMD the transport sums through the MLIP rank communicator (md_simulation.cpp).
"""
import threading
from contextlib import contextmanager
from typing import Optional

import torch
from torch import Tensor


class HaloTransport:
    """What the op needs from the data movement. feats / grad are the tensors of one rank."""

    def forward(self, feats: Tensor, n_ghost: int) -> Tensor:
        """[n_owned, F] -> [n_owned + n_ghost, F]: the owned rows followed by the ghost rows (the owners' features)."""
        raise NotImplementedError

    def reverse(self, grad: Tensor, n_owned: int) -> Tensor:
        """[n_owned + n_ghost, F] -> [n_owned, F]: the owned rows with the gradients of the ghost copies added."""
        raise NotImplementedError

    def sum_over_ranks(self, x: Tensor) -> Tensor:
        """x (a few numbers) -> the element-wise sum of x over all ranks. One process (no other ranks): a copy."""
        return x.clone()


class NoGhostTransport(HaloTransport):
    """The default: a rank without ghost atoms (a single rank). The exchange is the identity."""

    def forward(self, feats: Tensor, n_ghost: int) -> Tensor:
        if n_ghost != 0:
            raise RuntimeError(f"dsmd::halo_exchange: {n_ghost} ghost atoms but no halo transport is set (use halo_op.local_ghosts "
                               "for one process, or run the model inside the MD program)")
        return feats.clone()

    def reverse(self, grad: Tensor, n_owned: int) -> Tensor:
        return grad[:n_owned].clone()


class GhostMapTransport(HaloTransport):
    """One process: every ghost atom is a copy of an owned atom of the same process (periodic images, as in reference.json).
    owner [n_ghost] int64: the owned row each ghost is a copy of."""

    def __init__(self, owner: Tensor):
        self.owner = owner

    def forward(self, feats: Tensor, n_ghost: int) -> Tensor:
        if n_ghost != int(self.owner.numel()):
            raise RuntimeError(f"dsmd::halo_exchange: {n_ghost} ghost atoms, the ghost map has {int(self.owner.numel())}")
        owner = self.owner.to(feats.device)
        return torch.cat([feats, feats[owner]], dim=0)

    def reverse(self, grad: Tensor, n_owned: int) -> Tensor:
        owner = self.owner.to(grad.device)
        out = grad[:n_owned].clone()
        out.index_add_(0, owner, grad[n_owned:])
        return out


_local = threading.local()
_default = NoGhostTransport()


def get_transport() -> HaloTransport:
    return getattr(_local, "transport", None) or _default


def set_transport(transport: Optional[HaloTransport]) -> None:
    """Sets the transport of the calling thread (None: back to the default)."""
    _local.transport = transport


@contextmanager
def use_transport(transport: HaloTransport):
    previous = getattr(_local, "transport", None)
    _local.transport = transport
    try:
        yield transport
    finally:
        _local.transport = previous


@contextmanager
def local_ghosts(owner: Tensor):
    """Runs a model whose ghost atoms are copies of owned atoms of the same structure (owner [n_ghost]: the owned row of each)."""
    with use_transport(GhostMapTransport(torch.as_tensor(owner, dtype=torch.int64))) as transport:
        yield transport


class _AllreduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        transport = get_transport()
        ctx.transport = transport
        return transport.sum_over_ranks(x.detach().contiguous())

    @staticmethod
    def backward(ctx, grad: Tensor):
        return ctx.transport.sum_over_ranks(grad.contiguous())


class _HaloExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feats: Tensor, n_ghost: int) -> Tensor:
        transport = get_transport()
        ctx.transport = transport          # the backward may run in another thread: take the transport along
        ctx.n_owned = int(feats.shape[0])
        return transport.forward(feats.contiguous(), int(n_ghost))

    @staticmethod
    def backward(ctx, grad: Tensor):
        return ctx.transport.reverse(grad.contiguous(), ctx.n_owned), None


def _registered_by_the_md_program(name: str = "halo_exchange") -> bool:
    """True inside dsMD: src/HaloExchangeOp.cpp registered the op when the program started, before its embedded Python imports
    this file (a --backend pytorch model folder)."""
    try:
        getattr(torch.ops.dsmd, name)
    except (AttributeError, RuntimeError):
        return False
    return True


def _backend_impl(feats: Tensor, n_ghost: int) -> Tensor:
    return get_transport().forward(feats.contiguous(), int(n_ghost))


def _autograd_impl(feats: Tensor, n_ghost: int) -> Tensor:
    return _HaloExchange.apply(feats, n_ghost)


def _sum_backend_impl(x: Tensor) -> Tensor:
    return get_transport().sum_over_ranks(x.contiguous())


def _sum_autograd_impl(x: Tensor) -> Tensor:
    return _AllreduceSum.apply(x)


def _sum_missing_in_the_md_program(x: Tensor) -> Tensor:
    raise RuntimeError("dsmd::allreduce_sum: this dsMD build registers dsmd::halo_exchange but not dsmd::allreduce_sum: the model "
                       "needs a newer dsMD (rebuild it)")


# FRAGMENT: the MD program registers the same namespace from C++ (src/HaloExchangeOp.cpp), so it must not be claimed exclusively.
_md_program = _registered_by_the_md_program()
if not _md_program or not _registered_by_the_md_program("allreduce_sum"):
    _library = torch.library.Library("dsmd", "FRAGMENT")
    if not _md_program:
        _library.define("halo_exchange(Tensor feats, int n_ghost) -> Tensor")
        _library.impl("halo_exchange", _backend_impl, "CompositeExplicitAutograd")
        _library.impl("halo_exchange", _autograd_impl, "Autograd")
        _library.define("allreduce_sum(Tensor x) -> Tensor")
        _library.impl("allreduce_sum", _sum_backend_impl, "CompositeExplicitAutograd")
        _library.impl("allreduce_sum", _sum_autograd_impl, "Autograd")
    else:
        # an older dsMD: its ranks cannot sum, and a copy would give each rank its own partial sum without a word
        _library.define("allreduce_sum(Tensor x) -> Tensor")
        _library.impl("allreduce_sum", _sum_missing_in_the_md_program, "CompositeExplicitAutograd")


def halo_exchange(feats: Tensor, n_ghost: int) -> Tensor:
    """The op as a Python call (a scripted model calls torch.ops.dsmd.halo_exchange directly)."""
    return torch.ops.dsmd.halo_exchange(feats, n_ghost)


def allreduce_sum(x: Tensor) -> Tensor:
    """The op as a Python call (a scripted model calls torch.ops.dsmd.allreduce_sum directly)."""
    return torch.ops.dsmd.allreduce_sum(x)
