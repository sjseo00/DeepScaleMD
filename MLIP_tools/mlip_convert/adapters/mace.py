"""MACE adapter (design 3.4, as checked in section 11: mace-torch 0.3.16).

What the installed mace does, which this adapter relies on:
  * the model is called as model(data, training=False, compute_force=False) with a dict
      positions [N, 3]        requires grad; the wrapper provides it
      node_attrs [N, Z]       one-hot over model.atomic_numbers, in the model's dtype
      edge_index [2, E]       int64; edge_index[0] = sender, edge_index[1] = receiver
      shifts, unit_shifts [E, 3]   zeros here: ghosts and periodic images are real atoms
      cell [3, 3]             not used without virials
      batch [N], ptr [2]      one graph
      head [1]                head index
    vectors = positions[receiver] - positions[sender] + shifts. A full edge list (both directions) is symmetric, so the
    standard edge_index is passed as it is.
  * out["node_energy"] [N] holds E0 + scale/shift-ed interaction energy per atom (in eV): the sum over atoms is what the
    ASE calculator reports as the energy. Forces from the model are not used: MDWrapper differentiates the owned sum.
  * cutoff = model.r_max, num_layers = model.num_interactions, so ghost_cutoff = num_layers * r_max.
mace's own LAMMPS export (mace/cli/create_lammps_model.py) does the same thing: node energy times a local mask, then
autograd.grad with respect to positions.
"""
import copy
import os
import types
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# e3nn 0.4.4 (pinned by mace-torch) loads its constants with torch.load, which refuses them under torch >= 2.6; mace's
# own tools set this variable for the same reason. It only changes how files we are given are loaded.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import torch
from torch import Tensor

from .base import MLIPAdapter


def _e3nn_generated_code_is_scripted() -> bool:
    try:
        import e3nn
    except ImportError:
        return True
    return bool(e3nn.get_optimization_defaults()["jit_script_fx"])


def _rebuild_without_scripted_code(model: torch.nn.Module) -> torch.nn.Module:
    """A copy of `model` whose e3nn tensor products are plain fx modules. A pickled MACE model carries them as TorchScript
    modules, which make_fx cannot trace (the AOTI export); mace's own torch.compile path rebuilds the model from its
    configuration with e3nn's code generation unscripted, and so does this (extract_model, then simplify)."""
    from mace.tools.compile import simplify
    from mace.tools.scripts_utils import extract_model
    dtype = next(model.parameters()).dtype
    previous_default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)          # the copy is built in the default dtype: keep float64 weights float64
    try:
        return simplify(extract_model(model, map_location="cpu"))
    finally:
        torch.set_default_dtype(previous_default)


def _to_cueq(model: torch.nn.Module, device: str) -> torch.nn.Module:
    """The model with cuEquivariance tensor products (mace's own e3nn -> cueq conversion, cli/convert_e3nn_cueq.py). `device`
    matters: the fused convolution is only used for "cuda"."""
    try:
        from mace.cli.convert_e3nn_cueq import run as convert
    except ImportError as error:
        raise ValueError(f"--accel cueq needs the cuequivariance packages: {error}") from error
    previous_default = torch.get_default_dtype()
    try:
        return convert(model, device=device, return_model=True)      # it sets the default dtype to the model's
    finally:
        torch.set_default_dtype(previous_default)


def _to_openeq(model: torch.nn.Module, device: str) -> torch.nn.Module:
    """The model with OpenEquivariance convolutions (mace's own conversion, cli/convert_e3nn_oeq.py); the kernels are built for
    `device`, which has to be a GPU."""
    try:
        from mace.cli.convert_e3nn_oeq import run as convert
    except ImportError as error:
        raise ValueError(f"--accel openeq needs the Python package openequivariance (pip install openequivariance): "
                         f"{error}") from error
    previous_default = torch.get_default_dtype()
    try:
        return convert(model, device=device)
    except ImportError as error:           # mace builds the kernels, and imports their package, in the conversion itself
        raise ValueError(f"--accel openeq needs the Python package openequivariance (pip install openequivariance): "
                         f"{error}") from error
    finally:
        torch.set_default_dtype(previous_default)


def _make_patched_forwards_scriptable(model: torch.nn.Module) -> int:
    """mace's cueq / openeq conversions (mace/modules/wrapper_ops.py) replace a convolution's forward by a function of their own
    that calls `self.original_forward`, a bound method stored on the instance; TorchScript cannot convert such an attribute
    ("Module ... has no attribute 'original_forward'"). Here the convolution before the patch becomes a submodule of its own
    (same parameters, its own TorchScript class), so `self.original_forward(...)` is an ordinary submodule call. Returns how many
    modules were changed."""
    changed = 0
    for module in list(model.modules()):
        if not isinstance(module.__dict__.get("original_forward"), types.MethodType):
            continue
        original = copy.copy(module)
        original._parameters = dict(module._parameters)
        original._buffers = dict(module._buffers)
        original._modules = dict(module._modules)
        original.__class__ = type(type(module).__name__ + "Original", (type(module),), {})
        original.__dict__.pop("forward", None)
        original.__dict__.pop("original_forward", None)
        del module.__dict__["original_forward"]
        module.original_forward = original
        changed += 1
    return changed


def load_mace_model(path, dtype: torch.dtype = torch.float64, accel: str = "e3nn", device: str = "cpu") -> torch.nn.Module:
    """A saved MACE model (torch.save of the whole model, as mace's tools write it) in eval mode, without gradients.
    accel "cueq" / "openeq" converts it to cuEquivariance / OpenEquivariance tensor products (on `device`). While e3nn's
    scripting of generated code is off (the AOTI export turns it off), the model is rebuilt unscripted."""
    if not Path(path).is_file():
        raise ValueError(f"MACE model file not found: {path}")
    if accel not in ("e3nn", "cueq", "openeq"):
        raise ValueError(f"MACE supports --accel e3nn, cueq and openeq, not '{accel}'")
    model = torch.load(str(path), map_location="cpu")
    if accel == "cueq":
        model = _to_cueq(model, device)
    elif accel == "openeq":
        model = _to_openeq(model, device)
    elif not _e3nn_generated_code_is_scripted():
        model = _rebuild_without_scripted_code(model)
    model = model.to(dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if accel != "e3nn":
        _make_patched_forwards_scriptable(model)
    return model


class _StockInteraction(torch.nn.Module):
    """An interaction block of a kind this file does not know: mace's own forward on all rows, then the owned rows."""

    def __init__(self, block: torch.nn.Module):
        super().__init__()
        self.block = block

    def forward(self, node_attrs: Tensor, node_feats: Tensor, edge_attrs: Tensor, edge_feats: Tensor, edge_index: Tensor,
                cutoff: Optional[Tensor], n_owned: int) -> Tuple[Tensor, Optional[Tensor]]:
        message, skip = self.block(node_attrs=node_attrs, node_feats=node_feats, edge_attrs=edge_attrs, edge_feats=edge_feats,
                                   edge_index=edge_index, cutoff=cutoff, first_layer=False, lammps_class=None,
                                   lammps_natoms=(0, 0))
        if skip is not None:
            return message[:n_owned], skip[:n_owned]
        return message[:n_owned], None


class _OwnedRowsInteraction(torch.nn.Module):
    """mace's interaction block (the same modules, the same weights) for a rank with ghost atoms, computing only what the owned atoms
    need. The block of mace works on every row it is given, ghost rows included: its skip connection, the normalization and the
    reshape are applied to rows that are dropped afterwards, and the messages are summed into rows no edge feeds (only owned atoms
    receive edges). With ghost atoms outnumbering the owned ones (a domain of a few hundred atoms has ~4 times as many) that work does not shrink when more ranks
    are added. Here the ghost rows are used where they are needed, as senders of messages (linear_up, the convolution), and
    everything after the messages is done on the n_owned rows. The four block kinds of mace 0.3.x differ in where the skip
    connection sits (before the messages: residual; after the normalization: the others) and in the normalization (the number of
    neighbors, or a density). A fused convolution (cuEquivariance / OpenEquivariance: the block has `conv_fusion`) gathers and sums
    by itself and returns a row for every node: its first n_owned rows are the owned ones."""
    __constants__ = ["skip_before", "skip_after", "density", "fused"]

    def __init__(self, block: torch.nn.Module, skip_before: bool, skip_after: bool, density: bool):
        super().__init__()
        self.block = block
        self.skip_before = skip_before
        self.skip_after = skip_after
        self.density = density
        self.fused = hasattr(block, "conv_fusion")

    def forward(self, node_attrs: Tensor, node_feats: Tensor, edge_attrs: Tensor, edge_feats: Tensor, edge_index: Tensor,
                cutoff: Optional[Tensor], n_owned: int) -> Tuple[Tensor, Optional[Tensor]]:
        owned_attrs = node_attrs[:n_owned]
        skip: Optional[Tensor] = None
        if self.skip_before:
            skip = self.block.skip_tp(node_feats[:n_owned], owned_attrs)
        up = self.block.linear_up(node_feats)
        tp_weights = self.block.conv_tp_weights(edge_feats)
        if cutoff is not None:
            tp_weights = tp_weights * cutoff
        if self.fused:
            message = self.block.conv_tp(up, edge_attrs, tp_weights, edge_index)[:n_owned]
        else:
            messages = self.block.conv_tp(up[edge_index[0]], edge_attrs, tp_weights)
            message = messages.new_zeros([n_owned] + [size for size in messages.shape[1:]]).index_add_(0, edge_index[1], messages)
        if self.density:
            edge_density = torch.tanh(self.block.density_fn(edge_feats) ** 2)
            if cutoff is not None:
                edge_density = edge_density * cutoff
            density = edge_density.new_zeros([n_owned] + [size for size in edge_density.shape[1:]]).index_add_(0, edge_index[1], edge_density)
            message = self.block.linear(message) / (density + 1)
        else:
            message = self.block.linear(message) / self.block.avg_num_neighbors
        if self.skip_after:
            message = self.block.skip_tp(message, owned_attrs)
        return self.block.reshape(message), skip


# class name of mace's interaction block -> (skip connection before the messages, after them, density normalization)
_OWNED_ROWS_KINDS = {
    "RealAgnosticResidualInteractionBlock": (True, False, False),
    "RealAgnosticInteractionBlock": (False, True, False),
    "RealAgnosticDensityInteractionBlock": (False, True, True),
    "RealAgnosticDensityResidualInteractionBlock": (True, False, True),
}


def owned_rows_interaction(block: torch.nn.Module) -> torch.nn.Module:
    kind = _OWNED_ROWS_KINDS.get(type(block).__name__)
    if kind is None:
        return _StockInteraction(block)
    return _OwnedRowsInteraction(block, *kind)


class MACEAdapter(MLIPAdapter):
    model_family = "mace"
    e3nn_compile = True   # export_torchscript runs e3nn.util.jit.compile before scripting
    supports_shifts = True
    supports_layer_exchange = True
    edge_receiver_row = 1       # mace: edge_index[0] = sender, edge_index[1] = receiver

    def __init__(self, model: torch.nn.Module, head: Optional[str] = None, model_name: str = "mace", accel: str = "e3nn"):
        super().__init__([int(z) for z in model.atomic_numbers.tolist()])
        self.accel = accel
        self.model = model
        self.model_name = model_name
        self.cutoff = float(model.r_max)
        self.num_layers = int(model.num_interactions)

        self.has_extra_inputs = hasattr(model, "joint_embedding")    # fine-tuned models with extra node inputs: no layer exchange

        heads = list(getattr(model, "heads", None) or [None])
        if head is None:
            index = len(heads) - 1          # mace's own export picks the last head by default
        elif head in heads:
            index = heads.index(head)
        else:
            raise ValueError(f"head '{head}' is not one of the model's heads: {heads}")
        self.head_name = heads[index] if len(heads) > 1 else None
        self.register_buffer("head", torch.tensor([index], dtype=torch.long))
        self.register_buffer("species_one_hot", torch.eye(len(self.elements)))

    def enable_layer_exchange(self) -> None:
        """Switches layer exchange on and builds the per-layer wrappers that forward_layered runs (they share the weights of the
        model; they are not built otherwise, so a model converted without layer exchange is scripted exactly as before)."""
        super().enable_layer_exchange()
        self.layer_interactions = torch.nn.ModuleList([owned_rows_interaction(block) for block in self.model.interactions])

    def package_versions(self) -> Dict[str, Optional[str]]:
        import e3nn
        import mace
        return {"mace": mace.__version__, "e3nn": e3nn.__version__}

    @property
    def reference_elements(self):
        # A chemically ordinary mix for the reference structure: water-like when the model knows H and O.
        if 1 in self.elements and 8 in self.elements:
            return [8, 1]
        return self.elements[:2]

    def forward(self, pos: Tensor, species: Tensor, edge_index: Tensor) -> Tensor:
        n_atoms = pos.shape[0]
        n_edges = edge_index.shape[1]
        data: Dict[str, Tensor] = {
            "positions": pos,
            "node_attrs": self.species_one_hot.to(pos.dtype)[species],
            "edge_index": edge_index,
            "shifts": torch.zeros((n_edges, 3), dtype=pos.dtype, device=pos.device),
            "unit_shifts": torch.zeros((n_edges, 3), dtype=pos.dtype, device=pos.device),
            "cell": torch.zeros((3, 3), dtype=pos.dtype, device=pos.device),
            "batch": torch.zeros(n_atoms, dtype=torch.long, device=pos.device),
            # [0, n_atoms]; torch.tensor([0, n_atoms]) would turn a symbolic size into a constant when traced for AOTI
            "ptr": torch.arange(2, dtype=torch.long, device=pos.device) * n_atoms,
            "head": self.head,
        }
        out = self.model(data, training=False, compute_force=False)
        node_energy = out["node_energy"]
        assert node_energy is not None
        return node_energy.to(pos.dtype)

    def forward_layered(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor, n_owned: int) -> Tensor:
        """The layer loop of mace's ScaleShiftMACE.forward (mace-torch 0.3.16), with the node features of the owned atoms sent to
        their ghost copies before every layer but the first (mace's own LAMMPS ML-IAP path does the same through
        LAMMPS_MP, but is switched off in TorchScript). Only the owned atoms are receivers of edges, so after the first
        interaction only the owned rows are kept: the product block, the readouts and the energy are computed for them alone.
        Per atom: E0 + scale_shift(sum of the readouts [+ pair repulsion]), as in forward_with_shifts."""
        model = self.model
        n_all = pos.shape[0]
        n_ghost = n_all - n_owned
        node_attrs = self.species_one_hot.to(pos.dtype)[species]
        sender = edge_index[0]
        receiver = edge_index[1]
        vectors = pos[receiver] - pos[sender] + edge_shift
        lengths = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
        heads = self.head.expand(n_all)
        owned_heads = heads[:n_owned]
        owned_index = torch.arange(n_owned, device=pos.device)

        node_e0 = model.atomic_energies_fn(node_attrs[:n_owned])[owned_index, owned_heads]
        node_feats = model.node_embedding(node_attrs)
        edge_attrs = model.spherical_harmonics(vectors)
        edge_feats, cutoff = model.radial_embedding(lengths, node_attrs, edge_index, model.atomic_numbers)
        if hasattr(model, "pair_repulsion"):
            pair_node_energy = model.pair_repulsion_fn(lengths, node_attrs, edge_index, model.atomic_numbers)[:n_owned]
        else:
            pair_node_energy = torch.zeros_like(node_e0)

        node_es_list = [pair_node_energy]
        node_feats_list: List[Tensor] = []
        layer = 0
        for interaction, product in zip(self.layer_interactions, model.products):
            if layer > 0:
                node_feats = torch.ops.dsmd.halo_exchange(node_feats, n_ghost)
            # the owned rows only (see _OwnedRowsInteraction); the skip connection is None for the non-residual blocks (the first
            # layer of the MPA and OMAT models, ...)
            mixed, skip = interaction(node_attrs, node_feats, edge_attrs, edge_feats, edge_index, cutoff, n_owned)
            node_feats = product(node_feats=mixed, sc=skip, node_attrs=node_attrs[:n_owned])
            node_feats_list.append(node_feats)
            layer += 1

        for i, readout in enumerate(model.readouts):
            feat_idx = -1 if len(model.readouts) == 1 else i
            node_es_list.append(readout(node_feats_list[feat_idx], owned_heads)[owned_index, owned_heads])
        node_inter_es = torch.sum(torch.stack(node_es_list, dim=0), dim=0)
        if hasattr(model, "scale_shift"):
            node_inter_es = model.scale_shift(node_inter_es, owned_heads)
        node_energy = (node_e0 + node_inter_es).to(pos.dtype)
        return torch.cat([node_energy, torch.zeros(n_ghost, dtype=pos.dtype, device=pos.device)])

    def forward_with_shifts(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor) -> Tensor:
        # mace: vectors = positions[receiver] - positions[sender] + shifts, with the Cartesian shifts given directly
        n_atoms = pos.shape[0]
        n_edges = edge_index.shape[1]
        data: Dict[str, Tensor] = {
            "positions": pos,
            "node_attrs": self.species_one_hot.to(pos.dtype)[species],
            "edge_index": edge_index,
            "shifts": edge_shift,
            "unit_shifts": torch.zeros((n_edges, 3), dtype=pos.dtype, device=pos.device),
            "cell": torch.zeros((3, 3), dtype=pos.dtype, device=pos.device),
            "batch": torch.zeros(n_atoms, dtype=torch.long, device=pos.device),
            "ptr": torch.arange(2, dtype=torch.long, device=pos.device) * n_atoms,
            "head": self.head,
        }
        out = self.model(data, training=False, compute_force=False)
        node_energy = out["node_energy"]
        assert node_energy is not None
        return node_energy.to(pos.dtype)
