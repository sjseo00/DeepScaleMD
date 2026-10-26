"""Base class of the model adapters (design 3.4).

An adapter is a torch.nn.Module with forward(pos, species, edge_index) -> atom_energy. It does the model-specific
work: build the model's input dict from the standard input, match the edge direction convention, put zero shifts
where the model wants shifts, and pull the per-atom energies (with E0 and scale/shift applied) out of the output.
It must NOT take forces from the model and must not reduce over atoms: MDWrapper sums the owned atoms and
differentiates that sum.
"""
from typing import List

import torch
from torch import Tensor

from .. import halo_op as _halo_op   # noqa: F401  (registers the op dsmd::halo_exchange that layered adapters call)


class MLIPAdapter(torch.nn.Module):
    # The property below is plain Python for the converter; TorchScript must not try to compile it.
    __jit_unused_properties__ = ["ghost_cutoff", "reference_elements"]

    # Filled in by the subclass; the converter copies them into metadata.json.
    model_family: str = ""
    model_name: str = ""
    cutoff: float = 0.0         # neighbor cutoff (A) the edge list must be built with
    num_layers: int = 1         # message-passing layers; 1 for strictly local models
    accel: str = "e3nn"         # "e3nn" | "cueq" | "openeq"
    head_name = None            # name of the fixed model head, if the model has several (metadata `head`)
    e3nn_compile = False        # export runs e3nn.util.jit.compile before scripting (models built from e3nn modules)
    fixed_dtype_name = None     # "float32" | "float64" when the model cannot be converted to another dtype after it is built
    supports_shifts = False     # True when forward_with_shifts is implemented: periodic images can be given as edge shifts
    # Row of edge_index (0 = i, 1 = j) onto which the messages / edge energies of the model are summed: the atom an edge feeds.
    # None = not known. With a row, the MD leaves out the edges whose receiver lies more than (num_layers - 1) cutoffs outside
    # the owned atoms of a rank: they cannot change the energy of an owned atom (the energies of the last layer's receivers are
    # the owned atoms', and each earlier layer needs receivers one cutoff further out), which saves 25-60 % of the edges of a
    # rank with ghost atoms. Set it only for a model whose layers all use this edge list and cutoff, that sums onto this one
    # row, and whose message normalization does not depend on how many edges a sender has; and check it: forces on N ranks must
    # equal those on one (tests/mlip/test_mlip_domains.cpp, the 1 rank vs N rank runs in docs/mlip_integration_design.md).
    edge_receiver_row = None
    # True when forward_layered is implemented: the model can exchange node features between its message-passing layers
    # (halo_op.py), so that a rank needs ghost atoms one cutoff deep instead of num_layers cutoffs. Needs edge_receiver_row and
    # supports_shifts. `layer_exchange` is switched on by the converter (--layer-exchange) for the adapter instance.
    supports_layer_exchange = False
    layer_exchange = False

    def __init__(self, elements: List[int]):
        super().__init__()
        if len(set(elements)) != len(elements):
            raise ValueError(f"duplicate atomic numbers in elements: {elements}")
        # Atomic numbers in the model's own species order; the index in this list is the `species` value.
        self.elements: List[int] = [int(z) for z in elements]

    def enable_layer_exchange(self) -> None:
        """Switches layer exchange on for this instance (the converter calls it for --layer-exchange); an adapter that needs
        something built for its forward_layered extends it."""
        self.layer_exchange = True

    def package_versions(self):
        """Versions of the MLIP packages this adapter uses, for metadata.json (the C++ side compares torch and cuda)."""
        return {}

    @property
    def reference_elements(self):
        """Elements the reference structure (reference.json) is made of: all of them unless a model says otherwise."""
        return list(self.elements)

    @property
    def ghost_cutoff(self) -> float:
        """Ghost depth (A) that makes owned-atom results exact: num_layers * cutoff (design 5.1); with layer exchange one cutoff."""
        return self.cutoff if self.layer_exchange else self.num_layers * self.cutoff

    def forward(self, pos: Tensor, species: Tensor, edge_index: Tensor) -> Tensor:
        """pos [N, 3], species [N] int64 (model species index), edge_index [2, E] int64 (design 4)
        -> atom_energy [N], same dtype and device as pos."""
        raise NotImplementedError

    def forward_layered(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor, n_owned: int) -> Tensor:
        """forward_with_shifts for a model that exchanges node features between its layers. The atoms are n_owned owned ones
        and then their ghost copies (one cutoff deep); edge_index has only the edges whose receiver (edge_receiver_row) is an
        owned atom. Every layer after the first starts from
            torch.ops.dsmd.halo_exchange(h_owned, n_ghost)
        and only the owned rows are ever computed. Returns the per-atom energies [N], zero for the ghost atoms. Only adapters
        with supports_layer_exchange = True implement it."""
        raise NotImplementedError

    def forward_with_shifts(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor) -> Tensor:
        """Like forward, with a Cartesian vector added to every edge: the vector of edge (i, j) is
        pos[j] - pos[i] + edge_shift[e] (edge_shift [E, 3]). A neighbor across a periodic boundary is then the atom itself
        plus a lattice vector, not a second copy of it. Only adapters with supports_shifts = True implement it."""
        raise NotImplementedError
