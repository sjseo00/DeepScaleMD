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

    def __init__(self, elements: List[int]):
        super().__init__()
        if len(set(elements)) != len(elements):
            raise ValueError(f"duplicate atomic numbers in elements: {elements}")
        # Atomic numbers in the model's own species order; the index in this list is the `species` value.
        self.elements: List[int] = [int(z) for z in elements]

    def package_versions(self):
        """Versions of the MLIP packages this adapter uses, for metadata.json (the C++ side compares torch and cuda)."""
        return {}

    @property
    def reference_elements(self):
        """Elements the reference structure (reference.json) is made of: all of them unless a model says otherwise."""
        return list(self.elements)

    @property
    def ghost_cutoff(self) -> float:
        """Ghost depth (A) that makes owned-atom results exact: num_layers * cutoff (design 5.1)."""
        return self.num_layers * self.cutoff

    def forward(self, pos: Tensor, species: Tensor, edge_index: Tensor) -> Tensor:
        """pos [N, 3], species [N] int64 (model species index), edge_index [2, E] int64 (design 4)
        -> atom_energy [N], same dtype and device as pos."""
        raise NotImplementedError

    def forward_with_shifts(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor) -> Tensor:
        """Like forward, with a Cartesian vector added to every edge: the vector of edge (i, j) is
        pos[j] - pos[i] + edge_shift[e] (edge_shift [E, 3]). A neighbor across a periodic boundary is then the atom itself
        plus a lattice vector, not a second copy of it. Only adapters with supports_shifts = True implement it."""
        raise NotImplementedError
