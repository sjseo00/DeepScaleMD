"""MDWrapper: the standard interface the C++ MD code calls (design 3.1-3.3).

inputs   pos [N, 3] (owned first, then ghosts, ghost positions already shifted), atomic_numbers [N] int64,
         edge_index [2, E] int64 (both directions, only pairs within cutoff), n_owned [] int64
outputs  energy [1] (sum over owned atoms), atom_energy [N], forces [N, 3] = -d energy / d pos (ghosts included)

The forces are taken from the owned-atom energy sum only. A model's own force output differentiates the energy of
every atom including ghosts, which gives wrong forces under domain decomposition, so adapters return per-atom
energies and this wrapper does the differentiation.
"""
from typing import Sequence, Tuple

import torch
from torch import Tensor

from .adapters.base import MLIPAdapter

MAX_ATOMIC_NUMBER = 118


def make_z_to_index(elements: Sequence[int]) -> Tensor:
    """int64 [MAX_ATOMIC_NUMBER + 1]: atomic number -> model species index, -1 for unsupported elements."""
    table = torch.full((MAX_ATOMIC_NUMBER + 1,), -1, dtype=torch.int64)
    for index, z in enumerate(elements):
        if not 1 <= z <= MAX_ATOMIC_NUMBER:
            raise ValueError(f"atomic number out of range: {z}")
        table[z] = index
    return table


class MDWrapper(torch.nn.Module):
    def __init__(self, adapter: MLIPAdapter):
        super().__init__()
        self.adapter = adapter
        # Atomic numbers outside the table are rejected by the C++ side at startup (type -> Z check against
        # metadata `elements`); they would index -1 here.
        self.register_buffer("z_to_index", make_z_to_index(adapter.elements))

    def forward(self, pos: Tensor, atomic_numbers: Tensor, edge_index: Tensor,
                n_owned: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # The caller may run under no_grad (or C++ NoGradGuard); the force needs a graph, so turn grad mode on
        # and restore it afterwards. Function-style calls: TorchScript supports neither `with torch.enable_grad()`
        # nor `with torch.set_grad_enabled(...)`.
        previous_grad_mode = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        pos = pos.detach().requires_grad_(True)
        species = self.z_to_index[atomic_numbers]
        atom_energy = self.adapter(pos, species, edge_index)
        # Owned-atom sum as a mask instead of atom_energy[:n_owned]: no data-dependent slice, no host
        # synchronization, and it traces/exports with a dynamic N.
        owned = torch.arange(pos.shape[0], device=pos.device) < n_owned
        energy = torch.where(owned, atom_energy, torch.zeros_like(atom_energy)).sum()
        grads = torch.autograd.grad([energy], [pos], allow_unused=True)
        grad = grads[0]
        # (If/else instead of reassigning `grad`: TorchScript keeps Optional[Tensor] after a one-sided if.)
        if grad is None:
            force = torch.zeros_like(pos)
        else:
            force = -grad
        torch.set_grad_enabled(previous_grad_mode)
        return energy.reshape(1), atom_energy.detach(), force


class MDShiftWrapper(torch.nn.Module):
    """MDWrapper for models that take periodic images as edge shifts instead of as extra atoms (edge_shifts in metadata.json).

    inputs   pos [N, 3] (the atoms themselves: no periodic images), atomic_numbers [N] int64, edge_index [2, E] int64,
             edge_shift [E, 3] (the vector of edge (i, j) is pos[j] - pos[i] + edge_shift; zero for an edge to an atom
             that is really there, a lattice vector for one across a periodic boundary), n_owned [] int64
    outputs  energy [1], atom_energy [N], forces [N, 3], virial [1]

    The virial is W = sum_i r_i . F_i over the atoms the model saw, plus the shift term, which is what the sum over the
    atoms of the extended system (images included) would give: -dE/de with the positions and the shifts scaled by (1 + e).
    It has the same meaning as the MD side's sum of r.F over atoms and ghosts for a model without shifts, so the pressure
    follows from it the same way.
    """
    def __init__(self, adapter: MLIPAdapter):
        super().__init__()
        if not adapter.supports_shifts:
            raise ValueError(f"the {adapter.model_family} adapter does not take edge shifts")
        self.adapter = adapter
        self.register_buffer("z_to_index", make_z_to_index(adapter.elements))

    def forward(self, pos: Tensor, atomic_numbers: Tensor, edge_index: Tensor, edge_shift: Tensor,
                n_owned: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # grad mode: see MDWrapper.forward
        previous_grad_mode = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        pos = pos.detach().requires_grad_(True)
        strain = torch.zeros((), dtype=pos.dtype, device=pos.device).requires_grad_(True)
        scale = 1.0 + strain
        species = self.z_to_index[atomic_numbers]
        atom_energy = self.adapter.forward_with_shifts(pos * scale, species, edge_index, edge_shift * scale)
        owned = torch.arange(pos.shape[0], device=pos.device) < n_owned
        energy = torch.where(owned, atom_energy, torch.zeros_like(atom_energy)).sum()
        grads = torch.autograd.grad([energy], [pos, strain], allow_unused=True)
        grad = grads[0]
        grad_strain = grads[1]
        if grad is None:
            force = torch.zeros_like(pos)
        else:
            force = -grad
        if grad_strain is None:
            virial = torch.zeros(1, dtype=pos.dtype, device=pos.device)
        else:
            virial = -grad_strain.reshape(1)
        torch.set_grad_enabled(previous_grad_mode)
        return energy.reshape(1), atom_energy.detach(), force, virial
