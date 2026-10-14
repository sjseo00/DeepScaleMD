"""toy_pair: a pure-torch pair potential for testing the whole pipeline without any MLIP package.

phi(r) = 4 eps [(sigma/r)^12 - (sigma/r)^6] * 0.5 (cos(pi r / rc) + 1)   for r < rc, else 0
eps = sqrt(eps_a eps_b), sigma = (sigma_a + sigma_b) / 2 (Lorentz-Berthelot). Energies are in eV.
Atom i gets half of phi over its edges (i -> j); with both edge directions present the sum over atoms is the sum
over pairs, and the owned-atom sum splits correctly across domains (owned-ghost pairs count half on each side).
It is strictly local, so num_layers = 1 and the ghost depth is the cutoff.
"""
import math
from typing import Sequence

import torch
from torch import Tensor

from .base import MLIPAdapter

KCAL_PER_MOL_TO_EV = 1.0 / 23.0609

# Lennard-Jones-like parameters per atomic number: (epsilon in eV, sigma in A). Argon is the LJ example of
# examples/dsMD_input_lj (0.238 kcal/mol, 3.405 A); the others are only plausible-looking values for tests.
TOY_PAIR_PARAMS = {
    1: (0.0030, 2.40),
    8: (0.0065, 3.10),
    18: (0.238 * KCAL_PER_MOL_TO_EV, 3.405),
    36: (0.0140, 3.65),
}
DEFAULT_PARAMS = (0.0100, 3.00)


class ToyPairAdapter(MLIPAdapter):
    model_family = "toy_pair"
    model_name = "toy_pair"
    num_layers = 1
    supports_shifts = True

    def __init__(self, elements: Sequence[int] = (18,),
                 epsilon: Sequence[float] = (0.238 * KCAL_PER_MOL_TO_EV,),
                 sigma: Sequence[float] = (3.405,),
                 cutoff: float = 8.0):
        """Defaults are argon (LJ eps 0.238 kcal/mol, sigma 3.405 A) as in examples/dsMD_input_lj."""
        super().__init__(list(elements))
        if not (len(epsilon) == len(sigma) == len(self.elements)):
            raise ValueError("epsilon and sigma need one entry per element")
        if cutoff <= 0.0:
            raise ValueError("cutoff must be positive")
        self.cutoff = float(cutoff)
        # Indexed by the species index (position in `elements`).
        self.register_buffer("epsilon", torch.tensor(list(epsilon), dtype=torch.float64))
        self.register_buffer("sigma", torch.tensor(list(sigma), dtype=torch.float64))

    @classmethod
    def from_elements(cls, elements: Sequence[int], cutoff: float = 8.0) -> "ToyPairAdapter":
        """Per-element parameters from TOY_PAIR_PARAMS (DEFAULT_PARAMS for an element that is not listed)."""
        params = [TOY_PAIR_PARAMS.get(int(z), DEFAULT_PARAMS) for z in elements]
        return cls(elements=elements, epsilon=[p[0] for p in params], sigma=[p[1] for p in params], cutoff=cutoff)

    def forward(self, pos: Tensor, species: Tensor, edge_index: Tensor) -> Tensor:
        center = edge_index[0]
        neighbor = edge_index[1]
        vec = pos[neighbor] - pos[center]
        r = torch.sqrt((vec * vec).sum(dim=1))  # Tensor.norm(dim=...) does not script

        eps = torch.sqrt(self.epsilon[species[center]] * self.epsilon[species[neighbor]]).to(pos.dtype)
        sigma = (0.5 * (self.sigma[species[center]] + self.sigma[species[neighbor]])).to(pos.dtype)
        s6 = (sigma / r) ** 6
        smooth = 0.5 * (torch.cos(math.pi * r / self.cutoff) + 1.0)
        smooth = torch.where(r < self.cutoff, smooth, torch.zeros_like(smooth))
        pair_energy = 4.0 * eps * (s6 * s6 - s6) * smooth

        atom_energy = torch.zeros(pos.shape[0], dtype=pos.dtype, device=pos.device)
        return atom_energy.index_add(0, center, 0.5 * pair_energy)

    def forward_with_shifts(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor) -> Tensor:
        center = edge_index[0]
        neighbor = edge_index[1]
        vec = pos[neighbor] - pos[center] + edge_shift
        r = torch.sqrt((vec * vec).sum(dim=1))

        eps = torch.sqrt(self.epsilon[species[center]] * self.epsilon[species[neighbor]]).to(pos.dtype)
        sigma = (0.5 * (self.sigma[species[center]] + self.sigma[species[neighbor]])).to(pos.dtype)
        s6 = (sigma / r) ** 6
        smooth = 0.5 * (torch.cos(math.pi * r / self.cutoff) + 1.0)
        smooth = torch.where(r < self.cutoff, smooth, torch.zeros_like(smooth))
        pair_energy = 4.0 * eps * (s6 * s6 - s6) * smooth

        atom_energy = torch.zeros(pos.shape[0], dtype=pos.dtype, device=pos.device)
        return atom_energy.index_add(0, center, 0.5 * pair_energy)
