"""toy_mp: a two-layer message-passing toy model in pure torch (no MLIP package).

It exists to test what pair potentials cannot: with num_layers = 2 the energy of an atom depends on atoms up to two
cutoffs away, so a domain needs ghosts 2 * cutoff deep for exact owned-atom results (design 5.1).

    h0_i  = embedding[species_i]
    layer l = 0, 1:   m_ij = (rbf(r_ij) W_rbf[l]) * (h_j W_node[l]) * envelope(r_ij)     for each edge i <- j
                      h_i  = h_i + tanh(sum_j m_ij)
    E_i   = h2_i . readout + bias[species_i]

Weights come from a seeded generator, so a given (elements, cutoff, hidden, seed) always gives the same model.
Everything depends on distances only, so the energy is invariant under rotation and translation.
"""
import math
from typing import Sequence

import torch
from torch import Tensor

from .base import MLIPAdapter

NUM_LAYERS = 2


class ToyMPAdapter(MLIPAdapter):
    model_family = "toy_mp"
    model_name = "toy_mp"
    num_layers = NUM_LAYERS
    supports_shifts = True

    def __init__(self, elements: Sequence[int] = (18,), cutoff: float = 5.0, hidden: int = 8,
                 num_rbf: int = 6, seed: int = 0):
        super().__init__(list(elements))
        if cutoff <= 0.0:
            raise ValueError("cutoff must be positive")
        self.cutoff = float(cutoff)
        self.rbf_width = self.cutoff / num_rbf

        generator = torch.Generator().manual_seed(seed)

        def normal(*shape, scale):
            return (torch.randn(*shape, generator=generator, dtype=torch.float64) * scale)

        s = len(self.elements)
        self.register_buffer("rbf_centers", torch.linspace(0.0, self.cutoff, num_rbf, dtype=torch.float64))
        self.register_buffer("embedding", normal(s, hidden, scale=0.5))
        self.register_buffer("w_rbf", normal(NUM_LAYERS, num_rbf, hidden, scale=1.0))
        self.register_buffer("w_node", normal(NUM_LAYERS, hidden, hidden, scale=2.0 / math.sqrt(hidden)))
        self.register_buffer("readout", normal(hidden, scale=0.5))
        self.register_buffer("bias", normal(s, scale=0.1))

    def forward(self, pos: Tensor, species: Tensor, edge_index: Tensor) -> Tensor:
        center = edge_index[0]
        neighbor = edge_index[1]
        vec = pos[neighbor] - pos[center]
        r = torch.sqrt((vec * vec).sum(dim=1))  # Tensor.norm(dim=...) does not script

        envelope = 0.5 * (torch.cos(math.pi * r / self.cutoff) + 1.0)
        envelope = torch.where(r < self.cutoff, envelope, torch.zeros_like(envelope))
        rbf = torch.exp(-((r.unsqueeze(1) - self.rbf_centers.to(pos.dtype).unsqueeze(0)) / self.rbf_width) ** 2)

        h = self.embedding.to(pos.dtype)[species]
        for layer in range(self.w_rbf.shape[0]):
            radial = rbf @ self.w_rbf[layer].to(pos.dtype)
            message = radial * (h[neighbor] @ self.w_node[layer].to(pos.dtype)) * envelope.unsqueeze(1)
            aggregated = torch.zeros_like(h).index_add(0, center, message)
            h = h + torch.tanh(aggregated)

        return h @ self.readout.to(pos.dtype) + self.bias.to(pos.dtype)[species]

    def forward_with_shifts(self, pos: Tensor, species: Tensor, edge_index: Tensor, edge_shift: Tensor) -> Tensor:
        center = edge_index[0]
        neighbor = edge_index[1]
        vec = pos[neighbor] - pos[center] + edge_shift
        r = torch.sqrt((vec * vec).sum(dim=1))

        envelope = 0.5 * (torch.cos(math.pi * r / self.cutoff) + 1.0)
        envelope = torch.where(r < self.cutoff, envelope, torch.zeros_like(envelope))
        rbf = torch.exp(-((r.unsqueeze(1) - self.rbf_centers.to(pos.dtype).unsqueeze(0)) / self.rbf_width) ** 2)

        h = self.embedding.to(pos.dtype)[species]
        for layer in range(self.w_rbf.shape[0]):
            radial = rbf @ self.w_rbf[layer].to(pos.dtype)
            message = radial * (h[neighbor] @ self.w_node[layer].to(pos.dtype)) * envelope.unsqueeze(1)
            aggregated = torch.zeros_like(h).index_add(0, center, message)
            h = h + torch.tanh(aggregated)

        return h @ self.readout.to(pos.dtype) + self.bias.to(pos.dtype)[species]
