"""Angle (three-body) index lists from the standard edge list, with torch operations (design 3.4: "three-body models build the
angle (triplet) list from edge_index with torch operations").

Three-body models (CHGNet, M3GNet/MatterSim) work on pairs of bonds that share an atom. Their own converters build the pairs
from a periodic structure with a neighbor search; here the neighbor pairs are given (the symmetric directed edge list of the
standard input), and the pairs of edges follow from them.
"""
from typing import Tuple

import torch
from torch import Tensor


def ordered_edge_pairs(center: Tensor, length: Tensor, cutoff: float, n_atoms: int) -> Tuple[Tensor, Tensor]:
    """Every ordered pair (left, right) of different directed edges that have the same center atom, both shorter than `cutoff`.

    center [E] int64: the center atom (edge_index[0]) of every directed edge, length [E]: the edge lengths.
    Returns two int64 tensors [P]: the indices of the left and the right edge of every pair; an atom with k edges inside the
    cutoff gives k (k - 1) pairs. The order of the pairs is not part of the contract (the models sum over them).
    """
    short = torch.nonzero(length < cutoff).reshape(-1)                      # directed edges inside the cutoff
    short_center = center[short]
    order = torch.argsort(short_center)                                      # group the edges by center
    short = short[order]
    short_center = short_center[order]
    counts = torch.bincount(short_center, minlength=n_atoms)                 # edges inside the cutoff per center atom
    group_start = torch.cumsum(counts, 0) - counts                           # position of the first edge of every group
    own_start = group_start[short_center]
    partners = counts[short_center] - 1                                      # the other edges of the same group
    left_position = torch.repeat_interleave(torch.arange(short.shape[0], device=center.device), partners)
    first_row = torch.cumsum(partners, 0) - partners
    rank = torch.arange(left_position.shape[0], device=center.device) - torch.repeat_interleave(first_row, partners)
    within_group = left_position - own_start[left_position]                  # position of the left edge in its group
    right_position = own_start[left_position] + rank + (rank >= within_group).to(rank.dtype)   # skip the left edge itself
    return short[left_position], short[right_position]
