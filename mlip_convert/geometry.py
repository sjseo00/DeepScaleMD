"""Brute-force structure helpers: the edge list of design 4 and periodic ghost atoms (owned first, ghosts after).

They are deliberately simple (O(N^2), no cell lists) because they build small reference structures for the
converter's verification and the tests; the MD code builds its edges on the GPU (EdgeBuilder).
"""
import math
from typing import Tuple

import torch
from torch import Tensor


def brute_force_edges(pos: Tensor, cutoff: float) -> Tensor:
    """[2, E] int64: every ordered pair (i, j) with i != j and |pos_j - pos_i| < cutoff (both directions)."""
    n = pos.shape[0]
    distance = (pos[None, :, :] - pos[:, None, :]).norm(dim=-1)
    mask = (distance < cutoff) & ~torch.eye(n, dtype=torch.bool, device=pos.device)
    i, j = mask.nonzero(as_tuple=True)
    return torch.stack([i, j])


def periodic_ghosts(pos_owned: Tensor, box: Tensor, depth: float) -> Tuple[Tensor, Tensor]:
    """Periodic images of the owned atoms that lie outside the box but within `depth` of it (the box expanded by
    `depth` in every direction, corners included, like a halo exchange).

    pos_owned [N, 3] must lie in [0, box). box [3] are the orthorhombic box lengths.
    Returns (ghost_pos [G, 3], ghost_owner [G] int64): the owned index each ghost is an image of."""
    shells = int(math.ceil(depth / float(box.min())))
    positions, owners = [], []
    for sx in range(-shells, shells + 1):
        for sy in range(-shells, shells + 1):
            for sz in range(-shells, shells + 1):
                if sx == 0 and sy == 0 and sz == 0:
                    continue
                shifted = pos_owned + torch.tensor([sx, sy, sz], dtype=pos_owned.dtype) * box
                keep = ((shifted >= -depth) & (shifted <= box + depth)).all(dim=1)
                positions.append(shifted[keep])
                owners.append(keep.nonzero(as_tuple=True)[0])
    if not positions:
        return pos_owned.new_zeros((0, 3)), torch.zeros(0, dtype=torch.int64)
    return torch.cat(positions), torch.cat(owners)


def fold_ghost_forces(forces: Tensor, n_owned: int, ghost_owner: Tensor) -> Tensor:
    """What the MD reverse communication does: add each ghost's force to the owned atom it is an image of.
    forces [n_owned + G, 3] -> [n_owned, 3]."""
    folded = forces[:n_owned].clone()
    folded.index_add_(0, ghost_owner, forces[n_owned:])
    return folded
