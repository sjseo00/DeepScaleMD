"""reference.json: a small verification structure (owned + ghost atoms) with the inputs and the expected outputs.

The converter writes it from the eager wrapper; the C++ tests (TorchScriptRunner, LibTorchBackend) load the model and
compare against it (design 3.5, 9.2). The ghost layer is `ghost_cutoff` deep, so the owned-atom results are exact.
"""
import json
from pathlib import Path
from typing import Dict, NamedTuple

import torch
from torch import Tensor

from .adapters.base import MLIPAdapter
from .geometry import brute_force_edges, fold_ghost_forces, periodic_ghosts
from . import halo_op
from .wrapper import MDLayerShiftWrapper, MDShiftWrapper, MDWrapper

FORMAT_VERSION = 1


class ReferenceStructure(NamedTuple):
    pos: Tensor              # [N, 3] float64: n_owned owned atoms in [0, box)^3, then their periodic images (ghosts)
    atomic_numbers: Tensor   # [N]
    edge_index: Tensor       # [2, E]
    n_owned: int
    ghost_owner: Tensor      # [N - n_owned]: the owned atom each ghost is an image of
    box: float               # cubic box length


def make_reference_structure(adapter: MLIPAdapter, n_owned: int = 12, seed: int = 0, min_distance: float = 2.8,
                             density: float = 0.03, extra_depth: float = 0.0) -> ReferenceStructure:
    """Random periodic structure of n_owned atoms plus its ghost layer.

    Atomic numbers cycle through the adapter's elements so every element appears. `extra_depth` thickens the ghost
    layer beyond ghost_cutoff (tests use it to show that the layer is deep enough)."""
    box_length = max(7.0, (n_owned / density) ** (1.0 / 3.0))
    box = torch.full((3,), box_length, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)

    points = []
    for _ in range(100000):
        if len(points) == n_owned:
            break
        candidate = torch.rand(3, generator=generator, dtype=torch.float64) * box_length
        ok = True
        for q in points:
            d = candidate - q
            d = d - box_length * torch.round(d / box_length)
            if float(d @ d) < min_distance ** 2:
                ok = False
                break
        if ok:
            points.append(candidate)
    if len(points) < n_owned:
        raise RuntimeError(f"could not place {n_owned} atoms {min_distance} A apart in a {box_length:.1f} A box")
    pos_owned = torch.stack(points)

    elements = torch.tensor(adapter.reference_elements, dtype=torch.int64)
    z_owned = elements[torch.arange(n_owned) % len(elements)]

    ghost_pos, owner = periodic_ghosts(pos_owned, box, adapter.ghost_cutoff + extra_depth)
    pos = torch.cat([pos_owned, ghost_pos])
    z = torch.cat([z_owned, z_owned[owner]])
    edges = brute_force_edges(pos, adapter.cutoff)
    if adapter.layer_exchange:
        # Layer exchange: only the owned atoms receive edges (a ghost never runs a layer, it gets its features from its owner).
        edges = edges[:, edges[adapter.edge_receiver_row] < n_owned]
    return ReferenceStructure(pos, z, edges, n_owned, owner, box_length)


def make_reference(wrapper, adapter: MLIPAdapter, dtype: torch.dtype, device: str = "cpu", **structure_options) -> Dict:
    """`device`: where the wrapper is (a model with GPU kernels is built on the GPU); the results are brought back to the CPU."""
    s = make_reference_structure(adapter, **structure_options)
    pos = s.pos.to(dtype).to(device)
    z = s.atomic_numbers.to(device)
    edges = s.edge_index.to(device)
    n_owned = torch.tensor(s.n_owned).to(device)
    layered = isinstance(wrapper, MDLayerShiftWrapper)
    with torch.no_grad():
        if layered:
            # The ghosts are copies of owned atoms of this same structure: the exchange is a copy inside this process.
            shift = torch.zeros((edges.shape[1], 3), dtype=dtype, device=device)
            with halo_op.local_ghosts(s.ghost_owner):
                energy, atom_energy, forces, _ = wrapper(pos, z, edges, shift, n_owned)
        elif isinstance(wrapper, MDShiftWrapper):    # periodic images are atoms here, so every edge shift is zero
            shift = torch.zeros((edges.shape[1], 3), dtype=dtype, device=device)
            energy, atom_energy, forces, _ = wrapper(pos, z, edges, shift, n_owned)
        else:
            energy, atom_energy, forces = wrapper(pos, z, edges, n_owned)
    energy, atom_energy, forces = energy.cpu(), atom_energy.cpu(), forces.cpu()
    folded = fold_ghost_forces(forces.double(), s.n_owned, s.ghost_owner)   # what the reverse communication gives
    extra = {"layer_exchange": True, "ghost_owner": s.ghost_owner.tolist()} if layered else {}
    return {
        **extra,
        "format_version": FORMAT_VERSION,
        "dtype": str(dtype).replace("torch.", ""),
        "n_owned": int(s.n_owned),
        "n_atoms": int(s.pos.shape[0]),
        "box": [s.box, s.box, s.box],             # the owned atoms are one periodic cell of this box
        "pos": s.pos.tolist(),                    # float64; the model input is pos cast to dtype
        "atomic_numbers": s.atomic_numbers.tolist(),
        "edge_index": s.edge_index.tolist(),
        "energy": energy.double().tolist(),       # [1]
        "atom_energy": atom_energy.double().tolist(),
        "forces": forces.double().tolist(),       # on owned and ghost atoms
        "forces_folded": folded.tolist(),         # [n_owned, 3]: ghost forces added to the atoms they are images of
    }


def write_reference(path, reference: Dict) -> None:
    Path(path).write_text(json.dumps(reference) + "\n", encoding="utf-8")


def load_reference(path) -> Dict[str, Tensor]:
    """reference.json as tensors (pos/energy/atom_energy/forces float64, atomic_numbers/edge_index int64)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "dtype": getattr(torch, raw["dtype"]),
        "n_owned": raw["n_owned"],
        "box": raw["box"],
        "forces_folded": torch.tensor(raw["forces_folded"], dtype=torch.float64),
        "pos": torch.tensor(raw["pos"], dtype=torch.float64),
        "atomic_numbers": torch.tensor(raw["atomic_numbers"], dtype=torch.int64),
        "edge_index": torch.tensor(raw["edge_index"], dtype=torch.int64).reshape(2, -1),
        "energy": torch.tensor(raw["energy"], dtype=torch.float64),
        "atom_energy": torch.tensor(raw["atom_energy"], dtype=torch.float64),
        "forces": torch.tensor(raw["forces"], dtype=torch.float64),
    }
