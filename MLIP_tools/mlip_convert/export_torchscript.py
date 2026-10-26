"""TorchScript export of the standard wrapper (design 1, 3.3).

The wrapper is scripted, not traced: N and E change every step and the owned mask depends on n_owned, which a trace
would freeze. The saved file needs no Python and no mlip_convert at run time.
"""
from pathlib import Path
from typing import Dict

import torch

from . import halo_op
from .wrapper import MDWrapper


def export_torchscript(wrapper: MDWrapper, path) -> torch.jit.ScriptModule:
    wrapper = wrapper.eval()
    if getattr(wrapper.adapter, "e3nn_compile", False):
        # Models built from e3nn modules (MACE): e3nn traces or scripts each of its modules the way it was written to be
        # compiled; torch.jit.script alone cannot. The children are compiled in place, then the wrapper is scripted.
        from e3nn.util import jit
        wrapper = jit.compile(wrapper)
    scripted = torch.jit.script(wrapper)
    scripted.save(str(Path(path)))
    return scripted


def check_torchscript(path, reference: Dict, dtype: torch.dtype, device: str = "cpu", edge_shifts: bool = False) -> None:
    """Load the saved file again and run it on the reference structure (the dict that becomes reference.json): it has to give
    the eager wrapper's energy and forces. A model that scripts without error can still fail or be wrong when it runs
    (cuequivariance's scripted tensor products do, 11.3), and the saved file is what the MD code runs."""
    try:
        model = torch.jit.load(str(path), map_location=device)
        pos = torch.tensor(reference["pos"], dtype=dtype, device=device)
        z = torch.tensor(reference["atomic_numbers"], dtype=torch.int64, device=device)
        edges = torch.tensor(reference["edge_index"], dtype=torch.int64, device=device)
        n_owned = torch.tensor(reference["n_owned"], dtype=torch.int64, device=device)
        if reference.get("layer_exchange"):
            # the ghosts are copies of owned atoms of the reference structure: the exchange is a copy inside this process
            with halo_op.local_ghosts(torch.tensor(reference["ghost_owner"], dtype=torch.int64)):
                energy, _, forces, _ = model(pos, z, edges, torch.zeros((edges.shape[1], 3), dtype=dtype, device=device), n_owned)
        elif edge_shifts:      # the reference structure has its periodic images as atoms: every edge shift is zero
            energy, _, forces, _ = model(pos, z, edges, torch.zeros((edges.shape[1], 3), dtype=dtype, device=device), n_owned)
        else:
            energy, _, forces = model(pos, z, edges, n_owned)
    except Exception as error:      # noqa: BLE001  (the TorchScript interpreter raises its own error classes)
        raise RuntimeError(f"the exported TorchScript model fails when it is run on the reference structure ({device}): "
                           f"{str(error).strip().splitlines()[-1] if str(error).strip() else error}") from error
    expected_energy = torch.tensor(reference["energy"], dtype=torch.float64)
    expected_forces = torch.tensor(reference["forces"], dtype=torch.float64)
    rel_energy, rel_force = (1e-9, 1e-8) if dtype == torch.float64 else (1e-4, 1e-3)
    energy_error = abs(float(energy.double().cpu().reshape(-1)[0] - expected_energy[0])) / max(1.0, abs(float(expected_energy[0])))
    force_error = float((forces.double().cpu() - expected_forces).abs().max()) / max(float(expected_forces.abs().max()), 1e-6)
    if not (energy_error <= rel_energy and force_error <= rel_force):
        raise RuntimeError(f"the exported TorchScript model does not reproduce the eager wrapper ({device}): relative energy "
                           f"error {energy_error:.2e} (limit {rel_energy:.0e}), relative force error {force_error:.2e} "
                           f"(limit {rel_force:.0e})")
