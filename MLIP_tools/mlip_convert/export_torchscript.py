"""TorchScript export of the standard wrapper (design 1, 3.3).

The wrapper is scripted, not traced: N and E change every step and the owned mask depends on n_owned, which a trace
would freeze. The saved file needs no Python and no mlip_convert at run time.
"""
import re
from pathlib import Path
from typing import Dict

import torch

from . import halo_op, kspace_op
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


def _reason(error: Exception) -> str:
    """The line of an error that says what went wrong. A TorchScript run-time error ends with the Python call stack of the
    interpreter (its last line is then something like "<frozen runpy>(198): _run_module_as_main"), so the last line that names an
    exception class is taken."""
    lines = [line.strip() for line in str(error).strip().splitlines() if line.strip()]
    named = [line for line in lines if re.match(r"^[A-Za-z_.]*(Error|Exception)\b", line)]
    return named[-1] if named else (lines[-1] if lines else repr(error))


def check_torchscript(path, reference: Dict, dtype: torch.dtype, device: str = "cpu", edge_shifts: bool = False) -> None:
    """Load the saved file again and run it on the reference structure (the dict that becomes reference.json): it has to give
    the eager wrapper's energy and forces. A model that scripts without error can still fail or be wrong when it runs
    (cuequivariance's scripted tensor products do, 11.3), and the saved file is what the MD code runs."""
    what = "the exported TorchScript model"
    try:
        model = torch.jit.load(str(path), map_location=device)
    except Exception as error:      # noqa: BLE001  (the TorchScript interpreter raises its own error classes)
        raise RuntimeError(f"{what} fails when it is run on the reference structure ({device}): {_reason(error)}") from error
    check_model(model, reference, dtype, device=device, edge_shifts=edge_shifts, what=what)


def check_model(model, reference: Dict, dtype: torch.dtype, device: str = "cpu", edge_shifts: bool = False,
                what: str = "the model") -> None:
    """Runs `model` (a wrapper: scripted, or the Python one of a --backend pytorch folder) on the reference structure and compares
    its energy and forces with the eager wrapper's (reference.json)."""
    try:
        pos = torch.tensor(reference["pos"], dtype=dtype, device=device)
        z = torch.tensor(reference["atomic_numbers"], dtype=torch.int64, device=device)
        edges = torch.tensor(reference["edge_index"], dtype=torch.int64, device=device)
        n_owned = torch.tensor(reference["n_owned"], dtype=torch.int64, device=device)
        # --kspace: the reciprocal sums of the owned atoms in the reference box
        with kspace_op.maybe_context(reference.get("kspace"), reference["box"]):
            if reference.get("layer_exchange"):
                # the ghosts are copies of owned atoms of the reference structure: the exchange is a copy inside this process
                with halo_op.local_ghosts(torch.tensor(reference["ghost_owner"], dtype=torch.int64)):
                    energy, _, forces, _ = model(pos, z, edges, torch.zeros((edges.shape[1], 3), dtype=dtype, device=device), n_owned)
            elif edge_shifts:      # the reference structure has its periodic images as atoms: every edge shift is zero
                energy, _, forces, _ = model(pos, z, edges, torch.zeros((edges.shape[1], 3), dtype=dtype, device=device), n_owned)
            else:
                energy, _, forces = model(pos, z, edges, n_owned)
    except Exception as error:      # noqa: BLE001  (the TorchScript interpreter raises its own error classes)
        raise RuntimeError(f"{what} fails when it is run on the reference structure ({device}): {_reason(error)}") from error
    expected_energy = torch.tensor(reference["energy"], dtype=torch.float64)
    expected_forces = torch.tensor(reference["forces"], dtype=torch.float64)
    rel_energy, rel_force = (1e-9, 1e-8) if dtype == torch.float64 else (1e-4, 1e-3)
    energy_error = abs(float(energy.double().cpu().reshape(-1)[0] - expected_energy[0])) / max(1.0, abs(float(expected_energy[0])))
    force_error = float((forces.double().cpu() - expected_forces).abs().max()) / max(float(expected_forces.abs().max()), 1e-6)
    if not (energy_error <= rel_energy and force_error <= rel_force):
        raise RuntimeError(f"{what} does not reproduce the eager wrapper ({device}): relative energy "
                           f"error {energy_error:.2e} (limit {rel_energy:.0e}), relative force error {force_error:.2e} "
                           f"(limit {rel_force:.0e})")
