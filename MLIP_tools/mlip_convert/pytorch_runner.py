"""Runs a model folder converted with --backend pytorch inside dsMD (mlip_backend: pytorch; metadata "backend": "pytorch").

Such a folder has no model file. metadata.json keeps what the converter built the wrapper from (`python_model`: architecture,
model file, dtype, kernels, layer exchange), and dsMD's embedded Python builds the same wrapper again with the converter's own code
(cli.build_wrapper) and calls it every step as Python, without TorchScript. src/PyTorchModelRunner.cpp calls
    load(folder, device_index) -> PyTorchModel          once, when dsMD starts
    PyTorchModel.run(pos, atomic_numbers, edge_index, [edge_shift,] n_owned) -> (energy, atom_energy, forces[, virial])
every step. The tensors are DLPack capsules of dsMD's GPU arrays (shared, not copied) and n_owned is an int; the results go back as
capsules too. The operator dsmd::halo_exchange that a layer-exchange model calls between its layers is dsMD's own (HaloExchangeOp.cpp,
with dsMD's halo transport): halo_op.py does not register its Python version in that process.
GPU memory (UMA, eSEN: use_edge_chunks, edge_memory): before the first step the runner switches the adapter's low-memory mode on
when the adapter's estimate of the memory of this rank's edges is more than is free. A step that still runs out of GPU memory is run
again once with the mode on, and the mode stays on; not for a layer-exchange model, whose other ranks would be in the middle of their
exchanges (it stops with an error). Any other out-of-memory error goes to dsMD unchanged.
"""
import gc
import sys
import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.dlpack import to_dlpack


def build_from_recipe(recipe: Dict, device: str) -> torch.nn.Module:
    """The wrapper described by metadata `python_model`, on `device`, in eval mode."""
    from .cli import build_wrapper
    _, wrapper, _, _, _ = build_wrapper(
        recipe["arch"], backend="pytorch", dtype=recipe["dtype"], elements=recipe.get("elements"), cutoff=recipe.get("cutoff"),
        seed=recipe.get("seed", 0), model=recipe.get("model"), head=recipe.get("head"), device=device,
        accel=recipe.get("accel", "e3nn"), edge_shifts=recipe["edge_shifts"], layer_exchange=recipe["layer_exchange"],
        options=recipe.get("options"))
    return wrapper.to(device).eval()


class PyTorchModel:
    """The wrapper of a --backend pytorch folder, called by dsMD with DLPack capsules."""

    def __init__(self, wrapper: torch.nn.Module):
        self.wrapper = wrapper
        self.memory_checked = False

    def choose_edge_chunks(self, adapter, tensors) -> bool:
        """Before the first step: the edge chunks go on when the edges of this rank would need more GPU memory than is free (the
        adapter's estimate, x1.5 for the backward pass). Each rank decides for itself; the chunks change no data between ranks.
        False when there was nothing to estimate yet (dsMD's warm-up calls have two edges): then the next call checks again."""
        pos, atomic_numbers, edge_index = tensors[:3]
        if pos.device.type != "cuda":
            return True
        edge_shift = tensors[3] if len(tensors) > 3 else torch.zeros((edge_index.shape[1], 3), dtype=pos.dtype, device=pos.device)
        per_edge = adapter.edge_memory(pos, self.wrapper.z_to_index[atomic_numbers], edge_index, edge_shift)
        if per_edge <= 0.0:
            return False
        free = torch.cuda.mem_get_info(pos.device)[0] + torch.cuda.memory_reserved(pos.device) - torch.cuda.memory_allocated(pos.device)
        need = 1.5 * per_edge * edge_index.shape[1]
        if need > free:
            adapter.use_edge_chunks(True)
            sys.stderr.write(f"[MLIP] {edge_index.shape[1]} edges need about {need / 2**30:.1f} GB of GPU memory, {free / 2**30:.1f} GB are "
                             f"free: the model runs its edges in chunks of {adapter.edge_chunk} and computes them again for the forces "
                             "(less memory, slower)" + chr(10))
        return True

    def run(self, *inputs):
        # inputs: capsules of pos, atomic_numbers, edge_index (and edge_shift), then n_owned as an int. n_owned stays a CPU tensor,
        # as in the TorchScript call (the wrapper reads it with int(): no device synchronization).
        tensors = [torch.from_dlpack(capsule) for capsule in inputs[:-1]]
        n_owned = torch.tensor(int(inputs[-1]), dtype=torch.int64)
        adapter = getattr(self.wrapper, "adapter", None)
        if not self.memory_checked and hasattr(adapter, "edge_memory"):
            self.memory_checked = self.choose_edge_chunks(adapter, tensors)
        try:
            outputs = self.wrapper(*tensors, n_owned)
        except torch.OutOfMemoryError:
            if not hasattr(adapter, "use_edge_chunks") or adapter.uses_edge_chunks:
                raise
            if getattr(adapter, "layer_exchange", False):
                # the other ranks may be in the middle of their exchanges: a step run again here would pair its first exchange with
                # their later ones (same sizes), so the numbers would be wrong without an error
                raise RuntimeError("out of GPU memory in a layer-exchange model, although the memory estimate before the first step "
                                   "said the edges fit: use more GPUs (fewer atoms per GPU)") from None
            outputs = None
        if outputs is None:
            # Retried here, not in the except block: there the error's traceback still holds the tensors of the failed step.
            adapter.use_edge_chunks(True)
            gc.collect()
            torch.cuda.empty_cache()
            sys.stderr.write(f"[MLIP] out of GPU memory with {tensors[2].shape[1]} edges: the model now runs its edges in chunks of "
                             f"{adapter.edge_chunk} and computes them again for the forces (less memory, slower); it stays so" + chr(10))
            outputs = self.wrapper(*tensors, n_owned)
        return tuple(to_dlpack(t.detach().contiguous()) for t in outputs)


def load(folder: str, device_index: int) -> PyTorchModel:
    meta = json.loads((Path(folder) / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("backend") != "pytorch" or "python_model" not in meta:
        raise ValueError(f"{folder}: not a model folder converted with --backend pytorch (metadata backend '{meta.get('backend')}')")
    return PyTorchModel(build_from_recipe(meta["python_model"], f"cuda:{int(device_index)}"))
