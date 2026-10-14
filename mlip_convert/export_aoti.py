"""AOTInductor export of the standard wrapper (design 3.3, 11.2).

The wrapper cannot go to torch.export directly: its force is torch.autograd.grad inside forward, and export either refuses
it (strict) or leaves a fake-tensor constant in the backward graph (non-strict), which breaks for any input but the example.
Nor does make_fx on the model's Python code work for every model: MACE under make_fx gets the number of edges baked in as a
constant. So the export has two phases:
  1. the energy part alone (adapter, owned-atom sum; no autograd) is exported with torch.export, strict mode, dynamic N
     (atoms) and E (edges) -> a plain aten graph;
  2. make_fx traces "force = -d(owned energy)/d(pos)" over that graph with symbolic shapes, which unfolds the backward (the
     way NequIP does it); the result is exported again with dynamic N and E and compiled to a .pt2 package.
The finished package is loaded again and compared with the eager wrapper on examples of different sizes.

The package is compiled for one device: on a GPU it holds Triton-compiled kernels for that architecture (metadata
`target_gpu`, e.g. sm_90), which needs compute capability 7.0 or higher. A CPU package (`target_gpu: cpu`) is meant for
tests; the MD backend runs on the GPU and refuses it.
"""
import contextlib
from pathlib import Path
from typing import Callable, Sequence, Tuple

import torch
from torch import Tensor

from .wrapper import MDWrapper

Inputs = Tuple[Tensor, Tensor, Tensor, Tensor]     # pos, atomic_numbers, edge_index, n_owned (design 3.1)

MAX_SIZE = 2 ** 31 - 2      # upper bound of N and E; without one, export adds a guard against int64 max that it then rejects


class AOTIExportError(RuntimeError):
    pass


@contextlib.contextmanager
def tracing_friendly_e3nn():
    """e3nn scripts the code it generates for tensor products with TorchScript by default, and a scripted module cannot be
    traced by torch.export; this turns that off while the model is built and exported (what NequIP does for its AOTI export)."""
    try:
        import e3nn
    except ImportError:
        yield
        return
    previous = e3nn.get_optimization_defaults()["jit_script_fx"]
    e3nn.set_optimization_defaults(jit_script_fx=False)
    try:
        yield
    finally:
        e3nn.set_optimization_defaults(jit_script_fx=previous)


def target_of(device: torch.device) -> str:
    """The `target_gpu` metadata value: 'cpu' or 'sm_<major><minor>'."""
    if device.type == "cpu":
        return "cpu"
    if device.type != "cuda":
        raise AOTIExportError(f"device '{device}' is not supported: use cpu or cuda")
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}"


class EnergyPart(torch.nn.Module):
    """The wrapper without the force: (pos, atomic_numbers, edge_index, n_owned) -> (owned energy [1], atom_energy [N])."""

    def __init__(self, wrapper: MDWrapper):
        super().__init__()
        self.adapter = wrapper.adapter
        self.register_buffer("z_to_index", wrapper.z_to_index.clone())

    def forward(self, pos: Tensor, atomic_numbers: Tensor, edge_index: Tensor, n_owned: Tensor):
        species = self.z_to_index[atomic_numbers]
        atom_energy = self.adapter(pos, species, edge_index)
        owned = torch.arange(pos.shape[0], device=pos.device) < n_owned
        energy = torch.where(owned, atom_energy, torch.zeros_like(atom_energy)).sum()
        return energy.reshape(1), atom_energy


class WithForce(torch.nn.Module):
    """The MDWrapper interface over an exported energy graph: forces by autograd, to be unfolded by make_fx."""

    def __init__(self, energy_graph: Callable):
        super().__init__()
        self.energy_graph = energy_graph

    def forward(self, pos: Tensor, atomic_numbers: Tensor, edge_index: Tensor, n_owned: Tensor):
        torch.set_grad_enabled(True)
        pos = pos.detach().requires_grad_(True)
        energy, atom_energy = self.energy_graph(pos, atomic_numbers, edge_index, n_owned)
        grad = torch.autograd.grad([energy.sum()], [pos])[0]
        return energy, atom_energy.detach(), -grad


def _dynamic_shapes():
    from torch.export import Dim
    n_atoms, n_edges = Dim("N", min=2, max=MAX_SIZE), Dim("E", min=1, max=MAX_SIZE)
    return ({0: n_atoms}, {0: n_atoms}, {1: n_edges}, {})


def export_energy(wrapper: MDWrapper, inputs: Inputs) -> torch.fx.GraphModule:
    """Phase 1: the energy part as an aten graph, shapes dynamic."""
    try:
        exported = torch.export.export(EnergyPart(wrapper).eval(), inputs, dynamic_shapes=_dynamic_shapes(), strict=True)
    except Exception as error:      # noqa: BLE001
        raise AOTIExportError(f"torch.export could not trace the model's energy with dynamic sizes: {error}") from error
    return exported.module()


def unfold_backward(energy_graph: Callable, inputs: Inputs) -> torch.fx.GraphModule:
    """Phase 2: forces by autograd over the energy graph, traced into a plain aten FX graph; sizes stay symbolic."""
    from torch._decomp import core_aten_decompositions
    from torch.fx.experimental.proxy_tensor import make_fx

    # use_duck_shape=False: otherwise N or E that happens to equal another dimension of the example (a weight of that size)
    # is specialized to a constant.
    with torch.fx.experimental._config.patch(use_duck_shape=False):
        graph = make_fx(WithForce(energy_graph), decomposition_table=core_aten_decompositions(), tracing_mode="symbolic",
                        _allow_non_fake_inputs=True, _error_on_data_dependent_ops=True)(*[t.clone() for t in inputs])
    graph.graph.eliminate_dead_code()       # the symbolic trace leaves unused size()/stride() nodes
    graph.recompile()
    return graph


def _to_device(inputs: Inputs, device: torch.device, dtype: torch.dtype) -> Inputs:
    pos, z, edges, n_owned = inputs
    return pos.to(device=device, dtype=dtype), z.to(device), edges.to(device), n_owned.to(device)


def _tolerances(dtype: torch.dtype) -> Tuple[float, float]:
    """(relative energy tolerance, force tolerance relative to the largest force)."""
    return (1e-9, 1e-9) if dtype == torch.float64 else (1e-4, 1e-3)


def export_dynamic(graph: torch.fx.GraphModule, inputs: Inputs) -> torch.export.ExportedProgram:
    return torch.export.export(graph, inputs, dynamic_shapes=_dynamic_shapes())


def compile_aoti(exported: torch.export.ExportedProgram, path: Path) -> str:
    try:
        return torch._inductor.aoti_compile_and_package(exported, package_path=str(path))
    except Exception as error:      # noqa: BLE001  (torch raises a specific class that is not importable on every version)
        if type(error).__name__ == "GPUTooOldForTriton":
            raise AOTIExportError(f"{error}\nA GPU package cannot be built on this machine; convert on a node with a GPU of "
                                  f"compute capability 7.0 or higher, or use --device cpu for a CPU test package") from error
        raise


def check_against_eager(model: Callable, wrapper: MDWrapper, examples: Sequence[Inputs], dtype: torch.dtype, what: str) -> None:
    rel_energy, rel_force = _tolerances(dtype)
    for index, example in enumerate(examples):
        with torch.no_grad():
            expected = wrapper(*example)
            got = model(*example)
        scale = max(float(expected[2].abs().max()), 1e-6)
        energy_error = abs(float(got[0] - expected[0])) / max(1.0, abs(float(expected[0])))
        force_error = float((got[2] - expected[2]).abs().max()) / scale
        if energy_error > rel_energy or force_error > rel_force:
            raise AOTIExportError(f"{what} disagrees with the eager wrapper on example {index} (N={example[0].shape[0]}, "
                                  f"E={example[2].shape[1]}): relative energy error {energy_error:.2e} (limit {rel_energy:.0e}), "
                                  f"relative force error {force_error:.2e} (limit {rel_force:.0e})")


def export_aoti(wrapper: MDWrapper, path, examples: Sequence[Inputs], *, device: str = "cuda",
                dtype: torch.dtype = torch.float64, log: Callable[[str], None] = lambda text: None) -> str:
    """Export and compile `wrapper` for `device` into the package `path` (.pt2). The graph is traced with the first example;
    `examples` need inputs of at least two different sizes (N and E): every stage is compared with the eager wrapper on all of
    them, which fails when a size was baked into the graph (a constant for the size of the first example) and so would be
    wrong for other structures. Returns the `target_gpu` value."""
    if len(examples) < 2:
        raise AOTIExportError("export_aoti needs examples of two different sizes to detect size specialization")
    target_device = torch.device(device)
    target = target_of(target_device)
    wrapper = wrapper.eval().to(target_device)
    inputs = [_to_device(example, target_device, dtype) for example in examples]
    path = Path(path)
    if path.exists():
        path.unlink()

    log(f"exporting the energy part ({target})")
    energy_graph = export_energy(wrapper, inputs[0])
    log("unfolding the backward")
    graph = unfold_backward(energy_graph, inputs[0])
    check_against_eager(graph, wrapper, inputs, dtype, "the traced graph")
    exported = export_dynamic(graph, inputs[0])
    check_against_eager(exported.module(), wrapper, inputs, dtype, "the exported program")

    log("compiling the AOTInductor package (this takes a while)")
    compile_aoti(exported, path)

    log("checking the package against the eager wrapper")
    check_against_eager(torch._inductor.aoti_load_package(str(path)), wrapper, inputs, dtype, "the compiled package")
    return target
