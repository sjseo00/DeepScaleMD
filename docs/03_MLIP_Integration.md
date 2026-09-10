# MLIP Integration Architecture

## Overview

DeepScaleMD provides two independent MLIP integration paths, selected by `force_calculation.mode`.

| Path | Configuration | Characteristics |
|------|------|------|
| **libtorch mode** | `force_calculation.mode: libtorch` | TorchScript only, fully GPU-resident, best performance |
| **python mode** | `force_calculation.mode: python` | Supports all ASE calculators, including fairchem/MACE/NequIP |

---

## libtorch mode

`libtorch` mode runs a TorchScript model from C++ with no Python interpreter involved at all. What actually makes this GPU-resident and fast is a split between a one-time compilation step at startup and a per-step execution that never allocates or copies through the host.

### Startup: the YAML spec compiles into a fixed execution plan

`MlipRuntime::initialize()` parses `spec_file` (the `model_spec.yaml` below) once and calls `compile_execution_plan()`, which turns the declarative YAML into a concrete, reusable plan:

- Every entry under `inputs:`/`outputs:` is resolved to a **pointer-resolver callback** bound to a real MD device buffer (registered by the MD-side data-source bindings — `md.positions`, `md.species`, `md.neighbor_list`, `md.forces`, `md.energy`, etc.), not a name looked up again every step.
- Each preprocessing/postprocessing op named in the YAML (`species_map`, `full_directed_edge_list`, `unit_conversion`, `accumulate_force`, ...) is resolved once to an actual device-kernel implementation.
- Unit-conversion scale factors (e.g. eV → kcal/mol) are computed once from the `units:` block, not recomputed per step.
- `PyTorchBackend` loads the TorchScript module (`torch::jit::load(config.model_path, device)`) at this point too.

Nothing about this plan changes between MD steps; only the buffer contents and per-step particle/edge counts do.

### Per step: `MlipRuntime::compute()`, entirely device-side

1. **Preprocessing** (`run_preprocessing`) walks the resolved input bindings, fetches the current device pointer from each pointer-resolver, and runs its preprocessing ops (e.g. mapping MD atom types to atomic numbers via a device lookup table, converting the half neighbor list to a full directed edge list) as CUDA kernels writing into pre-sized scratch buffers. These buffers are allocated once and only grown (never shrunk or reallocated per step) when a step's particle/edge count exceeds the current capacity.
2. **Forward pass** (`PyTorchBackend::execute`): when `share_cuda_stream: true`, the MD step's own `cudaStream_t` is wrapped in an `at::cuda::CUDAStreamGuard`, making it PyTorch's *current* stream for the duration of `model.forward(...)` — the model's internal kernels are enqueued after the preprocessing kernels on the same stream, with no explicit cross-stream synchronization needed. Autograd is deliberately left enabled here (not wrapped in a `NoGradGuard`): the supported TorchScript models differentiate the predicted energy internally via `torch.autograd.grad` to produce forces, so disabling autograd would break force computation for those models.
3. **Postprocessing** (`run_postprocessing`) applies each output's ops (unit conversion, then `accumulate_force`) and writes the result directly into the MD force/energy device buffers that the pointer-resolvers were bound to at startup — there is no intermediate host buffer.

The net effect: adding a new TorchScript model only ever means writing a new `model_spec.yaml` — no new C++ preprocessing/postprocessing code, and no host round-trip is introduced anywhere in this path.

### MLIP Specification YAML Schema

`model_spec.yaml` (example: `examples/mlip_runtime/mock_lj_spec.yaml`):

```yaml
schema_version: 1

model:
  name: my_model
  backend: pytorch          # pytorch | onnx | tensorrt
  path: path/to/model.pt    # path to the TorchScript file
  device: cuda
  precision: float32

backend:
  format: torchscript
  call_style: dictionary    # positional | dictionary
  share_cuda_stream: true
  allow_cpu_fallback: false

inputs:
  positions:
    source: md.positions    # MD data source name
    tensor_name: pos        # model input key name
    shape: [nall, 3]
    dtype: float32
  species:
    source: md.species
    tensor_name: atomic_numbers
    shape: [nall]
    dtype: int64
    transform:
      - species_map          # MD type → atomic number conversion
  edge_index:
    source: md.neighbor_list
    tensor_name: edge_index
    shape: [2, nedges]
    dtype: int64
    transform:
      - full_directed_edge_list  # half list → full directed list

outputs:
  energy:
    tensor_name: energy
    target: md.energy
    operations:
      - unit_conversion     # eV → kcal/mol
  forces:
    tensor_name: forces
    target: md.forces
    sign: 1.0
    includes_ghost_atoms: true
    operations:
      - unit_conversion
      - accumulate_force

neighbor:
  cutoff: 6.0
  skin: 1.0
  list: full
  directed: true

species_map:
  input: md_type
  output: model_index
  values:
    0: 8    # MD type 0 (O) → atomic number 8
    1: 1    # MD type 1 (H) → atomic number 1

units:
  md:    { length: angstrom, energy: kcal/mol }
  model: { length: angstrom, energy: eV }

capabilities:
  energy: true
  forces: true
  virial: false
  external_neighbor_list: true
  external_cuda_stream: true
```

### GPU-resident Data Flow

```
ForceCalculatorCUDA (GPU)
│
├─ d_x[], d_y[], d_z[]          ← SoA positions (float64)
├─ d_edge_coo_packed[]          ← neighbor list COO (int32)
│
▼ [CUDA kernel: launch_interleave_positions]
│
├─ d_pos_interleaved[N,3]       ← AoS positions (model input format)
│
▼ [CUDA kernel: launch_half_to_full_edges]
│
├─ d_edge_index[2, E_full]      ← directed edges
│
▼ [PyTorchBackend::execute() — LibTorch CUDA forward]
│
├─ model output: energy (scalar)
├─ model output: forces [N, 3]
│
▼ [CUDA kernel: launch_accumulate_forces_scaled]
│
└─ d_fx[], d_fy[], d_fz[]       ← force accumulation complete
```

---

## python mode

### Design Goals

- Allow arbitrary ML calculators to be used without recompiling the C++ binary
- Support fairchem UMA, MACE, NequIP, CHGNet, SevenNet, ORB
- Share the ASE (Atomic Simulation Environment) calculator interface

### Architecture

```
bin/md_simulation_cuda  (C++)
    │
    │  PythonForceInterface (CPython Embedding)
    │      initialize_python("python_force")
    │      initialize_ml_model(model_path, model_type, ...)
    │            ↓
    │      ml_forces.initialize_ml_forcefield_generic(...)
    │            ↓
    │      mlip_interface.initialize(model_type=..., ...)
    │            ↓
    │      ADAPTER_REGISTRY[key]()  →  UMAAdapter / MACEAdapter / ...
    │
    │  Every MD step:
    │      compute_ml_forces(positions, types, cell, pbc)
    │            ↓
    │      mlip_interface.compute_forces(...)
    │            ↓
    │      adapter.compute_forces(...)  →  ASE atoms.get_forces()
    │
    └─ Force array returned → copied into C++ GPU memory
```

### Python-C++ Interface Data Transfer

```cpp
// src/force_python/python_interface.cpp
// Positions: shared as a NumPy array (no copy)
npy_intp pos_dims[2] = {n_particles, 3};
PyObject* py_positions = PyArray_SimpleNewFromData(
    2, pos_dims, NPY_DOUBLE, particle_data.x);

// Cell: 1D array [box_x, box_y, box_z]
cell_data[0] = box_x;
cell_data[1] = box_y;
cell_data[2] = box_z;

// PBC: int32 array [1,1,1]
pbc_data[i] = pbc_x ? 1 : 0;
```

### UMAAdapter (fairchem UMA only)

```python
# python_force/mlip_interface.py

class UMAAdapter(MLIPAdapter):
    def initialize(self, model_path, task_name="omat", ...):
        # external_graph_gen=True is required:
        # if False, r_edges=False, which causes a "No edges found" error
        inf_settings = InferenceSettings(external_graph_gen=True)
        predict_unit = MLIPPredictUnit(model_path, device=self.device,
                                       inference_settings=inf_settings)
        self._calculator = FAIRChemCalculator(predict_unit, task_name=task)

    def compute_forces(self, positions, atom_types, cell, pbc):
        self._sync_before()          # CUDA synchronize (wait for C++ GPU work to finish)
        self._update_atoms(...)      # update ASE Atoms positions
        forces = self._atoms.get_forces()
        self._sync_after()           # CUDA synchronize (wait for Python GPU work to finish)
        return forces
```

**Why `external_graph_gen=True` is needed:**
- When `False` (the default), `FAIRChemCalculator` calls `AtomicData.from_ase` with `r_edges=False`
- In that case the model's internal graph builder does not run, producing a "No edges found" error
- Setting it to `True` makes `r_edges=True`, so the radius graph is generated externally via ASE

### List of Supported Adapters

```python
# python_force/mlip_interface.py
ADAPTER_REGISTRY = {
    "mace":     MACEAdapter,    # MACE-MP-0, MACE-OFF
    "mace_mp":  MACEAdapter,
    "nequip":   NequIPAdapter,  # NequIP deployed model
    "chgnet":   CHGNetAdapter,  # CHGNet v0.x
    "sevennet": SevenNetAdapter,# SevenNet
    "7net":     SevenNetAdapter,
    "orb":      ORBAdapter,     # ORB models
    "uma":      UMAAdapter,     # fairchem UMA-S/M
    "uma_s":    UMAAdapter,
    "fairchem": UMAAdapter,
}
```

### GPU Memory Optimization (MLIPAdapter Base Class)

```python
class MLIPAdapter:
    def _ensure_gpu_resources(self, atom_types):
        # 1. Atom type GPU tensor — uploaded only on the first step, reused afterward
        if self._atom_types_gpu is None:
            self._atom_types_gpu = torch.tensor(atom_types, device=self.device)

        # 2. Pinned-memory staging buffer — accelerates PCIe DMA
        if self._pos_buf is None:
            self._pos_buf = torch.empty((n,3), dtype=torch.float64,
                                        pin_memory=True).numpy()

        # 3. Dedicated CUDA stream — overlaps H2D transfers
        if self._transfer_stream is None:
            self._transfer_stream = torch.cuda.Stream(device=self.device)
```

---

## Path Comparison

| Item | libtorch mode | python mode |
|------|-----------------|-----------------|
| **Supported model format** | TorchScript (.pt) | Any ASE calculator |
| **GPU-resident** | ✅ Fully GPU-resident | ⚠️ Goes through CPU (positions) |
| **C++ recompilation** | Not required (only YAML changes) | Not required |
| **Adding a new model** | Write a `model_spec.yaml` | Add a class to `ADAPTER_REGISTRY` |
| **fairchem UMA** | ❌ (TorchScript not supported) | ✅ |
| **MACE-MP** | ✅ (export required) | ✅ |
| **Speed** | Best | Same as the model's forward pass speed |

---

## How to Add a New Model

### libtorch mode

1. Export the model to TorchScript
2. Write a `model_spec.yaml` (input/output mapping, unit definitions)
3. In `input.yaml`, set `spec_file: path/to/model_spec.yaml`

### python mode

1. Add an adapter class to `python_force/mlip_interface.py`:
```python
class MyModelAdapter(MLIPAdapter):
    def initialize(self, model_path, ...): ...
    def compute_forces(self, positions, atom_types, cell, pbc): ...
```
2. Register it in `ADAPTER_REGISTRY`:
```python
ADAPTER_REGISTRY["mymodel"] = MyModelAdapter
```
3. `input.yaml`:
```yaml
force_calculation:
  mode: python
mlip_config:
  model_type: mymodel
  model_path: path/to/model.pt
```
