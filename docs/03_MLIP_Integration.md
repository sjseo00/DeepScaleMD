# MLIP Integration Architecture

## Overview

DeepScaleMD provides two independent MLIP integration paths.

| Path | Configuration | Characteristics |
|------|------|------|
| **C++ MLIP Runtime** | `force_calculation.mode: libtorch` | TorchScript only, GPU-resident, best performance |
| **Python Interface** | `force_calculation.mode: python` | Supports all ASE calculators, including fairchem/MACE/NequIP |

---

## Path 1: C++ MLIP Runtime (mlip_inst.md architecture)

### Design Goals

- Run TorchScript models directly in C++ without a Python interpreter
- Keep data resident in GPU memory (eliminate GPU → CPU → GPU round trips)
- Share the CUDA stream between DeepScaleMD and LibTorch

### Layer Structure

```
input.yaml
  └─ force_calculation.mode: libtorch
  └─ spec_file: examples/xxx/model_spec.yaml   ← MLIP specification YAML
                 │
                 ▼
         MlipRuntime::initialize()
                 │
         ┌───────────────────┐
         │  MlipModelSpec    │  — produced by the YAML parser
         │  (parsed spec)    │
         └───────────────────┘
                 │
         ┌───────────────────┐
         │  ExecutionPlan    │  — compiles the input/output pipeline
         │  (compiled plan)  │
         └───────────────────┘
                 │
         ┌───────────────────┐
         │  PyTorchBackend   │  — torch::jit::load() + forward()
         │  (model engine)   │
         └───────────────────┘
```

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

### Unit Conversion (UnitConversion)

Computed at compile time in `execution_plan.cpp`:

```cpp
uc.energy_scale = e_model / e_md;          // eV → kcal/mol: × 0.04336
uc.force_scale  = (e_model/l_model) / (e_md/l_md);
```

### Key File List

| File | Role |
|------|------|
| `src/force_libtorch/runtime/mlip_runtime.cpp` | Runtime entry point, initialization/execution |
| `src/force_libtorch/runtime/execution_plan.cpp` | Pipeline compilation, unit conversion |
| `src/force_libtorch/config/yaml_parser.cpp` | MLIP specification YAML parser |
| `src/force_libtorch/backend/pytorch_backend.cpp` | LibTorch wrapper (compiled with ABI=0) |
| `src/force_libtorch/backend/pytorch_backend_register.cpp` | Backend auto-registration (ABI=1) |
| `src/force_libtorch/binding/md_data_sources.cpp` | MD data source/target registration |
| `src/force_libtorch/cuda/edge_conversion.cu` | Edge conversion CUDA kernel |
| `src/force_libtorch/cuda/force_accumulation.cu` | Force accumulation CUDA kernel |
| `include/force_libtorch/runtime/model_specification.h` | Specification data structure definitions |
| `include/force_libtorch/backend/backend_interface.h` | Backend abstract interface + registry |

---

## Path 2: Python Interface

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

| Item | C++ MLIP Runtime | Python Interface |
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

### C++ MLIP Runtime Path

1. Export the model to TorchScript
2. Write a `model_spec.yaml` (input/output mapping, unit definitions)
3. In `input.yaml`, set `spec_file: path/to/model_spec.yaml`

### Python Interface Path

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
