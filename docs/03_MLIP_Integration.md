# MLIP 연동 아키텍처

## 개요

DeepScaleMD는 두 가지 독립적인 MLIP 연동 경로를 제공한다.

| 경로 | 설정 | 특징 |
|------|------|------|
| **C++ MLIP Runtime** | `force_calculation.mode: mlip_runtime` | TorchScript 전용, GPU-resident, 최고 성능 |
| **Python Interface** | `force_calculation.mode: python` | fairchem/MACE/NequIP 등 모든 ASE 계산기 지원 |

---

## 경로 1: C++ MLIP Runtime (mlip_inst.md 아키텍처)

### 설계 목표

- Python 인터프리터 없이 C++에서 직접 TorchScript 모델 실행
- 데이터가 GPU 메모리에 상주 (GPU → CPU → GPU 왕복 제거)
- CUDA 스트림을 DeepScaleMD와 LibTorch가 공유

### 계층 구조

```
input.yaml
  └─ force_calculation.mode: mlip_runtime
  └─ spec_file: examples/xxx/model_spec.yaml   ← MLIP 명세 YAML
                 │
                 ▼
         MlipRuntime::initialize()
                 │
         ┌───────────────────┐
         │  MlipModelSpec    │  — YAML 파서가 생성
         │  (parsed spec)    │
         └───────────────────┘
                 │
         ┌───────────────────┐
         │  ExecutionPlan    │  — 입출력 파이프라인 컴파일
         │  (compiled plan)  │
         └───────────────────┘
                 │
         ┌───────────────────┐
         │  PyTorchBackend   │  — torch::jit::load() + forward()
         │  (model engine)   │
         └───────────────────┘
```

### MLIP 명세 YAML 스키마

`model_spec.yaml` (예: `examples/mlip_runtime/mock_lj_spec.yaml`):

```yaml
schema_version: 1

model:
  name: my_model
  backend: pytorch          # pytorch | onnx | tensorrt
  path: path/to/model.pt    # TorchScript 파일 경로
  device: cuda
  precision: float32

backend:
  format: torchscript
  call_style: dictionary    # positional | dictionary
  share_cuda_stream: true
  allow_cpu_fallback: false

inputs:
  positions:
    source: md.positions    # MD 데이터 소스 이름
    tensor_name: pos        # 모델 입력 키 이름
    shape: [nall, 3]
    dtype: float32
  species:
    source: md.species
    tensor_name: atomic_numbers
    shape: [nall]
    dtype: int64
    transform:
      - species_map          # MD 타입 → 원자번호 변환
  edge_index:
    source: md.neighbor_list
    tensor_name: edge_index
    shape: [2, nedges]
    dtype: int64
    transform:
      - full_directed_edge_list  # 반목록 → 전방향 목록

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

### GPU-resident 데이터 흐름

```
ForceCalculatorCUDA (GPU)
│
├─ d_x[], d_y[], d_z[]          ← SoA 위치 (float64)
├─ d_edge_coo_packed[]          ← 반목록 COO (int32)
│
▼ [CUDA 커널: launch_interleave_positions]
│
├─ d_pos_interleaved[N,3]       ← AoS 위치 (모델 입력 형식)
│
▼ [CUDA 커널: launch_half_to_full_edges]
│
├─ d_edge_index[2, E_full]      ← 전방향 엣지
│
▼ [PyTorchBackend::execute() — LibTorch CUDA forward]
│
├─ model output: energy (scalar)
├─ model output: forces [N, 3]
│
▼ [CUDA 커널: launch_accumulate_forces_scaled]
│
└─ d_fx[], d_fy[], d_fz[]       ← 힘 누적 완료
```

### 단위 변환 (UnitConversion)

`execution_plan.cpp`에서 컴파일 시 계산:

```cpp
uc.energy_scale = e_model / e_md;          // eV → kcal/mol: × 0.04336
uc.force_scale  = (e_model/l_model) / (e_md/l_md);
```

### 핵심 파일 목록

| 파일 | 역할 |
|------|------|
| `src/mlip/runtime/mlip_runtime.cpp` | 런타임 진입점, 초기화/실행 |
| `src/mlip/runtime/execution_plan.cpp` | 파이프라인 컴파일, 단위 변환 |
| `src/mlip/config/yaml_parser.cpp` | MLIP 명세 YAML 파서 |
| `src/mlip/backend/pytorch_backend.cpp` | LibTorch 래퍼 (ABI=0으로 컴파일) |
| `src/mlip/backend/pytorch_backend_register.cpp` | 백엔드 자동 등록 (ABI=1) |
| `src/mlip/binding/md_data_sources.cpp` | MD 데이터 소스/타겟 등록 |
| `src/potential/mlip/cuda/edge_conversion.cu` | 엣지 변환 CUDA 커널 |
| `src/potential/mlip/cuda/force_accumulation.cu` | 힘 누적 CUDA 커널 |
| `include/mlip/runtime/model_specification.h` | 명세 데이터 구조 정의 |
| `include/mlip/backend/backend_interface.h` | 백엔드 추상 인터페이스 + 레지스트리 |

---

## 경로 2: Python Interface

### 설계 목표

- C++ 바이너리 재컴파일 없이 임의의 ML 계산기 사용 가능
- fairchem UMA, MACE, NequIP, CHGNet, SevenNet, ORB 지원
- ASE(Atomic Simulation Environment) 계산기 인터페이스 공유

### 아키텍처

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
    │  매 MD 스텝:
    │      compute_ml_forces(positions, types, cell, pbc)
    │            ↓
    │      mlip_interface.compute_forces(...)
    │            ↓
    │      adapter.compute_forces(...)  →  ASE atoms.get_forces()
    │
    └─ 힘 배열 반환 → C++ GPU 메모리에 복사
```

### Python-C++ 인터페이스 데이터 전달

```cpp
// src/python_interface.cpp
// 위치: NumPy 배열로 공유 (복사 없음)
npy_intp pos_dims[2] = {n_particles, 3};
PyObject* py_positions = PyArray_SimpleNewFromData(
    2, pos_dims, NPY_DOUBLE, particle_data.x);

// Cell: 1D 배열 [box_x, box_y, box_z]
cell_data[0] = box_x;
cell_data[1] = box_y;
cell_data[2] = box_z;

// PBC: int32 배열 [1,1,1]
pbc_data[i] = pbc_x ? 1 : 0;
```

### UMAAdapter (fairchem UMA 전용)

```python
# python_force/mlip_interface.py

class UMAAdapter(MLIPAdapter):
    def initialize(self, model_path, task_name="omat", ...):
        # external_graph_gen=True 필수:
        # False이면 r_edges=False → "No edges found" 오류 발생
        inf_settings = InferenceSettings(external_graph_gen=True)
        predict_unit = MLIPPredictUnit(model_path, device=self.device,
                                       inference_settings=inf_settings)
        self._calculator = FAIRChemCalculator(predict_unit, task_name=task)

    def compute_forces(self, positions, atom_types, cell, pbc):
        self._sync_before()          # CUDA synchronize (C++ GPU 작업 완료 대기)
        self._update_atoms(...)      # ASE Atoms 위치 갱신
        forces = self._atoms.get_forces()
        self._sync_after()           # CUDA synchronize (Python GPU 작업 완료 대기)
        return forces
```

**`external_graph_gen=True` 이유:**
- `False`(기본값)일 때 `FAIRChemCalculator`는 `r_edges=False`로 `AtomicData.from_ase` 호출
- 이 경우 모델 내부 그래프 빌더가 작동하지 않아 "No edges found" 발생
- `True`로 설정하면 `r_edges=True` → ASE 기반 radius-graph 외부 생성

### 지원 어댑터 목록

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

### GPU 메모리 최적화 (MLIPAdapter 기반 클래스)

```python
class MLIPAdapter:
    def _ensure_gpu_resources(self, atom_types):
        # 1. 원자 타입 GPU 텐서 — 첫 스텝에만 업로드, 이후 재사용
        if self._atom_types_gpu is None:
            self._atom_types_gpu = torch.tensor(atom_types, device=self.device)

        # 2. 핀 메모리 스테이징 버퍼 — PCIe DMA 가속
        if self._pos_buf is None:
            self._pos_buf = torch.empty((n,3), dtype=torch.float64,
                                        pin_memory=True).numpy()

        # 3. 전용 CUDA 스트림 — H2D 전송 오버랩
        if self._transfer_stream is None:
            self._transfer_stream = torch.cuda.Stream(device=self.device)
```

---

## 경로 비교

| 항목 | C++ MLIP Runtime | Python Interface |
|------|-----------------|-----------------|
| **지원 모델 형식** | TorchScript (.pt) | 모든 ASE 계산기 |
| **GPU-resident** | ✅ 완전 GPU 상주 | ⚠️ CPU 경유 (positions) |
| **C++ 재컴파일** | 불필요 (YAML만 수정) | 불필요 |
| **새 모델 추가** | `model_spec.yaml` 작성 | `ADAPTER_REGISTRY` 에 클래스 추가 |
| **fairchem UMA** | ❌ (TorchScript 비지원) | ✅ |
| **MACE-MP** | ✅ (export 필요) | ✅ |
| **속도** | 최고 | 모델 forward 속도와 동일 |

---

## 새 모델 추가 방법

### C++ MLIP Runtime 경로

1. 모델을 TorchScript로 export
2. `model_spec.yaml` 작성 (입출력 매핑, 단위 정의)
3. `input.yaml`에서 `spec_file: path/to/model_spec.yaml`

### Python Interface 경로

1. `python_force/mlip_interface.py`에 어댑터 클래스 추가:
```python
class MyModelAdapter(MLIPAdapter):
    def initialize(self, model_path, ...): ...
    def compute_forces(self, positions, atom_types, cell, pbc): ...
```
2. `ADAPTER_REGISTRY`에 등록:
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
