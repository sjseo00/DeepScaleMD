# DeepScaleMD MLP 인터페이스 데이터 흐름 & GPU 최적화 전략

작성일: 2026-04-21  
브랜치: main  
검증 환경: NVIDIA RTX 2080 Ti (11.3 GB), CUDA 12.9, PyTorch 2.6.0+cu124, MACE 0.3.15

---

## 1. 전체 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                    md_simulation.cpp                         │
│                  (메인 MD 루프 제어)                          │
│                                                             │
│  ┌──────────────────────┐   ┌──────────────────────────┐   │
│  │  ForceCalculatorCUDA │   │  PythonForceInterface    │   │
│  │  (GPU 적분/이웃목록)  │   │  (C++ ↔ Python 브릿지)   │   │
│  └──────────────────────┘   └──────────────────────────┘   │
│              │                          │                    │
│         CUDA Device               Python C API              │
│         (RTX 2080 Ti)           ml_forces.py → MACE         │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. GPU-Resident MD 루프에서의 MLP 데이터 흐름 (per-step)

```
[매 스텝 시작]
      │
      ▼
┌─────────────────────────────────────┐
│ 1. GPU: integrate_on_device()       │  ← Velocity-Verlet 1단계
│    d_x/y/z[t] → d_x/y/z[t+dt]      │    (위치 갱신, GPU에만 존재)
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ 2. ForceCalculatorCUDA::            │
│    download_positions_for_ml()      │  ← D→H (Device → Host)
│                                     │
│    cudaStreamSynchronize(stream)    │  GPU 커널 완료 대기
│    cudaMemcpyAsync(                 │
│      particles.x ← d_x,            │  pinned memory DMA
│      particles.y ← d_y,            │  (staging buffer 없음)
│      particles.z ← d_z )           │
│    cudaStreamSynchronize(stream)    │  전송 완료 대기
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ 3. PythonForceInterface::           │  ← C++ → Python C API
│    compute_ml_forces()              │
│                                     │
│    NumPy 배열 구성 (positions,       │  particles.x/y/z → ndarray
│                    atom_types,      │
│                    cell, pbc)       │
│                                     │
│    PyObject_CallObject(             │  Python 함수 호출
│      "ml_forces.compute_ml_forces") │
│                                     │
│    ┌─────────────────────────────┐  │
│    │  ml_forces.py               │  │
│    │  torch.cuda.synchronize()   │  │  C++ CUDA와의 컨텍스트 충돌 방지
│    │  ASE Atoms.get_forces()     │  │  ← MACE-MP 추론 (GPU)
│    │  torch.cuda.synchronize()   │  │  PyTorch CUDA 완료 보장
│    └─────────────────────────────┘  │
│                                     │
│    결과: particles.fx/fy/fz 갱신     │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ 4. ForceCalculatorCUDA::            │
│    upload_forces_from_ml()          │  ← H→D (Host → Device)
│                                     │
│    cudaMemcpyAsync(                 │
│      d_fx ← particles.fx,          │  pinned memory DMA
│      d_fy ← particles.fy,          │
│      d_fz ← particles.fz )         │
│    cudaStreamSynchronize(stream)    │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│ 5. GPU: complete_velocity_update()  │  ← Velocity-Verlet 2단계
│    d_vx/y/z 최종 갱신               │
└─────────────────────────────────────┘
      │
      ▼
[다음 스텝]
```

---

## 3. 초기화 시퀀스 (1회)

```
loadConfiguration()
    └─ mace_config 파싱 → mace_device, mace_model_path 등 저장
    └─ USE_CUDA + mace_device != "cpu" → mace_device = "cuda" (자동 승격)

initialize()
    └─ PythonForceInterface::initialize_python()
         └─ Py_Initialize() + _import_array() (NumPy C API)
         └─ sys.path에 python_force/ 추가

    └─ PythonForceInterface::initialize_ml_model()
         └─ PyRun_SimpleString("ml_forces.initialize_ml_forcefield_mace(...)")
              └─ _patch_mace_einsum_cuda_bug()    ← PyTorch cuBLAS 버그 패치
              └─ mace_mp(model=..., device='cuda') 로드
              └─ GPU warmup inference (cuBLAS lazy init 강제)
              └─ torch.cuda.synchronize()

    └─ cuda_calc->initial_upload_to_device()
         └─ 전체 particle data Host→Device (1회 전체 복사)

    └─ cuda_calc->register_ml_pinned_buffers()   ← 초기화 1회만 실행
         └─ cudaHostRegister(particles.x/y/z)    pinned memory 등록
         └─ cudaHostRegister(particles.fx/fy/fz)

    [종료 시]
    └─ cuda_calc->unregister_ml_pinned_buffers()
         └─ cudaHostUnregister(particles.x/y/z)
         └─ cudaHostUnregister(particles.fx/fy/fz)
```

---

## 4. 메모리 전송 최적화 전략

### 4.1 Pinned Memory (Page-Locked Memory)

```
일반 pageable 메모리 전송:
  Host RAM → [임시 staging buffer (pinned)] → GPU VRAM
  → DMA 2회 발생, CPU가 staging buffer 관리

Pinned memory 전송 (cudaHostRegister 후):
  Host RAM (page-locked) ──────────────────→ GPU VRAM
  → DMA 1회, CPU 개입 없음, ~2x 대역폭
```

| 버퍼 | 크기 (288 atoms, Real=double) | 전송 방향 | 빈도 |
|------|-------------------------------|-----------|------|
| `particles.x/y/z` | 6.9 KB (3 × 288 × 8 B) | D→H (매 스텝) | 매 스텝 |
| `particles.fx/fy/fz` | 6.9 KB | H→D (매 스텝) | 매 스텝 |
| 나머지 (vx/vy/vz, mass, type 등) | ~30 KB | H→D 초기 1회 | 초기화만 |

### 4.2 최소 전송 원칙

- ❌ **이전 방식**: 매 스텝 전체 particle data (위치+속도+힘 전부) H↔D 복사
- ✅ **현재 방식**: **위치만 D→H**, **힘만 H→D** (MLP 경로 전용 최소 전송)
- ✅ 적분/이웃목록/Thermostat 등 순수 GPU 연산은 **H↔D 전송 없음**

### 4.3 CUDA Stream 격리

```cpp
// force_stream에서 비동기 전송 + 동기화
cudaMemcpyAsync(dst, src, bytes, dir, force_stream);
cudaStreamSynchronize(force_stream);
```

- 기본 스트림과 분리된 `force_stream` 사용
- 추후 적분 커널과 MLP 전송의 overlap 가능성 확보

---

## 5. C++ ↔ Python 인터페이스 상세

### 5.1 `src/python_interface.cpp`: `compute_ml_forces()`

```
입력 (C++ → Python 변환):
  particles.x/y/z  → NumPy float64 [N, 3]  positions
  particles.type[] → NumPy int32   [N]      atom_types (0-indexed: OW=0, HW=1)
  box_x/y/z        → NumPy float64 [3]      cell (orthorhombic)
  pbc_x/y/z        → NumPy int32   [3]      periodic boundary flags

출력 (Python → C++ 역변환):
  NumPy float64 [N, 3] forces → particles.fx/fy/fz[]
```

### 5.2 `python_force/ml_forces.py`: `compute_ml_forces()`

```python
# CUDA 동기화 (C++ CUDA 컨텍스트와의 충돌 방지)
if _mace_device != "cpu":
    torch.cuda.synchronize()

# ASE Atoms 객체 재사용 (캐싱으로 불필요한 재생성 방지)
_atoms_cache.set_positions(positions)   # 위치만 갱신 (빠름)

# MACE 추론 (내부적으로 energy → −∂E/∂r = forces)
forces = _atoms_cache.get_forces()

# PyTorch CUDA 완료 보장 후 C++로 반환
if _mace_device != "cpu":
    torch.cuda.synchronize()
return np.ascontiguousarray(forces, dtype=np.float64)  # C-contiguous float64 보장
```

### 5.3 Atom Type 매핑

```
topology (water_96mol.top)
    OW → type ID 0   (0-indexed, md_system.cpp에서 할당)
    HW → type ID 1

python_interface.cpp
    particles.type[i] → NumPy int32 배열

ml_forces.py
    type 0 → 'O'  (ASE chemical symbol)
    type 1 → 'H'
    → MACE-MP에 올바른 원소 정보 전달
```

---

## 6. Makefile: CUDA 빌드에 Python 지원 추가

기존 `NVCC_FLAGS`에 Python/NumPy include 경로가 누락되어 CUDA 바이너리에서  
`ENABLE_PYTHON_INTERFACE`가 비활성화되었던 문제를 수정:

```makefile
# Python C API include (gcc-specific 경고 플래그 제외, include 경로만 추출)
PYTHON_INCLUDES = -I$(shell python3 -c "import sysconfig; print(sysconfig.get_path('include'))")

NVCC_FLAGS = ... -DENABLE_PYTHON_INTERFACE $(PYTHON_INCLUDES) $(NUMPY_FLAGS)
#                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                이전에는 이 부분이 없어서 CUDA 바이너리에서 Python이 disabled였음
```

---

## 7. 환경별 성능 결과

### 192분자 (576 atoms, 10 스텝)

| 설정 | ms/스텝 | 비율 |
|------|---------|------|
| CPU 바이너리 + MACE CPU | ~5,760 ms | 1× (기준) |
| **GPU 바이너리 + MACE GPU** | **~362 ms** | **🚀 15.9×** |

### 96분자 (288 atoms, 50 스텝)

| 설정 | ms/스텝 | 비고 |
|------|---------|------|
| CPU 바이너리 + MACE CPU | ~113 ms | — |
| GPU 바이너리 + MACE GPU | ~111 ms | 소형 시스템 → GPU 이점 미미 |

> **소형 시스템 (≤288 atoms)**: MACE 추론 시간이 짧아 GPU 커널 launch overhead가 상대적으로 커서 CPU와 차이 없음  
> **대형 시스템 (≥576 atoms)**: GPU 병렬 연산 효율이 높아 15× 이상 가속

---

## 8. 주요 버그 및 해결 이력

| # | 버그 | 원인 | 해결 |
|---|------|------|------|
| 1 | YAML `mace_config` 섹션 미파싱 | `trim()` 후 `line[0]` 검사 → 섹션 헤더 미인식 | trim 전 원본 라인으로 `[...]` 감지 |
| 2 | `particles.type[]` 전부 0 | `assign_masses_from_topology()`에서 type ID 미할당 | 0-indexed type ID 할당 코드 추가 |
| 3 | MACE 모델 미초기화 | `initialize_ml_model()` 구현 누락 | `python_interface.cpp`에 구현 추가 |
| 4 | GPU-resident stale position | 적분 후 위치가 GPU에만 존재, host 미갱신 | `download_positions_for_ml()` D→H 추가 |
| 5 | float32 forces segfault | MACE float32 반환 → C++ `double*` 역참조 | `np.ascontiguousarray(..., dtype=np.float64)` |
| 6 | CUDA 바이너리 Python 비활성화 | `NVCC_FLAGS`에 `-DENABLE_PYTHON_INTERFACE` 누락 | `PYTHON_INCLUDES`를 `NVCC_FLAGS`에 추가 |
| 7 | `CUBLAS_STATUS_INVALID_VALUE` | PyTorch **2.10.0** nightly의 `cublasSgemmStridedBatched` 버그 (batch_size ≥ 2) | PyTorch **2.6.0+cu124** 안정 버전으로 다운그레이드 |

---

## 9. 관련 파일 목록

| 파일 | 역할 |
|------|------|
| `src/md_simulation.cpp` | MLP 데이터 흐름 제어 (D→H, 호출, H→D) |
| `src/python_interface.cpp` | C++ ↔ Python C API 브릿지, NumPy 배열 변환 |
| `src/force_calculator_cuda.cu` | `download_positions_for_ml`, `upload_forces_from_ml`, `register_ml_pinned_buffers` 구현 |
| `include/force_calculator_cuda.h` | 위 함수들 선언 |
| `python_force/ml_forces.py` | MACE ASE calculator 래퍼, `_patch_mace_einsum_cuda_bug()` |
| `Makefile` | CUDA 빌드에 Python/NumPy 헤더 포함 |
| `mace_bench_test.yaml` | 96분자 MACE GPU 벤치마크 설정 (`device: "cuda"`) |
| `bench_192mol_gpu.yaml` | 192분자 MACE GPU 벤치마크 설정 |
| `models/mace/mace_mp_small.model` | MACE-MP-0 small 모델 파일 (32 MB) |
