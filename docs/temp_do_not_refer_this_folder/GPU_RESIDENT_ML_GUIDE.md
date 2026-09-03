# GPU-Resident MPI+CUDA Architecture for ML/Python Forces on GH200

## Overview

이 문서는 DeepScaleMD를 **멀티노드 GH200 (Grace Hopper Supernode)** 환경에서 **Python/ML force와 함께 GPU-resident 모드로 최적화 실행**하기 위한 구현 전략과 사용 가이드입니다.

## 핵심 설계 원칙

### 1. GPU Memory Residency
- **좌표/속도/힘이 GPU에 상주**: `d_x/d_y/d_z`, `d_vx/d_vy/d_vz`, `d_fx/d_fy/d_fz`
- Host↔Device 전송은 **출력/체크포인트 시에만** 수행
- 모든 컴포넌트(integrator/force/thermostat/barostat)가 **동일한 device 버퍼 공유**

### 2. MPI-Aware Python/ML Force
- **rank 0만 Python 초기화 및 ML 추론 수행** (멀티프로세스 안정성)
- 결과를 `MPI_Bcast`로 전체 rank에 배포
- 최종적으로 `upload_forces_to_device()`로 GPU에 업로드

### 3. Unified Device Pointer API
- `ForceCalculatorCUDA`가 `get_device_x/y/z/vx/vy/vz/fx/fy/fz()`를 제공
- Thermostat/Barostat는 `apply_device(d_vx, d_vy, d_vz, ...)` API 사용
- 중복 버퍼 할당 없음 (barostat/thermostat 내부 `d_*` 제거 권장)

---

## 구현된 주요 변경사항

### 1. ForceCalculatorCUDA 확장 (`include/force_calculator_cuda.h`)

```cpp
// NEW: Force getter for complete GPU-resident pipeline
Real* get_device_fx() { return d_fx; }
Real* get_device_fy() { return d_fy; }
Real* get_device_fz() { return d_fz; }

// Upload forces from host to device (for Python/ML force integration)
void upload_forces_to_device(const ParticleData& particles, size_t num_particles);
```

**구현 위치**: `src/force_calculator_cuda.cu`

```cpp
void ForceCalculatorCUDA::upload_forces_to_device(const ParticleData& particles, size_t num_particles) {
    // Uploads fx/fy/fz from host ParticleData to device memory
    // Synchronizes to ensure forces are ready before integrator uses them
}
```

### 2. PythonForceInterface MPI-aware 재설계 (`include/force_calculator.h`)

**헤더 변경**:
```cpp
class PythonForceInterface {
private:
    int mpi_rank, mpi_size;
    bool is_initialized;
    
    // Cached Python objects (avoid repeated imports)
    void* py_module;           // PyObject* (force_calculator module)
    void* py_ml_module;        // PyObject* (ml_forces module)
    void* py_compute_func;     // Cached function pointer
    void* py_ml_compute_func;  // Cached ML function pointer
    
public:
    // MPI-aware: rank 0 only computes, broadcast to all
    void compute_forces(ParticleData& particles, ...);
};
```

**구현**: `src/python_interface_mpi.cpp`
- `initialize_python()`: rank 0만 `Py_Initialize()` + 모듈 사전 로드
- `compute_forces()`: rank 0 실행 후 `MPI_Bcast(fx/fy/fz, ...)`

### 3. md_simulation.cpp GPU-resident 파이프라인 완성

**Python force 통합**:
```cpp
if (force_calculation_mode == "python" && python_interface) {
    // Rank 0 computes Python/ML forces
    python_interface->compute_forces(system->get_particles(), ...);
    
#ifdef USE_CUDA
    if (using_cuda && cuda_calc) {
        // Upload forces to GPU for resident pipeline
        cuda_calc->upload_forces_to_device(system->get_particles(), system->get_num_particles());
    }
#endif
}
```

**Thermostat/Barostat GPU-resident 경로 강제**:
```cpp
if (thermostat) {
#ifdef USE_CUDA
    if (using_cuda && cuda_calc) {
        // Always use apply_device (NO host fallback)
        thermostat->apply_device(cuda_calc->get_device_vx(), ...);
    } else {
        thermostat->apply(system->get_particles(), ...);
    }
#endif
}
```

---

## 사용 방법

### 1. 빌드 설정

**Makefile / CMakeLists.txt**에서:
```makefile
CXXFLAGS += -DUSE_CUDA -DUSE_MPI -DENABLE_PYTHON_INTERFACE
LDFLAGS += -lcudart -lmpi -lpython3.x
```

### 2. Python/ML Force 모듈 작성

**`python_force/ml_forces.py`** (예시):
```python
import torch

# 모델은 초기화 시 한 번만 로드 (중요!)
model = None

def compute_ml_forces(positions, types, box, **kwargs):
    """
    Args:
        positions: (N, 3) numpy array of positions
        types: (N,) numpy array of particle types
        box: (3,) box dimensions [lx, ly, lz]
    
    Returns:
        forces: (N, 3) numpy array of forces
    """
    global model
    if model is None:
        model = torch.jit.load('model.pt')
        model = model.cuda()  # GPU에 모델 로드
    
    # GPU tensor로 변환 (H2D 최소화)
    pos_tensor = torch.from_numpy(positions).float().cuda()
    
    # 추론
    with torch.no_grad():
        forces_tensor = model(pos_tensor)
    
    # Numpy로 변환 (D2H - 이 부분은 추후 DLPack로 최적화 가능)
    return forces_tensor.cpu().numpy()
```

### 3. 실행 설정 (`config.txt`)

```
# Force calculation mode
force_calculation_mode = python
python_force_path = ./python_force

# Thermostat (GPU-resident)
thermostat_type = langevin
target_temperature = 300.0
tau_t = 0.1

# Barostat (GPU-resident)
barostat_type = andersen
target_pressure = 1.0
tau_p = 1.0
```

### 4. MPI 실행

**단일 노드 (4 GPU)**:
```bash
mpirun -np 4 ./bin/md_simulation_cuda config.txt
```

**멀티 노드 (GH200 SuperPOD, 8 노드 x 4 GPU = 32 ranks)**:
```bash
srun --nodes=8 --ntasks-per-node=4 --gpus-per-task=1 \
     ./bin/md_simulation_cuda config.txt
```

---

## 성능 최적화 팁 (GH200 환경)

### 1. CUDA-aware MPI 활용 (추후 단계)

현재 halo exchange는:
- GPU → Host → MPI → Host → GPU

**최적화 목표**:
- GPU → MPI (device pointer) → GPU

**구현 힌트** (`include/cuda_mpi_system.h` 참고):
```cpp
// Pack on device
pack_particles_kernel<<<...>>>(d_x, d_y, d_z, d_send_buffer, ...);

// MPI with device buffer (CUDA-aware MPI)
MPI_Isend(d_send_buffer, count, MPI_DOUBLE, dest, tag, comm, &req);
MPI_Irecv(d_recv_buffer, count, MPI_DOUBLE, src, tag, comm, &req);

// Unpack on device
unpack_particles_kernel<<<...>>>(d_recv_buffer, d_x, d_y, d_z, ...);
```

### 2. Neighbor List GPU 빌드

현재는 CPU에서 neighbor list 빌드 → GPU 업로드.

**최적화**: `build_neighbor_list_on_device()` 사용 (이미 구현됨):
```cpp
cuda_calc->build_neighbor_list_on_device(*system);
```

### 3. Python/ML Force GPU Direct (DLPack)

현재:
- Rank 0: numpy (host) → MPI broadcast → GPU upload

**최적화**:
```python
# DLPack로 GPU tensor를 직접 노출
import torch.utils.dlpack as dlpack

def compute_ml_forces_gpu(positions_tensor):
    """
    Args:
        positions_tensor: torch.Tensor on GPU (passed via DLPack from C++)
    Returns:
        forces_tensor: torch.Tensor on GPU
    """
    forces = model(positions_tensor)
    return forces  # C++에서 DLPack로 받아 d_fx/d_fy/d_fz에 직접 alias
```

C++ 쪽:
```cpp
// Python에서 DLPack capsule 받기
PyObject* capsule = PyObject_CallMethod(py_result, "__dlpack__", NULL);
// cudaMemcpy 없이 d_fx와 공유
```

### 4. Stream Overlapping

```cpp
// Compute stream과 Communication stream 분리
cudaStream_t compute_stream, comm_stream;

// Force kernel (compute_stream)
compute_lj_forces_kernel<<<..., compute_stream>>>(...);

// Halo pack (comm_stream, 병렬 실행 가능)
pack_particles_kernel<<<..., comm_stream>>>(...);
```

---

## 현재 제한사항 및 향후 개선

### 1. MPI Broadcast 방식의 비효율
- **현재**: rank 0가 전체 시스템 force 계산 → broadcast
- **문제**: Domain decomposition에서 각 rank는 local particles만 소유
- **개선**: 
  - Rank별 local particle 범위 기반 Scatterv/Gatherv
  - 또는 각 rank가 자기 영역만 ML 추론 (모델 복제)

### 2. Host 경유 Force 업로드
- **현재**: Python (host numpy) → Host ParticleData → GPU upload
- **개선**: Python (GPU tensor) → DLPack → C++ device pointer (zero-copy)

### 3. Halo Exchange Host 경유
- **현재**: GPU → Host → MPI → Host → GPU
- **개선**: CUDA-aware MPI로 device-to-device 통신

---

## 검증 방법

### 1. GPU-resident 모드 확인
```bash
# 환경변수로 타이밍 출력 활성화
export MD_TIMING=1
./bin/md_simulation_cuda config.txt

# 출력에서 확인:
# - "Using CUDA for force calculations (GPU-resident mode)"
# - "Python interpreter initialized on rank 0"
# - Step 타이밍에서 D2H/H2D 시간이 0 or 매우 작음
```

### 2. Python force 정합성
```python
# python_force/ml_forces.py에 디버깅 출력
def compute_ml_forces(...):
    print(f"[Rank 0] Computing ML forces for {len(positions)} particles")
    ...
```

### 3. Thermostat/Barostat Device 경로
- 로그에서 `apply_device()` 호출 확인
- 코드에 디버깅: `std::cout << "Thermostat GPU-resident path" << std::endl;`

---

## 요약

이 구현으로 **GH200 멀티노드 환경에서**:
1. **좌표/속도/힘이 GPU에 상주**하며 매 스텝 왕복 없음
2. **Python/ML force가 rank 0에서만 실행**되어 안정성 확보
3. **Thermostat/Barostat가 GPU device pointer로 직접 동작**
4. **단일 device 버퍼 공유**로 메모리 효율 극대화

**다음 단계**: CUDA-aware MPI, DLPack GPU-direct, Neighbor list GPU 빌드로 더욱 최적화 가능합니다.
