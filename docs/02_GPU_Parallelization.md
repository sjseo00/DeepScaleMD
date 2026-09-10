# GPU 병렬화 및 최적화

## 개요

DeepScaleMD는 NVIDIA CUDA를 통해 MD 시뮬레이션의 핵심 연산을 GPU에서 수행한다. 힘 계산, 이웃 목록 구축, 적분, PME 정전기 계산이 모두 GPU 커널로 구현되어 있으며, GPU-resident 방식으로 데이터를 GPU 메모리에 상주시켜 PCIe 전송을 최소화한다.

---

## 1. 빌드 구성

```makefile
# Makefile — CUDA 빌드
make cuda LIBTORCH=ON  # CUDA + LibTorch (MLIP 지원)
make cuda            # CUDA only
make cpu             # CPU fallback
```

**컴파일러:** `nvcc` + `g++-11 (ccbin)`  
**CUDA 버전:** 12.9  
**대상 아키텍처:** SM 7.5+ (RTX 2080 Ti 기준)

---

## 2. ForceCalculatorCUDA — GPU 힘 계산기

### 관련 파일
- `src/force_calculator_cuda.cu`
- `include/force_calculator_cuda.h`

### GPU-resident 데이터 레이아웃

모든 입자 데이터를 GPU에 상주시킨다. SOA(Structure of Arrays) 방식:

```
GPU 메모리:
  d_x[N], d_y[N], d_z[N]       — 위치
  d_vx[N], d_vy[N], d_vz[N]    — 속도
  d_fx[N], d_fy[N], d_fz[N]    — 힘
  d_mass[N], d_charge[N]        — 질량, 전하
  d_type[N]                     — 원자 타입
```

### 최소 이미지 관례 (Minimum Image Convention)

**브랜치리스(branchless) 최적화:**
```cuda
__device__ inline void apply_minimum_image_device(
    Real& dx, Real& dy, Real& dz,
    Real box_x, Real box_y, Real box_z,
    Real inv_box_x, Real inv_box_y, Real inv_box_z)
{
    // round() 사용으로 if-branch 제거 → warp divergence 방지
    dx -= box_x * round(dx * inv_box_x);
    dy -= box_y * round(dy * inv_box_y);
    dz -= box_z * round(dz * inv_box_z);
}
```

### Lennard-Jones 힘 커널

```cuda
__global__ void lj_force_kernel(
    Real* x, Real* y, Real* z,
    Real* fx, Real* fy, Real* fz,
    int* neighbor_list, int* num_neighbors,
    Real cutoff_sq, ...)
{
    // 한 스레드 = 한 입자
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 이웃 목록 순회 → 힘 누적
    // atomicAdd로 반응 힘(reaction force) 기록
}
```

### Double precision atomicAdd

구형 아키텍처(< SM 6.0) 지원을 위한 fallback 구현:
```cuda
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
// 네이티브 double atomicAdd 사용
#else
__device__ double atomicAdd(double* address, double val) {
    // CAS 기반 emulation
}
#endif
```

### __constant__ 메모리 활용

자주 사용하는 상수를 constant cache에 배치:
```cuda
__constant__ Real d_cutoff_sq;
__constant__ Real d_k_coulomb;
__constant__ size_t d_local_particles;
__constant__ size_t d_total_particles;
```

---

## 3. 이웃 목록 (Neighbor List)

### 관련 파일
- `src/neighbor_list_kernels.cu`
- `src/neighbor_list_kernels_optimized.cu`
- `include/neighbor_list.h`

### Verlet-Cell 하이브리드 방식

```
Cell List 구축 → O(N) 셀 할당
    ↓
Verlet skin 반경으로 이웃 후보 필터링
    ↓
Neighbor list 저장 (반목록, E_half 엣지)
```

**Skin 거리:** `cutoff + verlet_skin`까지의 입자를 캐시하여 매 스텝 재구축 불필요

### GPU 기반 재구축 필요성 판단

```cuda
// 현재 위치 vs 마지막 구축 위치 변위 계산
__global__ void compute_displacement_sq_kernel(...) {
    Real dx = x_new[i] - x_old[i];
    // 최소 이미지 관례 적용
    d_disp_sq[i] = dx*dx + dy*dy + dz*dz;
}

// Thrust로 GPU에서 최대값 계산
Real max_disp_sq = thrust::reduce(
    thrust::cuda::par.on(stream),
    thrust::device_pointer_cast(d_disp_sq_tmp), ...
);
// max_disp > verlet_skin/2 이면 재구축
```

CPU로 복사 없이 GPU에서 재구축 필요성을 판단한다.

### Morton Z-order 정렬 (옵션)

```yaml
neighbor_list_type: morton_code
```

공간 locality 향상을 위해 Morton 코드(Z-curve)로 입자 재정렬:
- 인접한 입자가 메모리에도 인접 → 캐시 히트율 향상
- `thrust::sort` 사용

---

## 4. PME 정전기 계산 (CUDA)

### 관련 파일: `src/pme_calculator.cu`, `src/pme_kernels.cu`

Particle Mesh Ewald 방법으로 장거리 정전기를 O(N log N)으로 계산한다.

### GPU 파이프라인

```
1. Charge spreading  (GPU kernel) — B-spline으로 전하를 격자에 분산
       ↓
2. 3D FFT           (cuFFT)       — 전하 격자 → 역공간
       ↓
3. Poisson solver   (GPU kernel) — 역공간에서 에너지/힘 계산
       ↓
4. Inverse FFT      (cuFFT)       — 역공간 → 실공간
       ↓
5. Force interpolation (GPU kernel) — 격자 → 입자 힘
```

**B-spline 차수:** 4차 (cubic)  
**격자 크기:** `input.yaml`에서 `pme_grid_size: [64, 64, 64]`로 설정

---

## 5. MLIP GPU-resident 연동

### C++ MLIP Runtime (TorchScript 모드)

```
GPU 메모리 (DeepScaleMD)
  d_x, d_y, d_z [N]      — SoA 위치
  d_edge_coo_packed       — 반목록 COO 엣지
        ↓
  launch_interleave_positions()  — SoA → AoS 변환 (GPU 커널)
  launch_half_to_full_edges()    — 반목록 → 전방향 목록 (GPU 커널)
        ↓
  torch::jit::load() 모델 forward (CUDA 텐서, 스트림 공유)
        ↓
  launch_accumulate_forces_scaled()  — 모델 출력 힘 누적 (GPU 커널)
```

**스트림 공유:** `share_cuda_stream: true` 설정 시 DeepScaleMD CUDA 스트림을 LibTorch에 전달하여 동기화 오버헤드 제거

### 위치 인터리빙 커널

```cuda
// SoA (d_x[], d_y[], d_z[]) → AoS ([x0,y0,z0, x1,y1,z1, ...])
// 모델 입력 형식: [N, 3] float64
void launch_interleave_positions(
    const double* d_x, const double* d_y, const double* d_z,
    double* out, int N, cudaStream_t stream);
```

---

## 6. 이중 정밀도 vs 단정밀도

| 데이터 | 정밀도 | 이유 |
|--------|--------|------|
| 입자 위치/힘 | `double` (float64) | MD 정확도 요구사항 |
| PME 격자 | `Real` (빌드 설정) | 정확도/성능 트레이드오프 |
| MLIP 입력 | `float32` (설정 가능) | 모델 학습 정밀도와 일치 |

---

## 7. GPU 프로파일링

```bash
# Nsight Systems로 프로파일
nsys profile --trace=cuda,mpi bin/md_simulation_cuda input.yaml

# 환경변수로 타이밍 활성화
MD_TIMING=1 bin/md_simulation_cuda input.yaml
```

`timing_breakdown.cpp`에서 각 구성요소별 시간 측정:
- Pair (힘 계산)
- Neigh (이웃 목록)
- Comm (MPI 통신)
- Output (출력)

---

## 8. 주요 성능 지표 (RTX 2080 Ti 기준)

| 시스템 | MLIP | 스텝당 시간 |
|--------|------|------------|
| 32 mol water | LJ (builtin) | ~1 ms |
| 32 mol water | MACE-MP-0 small | ~15 ms |
| 32 mol water | UMA-S-1.1 | ~125 ms |
| 96 mol water | LJ (builtin) | ~3 ms |
