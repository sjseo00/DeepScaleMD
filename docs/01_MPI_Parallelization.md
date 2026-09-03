# MPI 병렬화 및 최적화

> **현재 architecture:** Cartesian decomposition, halo exchange, migration은
> `dsDecomp`를 사용한다. 이 문서의 삭제된 legacy `DomainDecomposition`
> source 참조는 역사적 설명이다. 현재 경계는
> `dsDecomp/docs/integration-boundary-audit.md`를 참고한다.

## 개요

DeepScaleMD는 MPI(Message Passing Interface)를 기반으로 다중 프로세스 분산 MD 시뮬레이션을 지원한다. 도메인 분해(Domain Decomposition) 방식을 채택하여 시뮬레이션 박스를 3D 격자로 분할하고, 각 MPI 랭크가 하나의 서브도메인을 담당한다.

---

## 1. MPI 초기화

### 진입점: `src/main.cpp`

```cpp
MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
initialize_mpi_utils();
```

- `MPI_THREAD_MULTIPLE` 수준으로 초기화하여 멀티스레드 환경에서 안전한 MPI 호출 보장
- `initialize_mpi_utils()` 로 전역 rank/size를 캐시 → 매 호출마다 `MPI_Comm_rank` 하지 않음

### OpenMP 자동 조정

```cpp
threads_per_rank = std::max(1, max_threads / mpi_size);
omp_set_num_threads(threads_per_rank);
```

- `OMP_NUM_THREADS` 미설정 시 전체 코어 수를 MPI 랭크 수로 나눠 자동 할당
- 과다 구독(oversubscription) 방지

---

## 2. 도메인 분해 (Domain Decomposition)

### 관련 파일
- `src/domain_decomposition.cpp`
- `include/domain_decomposition.h`

### 3D 격자 분해

```
+-------+-------+
| rank0 | rank1 |    예: 4 MPI 프로세스 → 2x2x1 격자
+-------+-------+
| rank2 | rank3 |
+-------+-------+
```

**최적 격자 결정 알고리즘:**
- 전체 프로세스 수 `nprocs`의 3D 인수분해 시도
- 최소 종횡비(aspect ratio)를 갖는 분해 선택 → 가장 정육면체에 가까운 격자
- `max_dim / min_dim` 최소화

**격자 통신자 생성:**
```cpp
MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart_comm_);
```
- 주기 경계 조건에 대응하는 Cartesian 통신자

---

## 3. 헤일로 교환 (Halo Exchange)

각 서브도메인 경계 근방 입자의 위치/힘을 인접 랭크와 교환한다.

### 헤일로 폭 계산
```cpp
halo_width_ = cutoff * 1.2;  // cutoff + 20% 여유
```

### 교환 과정

```
[Step 1] 위치 전송 (forward halo)
  - 로컬 입자 위치 계산 후
  - 경계 영역 입자를 인접 랭크로 전송 (ghost particles 생성)

[Step 2] 힘 반환 (reverse halo)
  - 모든 랭크가 로컬+ghost 입자 힘 계산 후
  - ghost 입자 힘을 원 소유 랭크로 반환하여 누적
```

---

## 4. MPIOptimizer — 비동기 집합 통신

### 관련 파일: `src/mpi_optimizer.cpp`, `include/mpi_optimizer.h`

### 주요 최적화

| 기법 | 설명 |
|------|------|
| **비동기 Allreduce** | `MPI_Iallreduce` 사용, 통신과 계산 오버랩 |
| **요청 배칭** | 다수의 소규모 reduction을 모아서 한 번에 처리 |
| **지연 동기화** | `wait_all()` 시점까지 완료 대기 지연 |

```cpp
// 비동기 double 배열 리덕션
void MPIOptimizer::allreduce_async_double(
    const double* sendbuf, double* recvbuf,
    int count, MPI_Op op)
{
    MPI_Iallreduce(..., &req.request);  // 즉시 반환
    requests_.push_back(req);           // 큐에 저장
}

// 나중에 일괄 대기
void MPIOptimizer::wait_all();
```

---

## 5. CUDA-aware MPI

### 관련 파일: `src/cuda_mpi_system.cpp`, `include/cuda_mpi_system.h`

GPU 메모리 버퍼를 CPU로 복사 없이 직접 MPI 전송한다.

### 탐지 방법
1. 환경변수 `OMPI_MCA_opal_cuda_support=true` 확인
2. 런타임 테스트: GPU 포인터로 `MPI_Buffer_attach` 시도

### 활성화 시 동작
```cpp
if (cuda_aware_mpi_available) {
    // GPU 포인터를 MPI에 직접 전달
    allocate_migration_buffers();  // GPU 이주 버퍼 할당
}
```

### NCCL 지원 (USE_NCCL 빌드 시)
```cpp
#ifdef USE_NCCL
    initialize_nccl();  // GPU간 직접 collective 통신
#endif
```

---

## 6. Overlap Force Calculator — 통신/계산 오버랩

### 관련 파일: `src/overlap_force_calculator.cpp`

경계(boundary) 입자와 내부(interior) 입자를 분리하여 동시 처리한다.

```
┌─────────────────────────────────────────────────────┐
│ Stream 1 (interior):   [force calc — interior]       │
│ Stream 2 (boundary):   [force calc — boundary] ──→  │
│ Stream 3 (comm):                    [MPI send/recv]  │
└─────────────────────────────────────────────────────┘
```

**3개의 CUDA 스트림:**
- `interior_stream`: 내부 입자 힘 계산 (통신과 완전 오버랩 가능)
- `boundary_stream`: 경계 입자 힘 계산
- `communication_stream`: MPI 통신

**CUDA 이벤트 동기화:**
```cpp
cudaEventCreate(&interior_complete_event);
cudaEventCreate(&boundary_ready_event);
cudaEventCreate(&communication_complete_event);
```

---

## 7. 힘 집합 (Force Reduction)

각 랭크에서 계산된 힘을 집합 합산한다.

```cpp
// 비동기로 전체 힘 벡터 합산
mpi_optimizer.allreduce_async_double(local_fx, global_fx, N, MPI_SUM);
mpi_optimizer.allreduce_async_double(local_fy, global_fy, N, MPI_SUM);
mpi_optimizer.allreduce_async_double(local_fz, global_fz, N, MPI_SUM);
// ...다른 계산 수행...
mpi_optimizer.wait_all();  // 완료 대기
```

---

## 8. MPI 스케일링 특성

### 구현된 최적화 요소

| 최적화 | 구현 위치 | 효과 |
|--------|-----------|------|
| 비동기 집합 통신 | `MPIOptimizer` | 통신 지연 은폐 |
| 도메인 분해 격자 최적화 | `DomainDecomposition` | 통신량 최소화 |
| 통신/계산 오버랩 | `OverlapForceCalculator` | 경계 통신 숨김 |
| CUDA-aware MPI | `CUDAMPISystem` | H2D/D2H 복사 제거 |
| OpenMP 자동 스레드 조정 | `main.cpp` | 과다 구독 방지 |

### 실행 예시

```bash
# 4 MPI 프로세스, 각 4 OpenMP 스레드
mpirun -n 4 bin/md_simulation_cuda input.yaml

# CUDA-aware MPI 활성화
OMPI_MCA_opal_cuda_support=true mpirun -n 4 bin/md_simulation_cuda input.yaml
```
