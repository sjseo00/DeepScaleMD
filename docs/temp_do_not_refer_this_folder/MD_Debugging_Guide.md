# MD Simulation Debugging Guide (NVE Ensemble)
> **목적**: CPU 단독 / MPI / CUDA / MPI+CUDA 빌드가 모두 정상 동작할 때까지 반복 디버깅

---

## 0. 전제 조건 확인

```bash
# 컴파일러 / 런타임 버전 확인 (설치 없이 현황만 파악)
echo "=== Compiler & Runtime Check ==="
which g++ && g++ --version
which mpicxx && mpicxx --version
which nvcc  && nvcc --version
which mpirun && mpirun --version

# GPU 상태
nvidia-smi 2>/dev/null || echo "[WARN] nvidia-smi 없음 – CUDA 환경 없음"

# 예제 입력 파일 존재 확인
ls -lh examples/lj_32000/input.yaml 2>/dev/null || echo "[ERROR] input.yaml 없음 – examples/ 경로 점검 필요"
```

---

## 1. 소스 코드 구조 점검

```bash
echo "=== Source Tree ==="
find . -name "*.cpp" -o -name "*.cu" -o -name "*.h" | sort

echo "=== Makefile 내용 ==="
cat Makefile
```

디버깅 포인트:
- `Makefile` 에 `cuda` 타겟이 정의되어 있는지 확인
- `mpicxx` / `nvcc` 링크 플래그(`-lmpi`, `-lcudart`) 누락 여부 확인

---

## 2. CPU 단독 빌드 & 실행

### 2-1. 빌드

```bash
echo "=== [STEP 1] CPU 빌드 ==="
make clean 2>&1 | tail -5
make 2>&1          # 에러 전체 출력 보존
echo "Exit code: $?"
ls -lh md_simulation_cpu 2>/dev/null || echo "[ERROR] 바이너리 생성 실패"
```

**빌드 실패 시 체크리스트**

| 증상 | 원인 | 조치 |
|------|------|------|
| `undefined reference to MPI_*` | CPU 빌드에 MPI 플래그 혼입 | `Makefile` CPU 타겟에서 `-lmpi` 제거 |
| `fatal error: yaml-cpp/yaml.h` | yaml-cpp 미설치 | `sudo apt install libyaml-cpp-dev` 또는 헤더 경로 추가 |
| `error: 'omp_*' not declared` | OpenMP 플래그 누락 | `-fopenmp` 추가 |

### 2-2. 실행

```bash
echo "=== [STEP 1] CPU 실행 ==="
./md_simulation_cpu examples/lj_32000/input.yaml 2>&1
echo "Exit code: $?"
```

**실행 실패 시 체크리스트**

```bash
# segfault 위치 추적
g++ -g -O0 -o md_simulation_cpu_dbg <소스파일들> 2>&1
valgrind --leak-check=full ./md_simulation_cpu_dbg examples/lj_32000/input.yaml 2>&1 | head -80

# 또는 AddressSanitizer
g++ -fsanitize=address -g -O1 -o md_simulation_cpu_asan <소스파일들> 2>&1
./md_simulation_cpu_asan examples/lj_32000/input.yaml 2>&1 | head -80
```

| 증상 | 원인 | 조치 |
|------|------|------|
| `Segmentation fault` | 배열 경계 초과 / null 포인터 | valgrind / ASAN 로그 확인 |
| `NaN` 발생 | 초기 입자 배치 겹침 | 최소 거리 조건 확인, σ 단위 초기화 검증 |
| 에너지 발산 | dt 너무 큼 | `input.yaml`에서 `dt: 0.001` 이하로 감소 |
| 성능 줄 미출력 | 로그 포맷 불일치 | 마지막 2번째 줄 `=== Total execution time:` 포함 여부 확인 |

---

## 3. MPI 빌드 & 실행

### 3-1. 빌드

```bash
echo "=== [STEP 2] MPI 빌드 ==="
make clean && make mpi 2>&1   # 또는 타겟 이름이 다를 경우 아래 직접 확인
# mpicxx -O2 -o md_simulation_mpi <소스파일들> -lyaml-cpp 2>&1
echo "Exit code: $?"
ls -lh md_simulation_mpi 2>/dev/null || echo "[ERROR] MPI 바이너리 없음"
```

**MPI 빌드 실패 체크리스트**

```bash
# mpicxx 래퍼가 어떤 플래그를 주입하는지 확인
mpicxx --showme 2>/dev/null || mpicxx -show 2>/dev/null

# include 경로 수동 확인
mpicc --showme:incdirs
mpicc --showme:libdirs
```

| 증상 | 원인 | 조치 |
|------|------|------|
| `mpicxx: command not found` | MPI 미설치 or PATH 문제 | `which mpicxx` 후 PATH 설정 |
| `undefined reference to MPI_Init` | `mpicxx` 대신 `g++` 사용 | 컴파일러를 `mpicxx`로 교체 |
| 링크 시 `-lmpi_cxx` 없음 | OpenMPI vs MPICH 차이 | `mpicxx --showme:libs` 결과 반영 |

### 3-2. 실행

```bash
echo "=== [STEP 2] MPI 실행 (rank=4) ==="
mpirun -np 4 ./md_simulation_mpi examples/lj_32000/input.yaml 2>&1
echo "Exit code: $?"

# rank 수 조정 테스트
mpirun -np 1 ./md_simulation_mpi examples/lj_32000/input.yaml 2>&1 | tail -5
mpirun -np 2 ./md_simulation_mpi examples/lj_32000/input.yaml 2>&1 | tail -5
```

**MPI 실행 실패 체크리스트**

```bash
# MPI 초기화 오류 진단
OMPI_MCA_mpi_show_mca_params=1 mpirun -np 2 ./md_simulation_mpi examples/lj_32000/input.yaml 2>&1 | grep -i error

# rank 별 로그 분리
mpirun -np 4 --output-filename /tmp/mpi_log ./md_simulation_mpi examples/lj_32000/input.yaml
cat /tmp/mpi_log.1.0  # rank 0 로그
```

| 증상 | 원인 | 조치 |
|------|------|------|
| `MPI_Init` 이후 hang | Barrier 불균형 | 모든 rank가 `MPI_Barrier` 도달하는지 확인 |
| rank간 에너지 불일치 | 도메인 분해 경계 ghost 교환 누락 | halo exchange 로직 점검 |
| 결과가 CPU 단독과 다름 | `MPI_Allreduce` 누락 | 전체 에너지 합산 부분 확인 |
| `MPI_Finalize` 후 hang | 비동기 send/recv 미완료 | `MPI_Wait` / `MPI_Waitall` 추가 |

**MPI 도메인 분해 핵심 검증 코드 삽입 위치**

```cpp
// force_compute() 전후 디버깅 출력 추가
#ifdef DEBUG_MPI
  double local_ke, global_ke;
  local_ke = compute_kinetic_energy();
  MPI_Allreduce(&local_ke, &global_ke, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0)
    printf("[DEBUG MPI] step=%d global_KE=%.6f\n", step, global_ke);
#endif
```

---

## 4. CUDA 빌드 & 실행

### 4-1. 빌드

```bash
echo "=== [STEP 3] CUDA 빌드 ==="
make clean && make cuda 2>&1
echo "Exit code: $?"
ls -lh md_simulation_cuda 2>/dev/null || echo "[ERROR] CUDA 바이너리 없음"
```

**CUDA 빌드 실패 체크리스트**

```bash
# nvcc 아키텍처 플래그 확인 (GPU compute capability 맞춰야 함)
nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null
# 예: 8.6 → -arch=sm_86

# 현재 Makefile의 아키텍처 플래그 확인
grep -E "arch|sm_|compute_" Makefile
```

| 증상 | 원인 | 조치 |
|------|------|------|
| `nvcc: command not found` | CUDA toolkit PATH 문제 | `export PATH=/usr/local/cuda/bin:$PATH` |
| `error: identifier "__device__" is undefined` | `.cpp` 파일을 nvcc로 컴파일 안 함 | CUDA 커널 포함 파일 확장자 `.cu`로 변경 |
| `ptxas error: Unresolved extern function` | `__global__` 함수 forward declaration 누락 | 헤더에 선언 추가 |
| `compute capability mismatch` | `-arch` 플래그 불일치 | GPU capability에 맞게 수정 |
| `undefined reference to cudaMemcpy` | `-lcudart` 누락 | Makefile 링크 플래그에 `-lcudart` 추가 |

### 4-2. 실행

```bash
echo "=== [STEP 3] CUDA 실행 ==="
./md_simulation_cuda examples/lj_32000/input.yaml 2>&1
echo "Exit code: $?"
```

**CUDA 런타임 실패 체크리스트**

```bash
# cuda-memcheck (또는 compute-sanitizer)
cuda-memcheck ./md_simulation_cuda examples/lj_32000/input.yaml 2>&1 | head -50
# CUDA 12+
compute-sanitizer ./md_simulation_cuda examples/lj_32000/input.yaml 2>&1 | head -50
```

**CUDA 커널에 삽입할 디버깅 매크로**

```cuda
// 모든 CUDA API 호출 래핑
#define CUDA_CHECK(call) \
  do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
      fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n", \
              __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

// 사용 예
CUDA_CHECK(cudaMalloc(&d_pos, N * sizeof(double3)));
CUDA_CHECK(cudaMemcpy(d_pos, h_pos, N * sizeof(double3), cudaMemcpyHostToDevice));
```

| 증상 | 원인 | 조치 |
|------|------|------|
| `cudaErrorInvalidDevice` | GPU 없음 or CUDA_VISIBLE_DEVICES 문제 | `nvidia-smi` 확인, `CUDA_VISIBLE_DEVICES=0` 설정 |
| `cudaErrorIllegalAddress` | 배열 범위 초과 (GPU) | `compute-sanitizer` 로 확인 |
| `cudaErrorLaunchFailure` | 블록/그리드 크기 초과 | `cudaDeviceProp.maxThreadsPerBlock` 이내로 설정 |
| NVE 에너지 드리프트 큼 | FP32 vs FP64 혼용 | 커널 내 `float` → `double` 통일 |
| CPU 결과와 숫자 불일치 | 원자적 연산 누락 | `atomicAdd` 사용 또는 reduction 커널 추가 |

**CUDA 커널 블록/그리드 크기 디버깅**

```cuda
// 커널 실행 전 출력
int block_size = 256;
int grid_size  = (N + block_size - 1) / block_size;
fprintf(stderr, "[DEBUG CUDA] N=%d  grid=%d  block=%d\n", N, grid_size, block_size);
force_kernel<<<grid_size, block_size>>>(d_pos, d_force, N, box_len, cutoff);
CUDA_CHECK(cudaGetLastError());   // 런치 오류 즉시 포착
CUDA_CHECK(cudaDeviceSynchronize()); // 커널 완료 대기
```

---

## 5. MPI + CUDA 빌드 & 실행

### 5-1. 빌드

```bash
echo "=== [STEP 4] MPI+CUDA 빌드 ==="
make clean && make cuda_mpi 2>&1   # 타겟 이름 확인 후 조정
echo "Exit code: $?"
ls -lh md_simulation_cuda_mpi 2>/dev/null || echo "[ERROR] MPI+CUDA 바이너리 없음"
```

**MPI+CUDA Makefile 예시 (참고용)**

```makefile
NVCC     = nvcc
MPICXX   = mpicxx
CUDA_ARCH = -arch=sm_80   # GPU에 맞게 수정

# mpicxx 컴파일 플래그 추출
MPI_CFLAGS  := $(shell mpicxx --showme:compile)
MPI_LDFLAGS := $(shell mpicxx --showme:link)

cuda_mpi: main.cu integrator.cu force.cu
	$(NVCC) $(CUDA_ARCH) -Xcompiler "$(MPI_CFLAGS)" \
	    main.cu integrator.cu force.cu \
	    -Xlinker "$(MPI_LDFLAGS)" \
	    -lcudart -lyaml-cpp \
	    -o md_simulation_cuda_mpi
```

| 증상 | 원인 | 조치 |
|------|------|------|
| `nvcc` + `mpicxx` 동시 사용 시 링크 오류 | nvcc가 MPI 헤더를 못 찾음 | `--showme:compile` 결과를 `-Xcompiler`로 전달 |
| `multiple definition of MPI_Init` | 두 컴파일러 모두 MPI 링크 | 링크는 `nvcc` 한 쪽에서만 수행 |
| `cudaSetDevice` 오류 (multi-node) | rank당 GPU 매핑 누락 | `cudaSetDevice(rank % num_gpus)` 추가 |

### 5-2. 실행

```bash
echo "=== [STEP 4] MPI+CUDA 실행 (rank=2, GPU 각 1개) ==="
mpirun -np 2 ./md_simulation_cuda_mpi examples/lj_32000/input.yaml 2>&1
echo "Exit code: $?"
```

**MPI+CUDA 런타임 핵심 디버깅**

```bash
# 각 rank가 어떤 GPU를 쓰는지 확인
mpirun -np 4 bash -c 'echo "rank=$OMPI_COMM_WORLD_RANK  GPU=$CUDA_VISIBLE_DEVICES"'

# GPU-aware MPI 지원 확인 (UCX 기반 OpenMPI)
ompi_info --parsable --all | grep -i "btl.*cuda\|ucx.*cuda" 2>/dev/null
```

**rank당 GPU 할당 코드 (필수 삽입)**

```cpp
// MPI_Init 직후 삽입
int rank, nprocs, num_gpus;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
cudaGetDeviceCount(&num_gpus);
cudaSetDevice(rank % num_gpus);  // ← 이 줄 없으면 모든 rank가 GPU 0 사용
fprintf(stderr, "[DEBUG] rank=%d  using GPU=%d / %d\n", rank, rank % num_gpus, num_gpus);
```

---

## 6. NVE 물리 검증 (공통)

빌드·실행이 성공한 후에도 물리가 틀릴 수 있음.

```bash
# 각 바이너리 실행 후 에너지 로그 비교
for BIN in md_simulation_cpu md_simulation_mpi md_simulation_cuda md_simulation_cuda_mpi; do
  if [ -x "./$BIN" ]; then
    echo "=== $BIN 에너지 로그 ==="
    ./$BIN examples/lj_32000/input.yaml 2>&1 | grep -E "step|energy|KE|PE|Total|drift" | head -20
  fi
done
```

| 검증 항목 | 기준값 | 조치 |
|-----------|--------|------|
| 총 에너지 드리프트 | < 0.01% / 1000 step | `dt` 감소, 적분기 확인 (Velocity Verlet 권장) |
| KE + PE = const | 허용 오차 1e-4 | force 계산 정확도 확인 |
| 모멘텀 보존 | ΔP < 1e-10 | 뉴턴 3법칙 쌍 누락 여부 확인 |
| 압력 값 | LJ 액체 기준 양수 | virial 계산 부호 확인 |

**Velocity Verlet 구현 검증**

```cpp
// 올바른 순서 확인
// 1. v(t + dt/2) = v(t) + 0.5*dt*a(t)
// 2. r(t + dt)   = r(t) + dt*v(t + dt/2)
// 3. a(t + dt)   계산 (force 재계산)
// 4. v(t + dt)   = v(t + dt/2) + 0.5*dt*a(t + dt)
```

---

## 7. 성능 측정 로그 포맷 확인

```bash
echo "=== 성능 로그 포맷 검증 ==="
for BIN in md_simulation_cpu md_simulation_cuda; do
  if [ -x "./$BIN" ]; then
    LAST_LINES=$(./$BIN examples/lj_32000/input.yaml 2>&1 | tail -3)
    echo "$BIN 마지막 3줄:"
    echo "$LAST_LINES"
    echo "$LAST_LINES" | grep -q "Total execution time" \
      && echo "[OK] 성능 줄 존재" \
      || echo "[FAIL] '=== Total execution time: XX.XXX seconds ===' 줄 없음 – 출력 코드 확인 필요"
  fi
done
```

**출력 코드 예시 (main 함수 끝)**

```cpp
auto t_end = std::chrono::high_resolution_clock::now();
double elapsed = std::chrono::duration<double>(t_end - t_start).count();
// 마지막에서 두 번째 줄 (빈 줄 전)
printf("=== Total execution time: %.3f seconds ===\n", elapsed);
printf("\n");  // 마지막 줄 (빈 줄)
```

---

## 8. 반복 디버깅 루프 (자동화)

```bash
#!/bin/bash
# debug_loop.sh – 에러가 있어도 멈추지 않고 계속 진행

PASS=0; FAIL=0
run_test() {
  local label="$1"; shift
  echo ""
  echo "========================================"
  echo "  TEST: $label"
  echo "========================================"
  "$@" 2>&1
  local code=$?
  if [ $code -eq 0 ]; then
    echo "[PASS] $label (exit=$code)"
    ((PASS++))
  else
    echo "[FAIL] $label (exit=$code) – 계속 진행"
    ((FAIL++))
  fi
}

make clean
run_test "CPU 빌드"       make
run_test "CPU 실행"       ./md_simulation_cpu examples/lj_32000/input.yaml
run_test "MPI 빌드"       make mpi
run_test "MPI 실행 np=4"  mpirun -np 4 ./md_simulation_mpi examples/lj_32000/input.yaml
run_test "CUDA 빌드"      make cuda
run_test "CUDA 실행"      ./md_simulation_cuda examples/lj_32000/input.yaml
run_test "MPI+CUDA 빌드"  make cuda_mpi
run_test "MPI+CUDA 실행"  mpirun -np 2 ./md_simulation_cuda_mpi examples/lj_32000/input.yaml

echo ""
echo "========================================"
echo "  결과: PASS=$PASS  FAIL=$FAIL"
echo "========================================"
```

```bash
chmod +x debug_loop.sh && bash debug_loop.sh 2>&1 | tee debug_run_$(date +%Y%m%d_%H%M%S).log
```

---

## 9. 빠른 참조 – 에러 코드별 조치

| 에러 / 증상 | 즉시 실행할 명령 |
|-------------|-----------------|
| 빌드 에러 전체 보기 | `make 2>&1 \| tee build.log` |
| segfault 위치 | `gdb ./md_simulation_cpu -ex run --args examples/lj_32000/input.yaml` |
| CUDA 메모리 오류 | `compute-sanitizer --tool memcheck ./md_simulation_cuda examples/lj_32000/input.yaml` |
| MPI hang | `mpirun -np 4 --timeout 60 ./md_simulation_mpi examples/lj_32000/input.yaml` |
| 에너지 드리프트 | `dt`를 절반으로 줄이고 재실행 |
| GPU 인식 안 됨 | `CUDA_VISIBLE_DEVICES=0 ./md_simulation_cuda examples/lj_32000/input.yaml` |

---

> **이 파일을 Copilot에 읽힌 후**: 각 섹션의 bash 블록을 순서대로 실행하고, `[FAIL]` 이 나오면 해당 섹션의 체크리스트와 코드 스니펫을 참고하여 소스를 수정한 뒤 `debug_loop.sh`를 다시 실행하세요.

---

---

# MD Simulation Performance Benchmarking & Optimization Guide
> **목적**: CPU / MPI / CUDA / MPI+CUDA 성능을 수치로 비교하고, 병목을 찾아 반복 개선

---

## 10. 성능 벤치마크 수집

### 10-1. 자동 벤치마크 스크립트

```bash
#!/bin/bash
# bench.sh – 각 바이너리의 실행 시간을 추출해 비교표 출력
# 사용: bash bench.sh 2>&1 | tee bench_$(date +%Y%m%d_%H%M%S).log

INPUT="examples/lj_32000/input.yaml"
RESULT_FILE="bench_results.tsv"

echo -e "Binary\tRanks\tTime(s)" > $RESULT_FILE

run_bench() {
  local label="$1"
  local cmd=("${@:2}")
  echo "--- Running: $label ---"
  local output
  output=$("${cmd[@]}" 2>&1)
  echo "$output" | tail -5          # 마지막 5줄 미리보기
  local t
  t=$(echo "$output" | grep -oP "Total execution time: \K[0-9]+\.[0-9]+")
  if [ -z "$t" ]; then
    t="FAILED"
    echo "[WARN] 시간 추출 실패 – 출력 포맷 확인 필요"
  fi
  echo -e "$label\t${cmd[1]:-1}\t$t" >> $RESULT_FILE
  echo ""
}

# CPU 단독
[ -x "./md_simulation_cpu" ] && \
  run_bench "CPU-1core"  ./md_simulation_cpu $INPUT

# OpenMP (CPU 멀티스레드) – OMP_NUM_THREADS 변경
for T in 2 4 8; do
  [ -x "./md_simulation_cpu" ] && \
    run_bench "CPU-OMP${T}t" env OMP_NUM_THREADS=$T ./md_simulation_cpu $INPUT
done

# MPI 스케일링
for NP in 1 2 4 8; do
  [ -x "./md_simulation_mpi" ] && \
    run_bench "MPI-np${NP}" mpirun -np $NP ./md_simulation_mpi $INPUT
done

# CUDA
[ -x "./md_simulation_cuda" ] && \
  run_bench "CUDA"  ./md_simulation_cuda $INPUT

# MPI + CUDA
for NP in 1 2 4; do
  [ -x "./md_simulation_cuda_mpi" ] && \
    run_bench "MPI+CUDA-np${NP}" mpirun -np $NP ./md_simulation_cuda_mpi $INPUT
done

echo ""
echo "========= 벤치마크 결과 ========="
column -t $RESULT_FILE
echo "================================="
```

```bash
chmod +x bench.sh && bash bench.sh
```

### 10-2. 예상 결과 해석 기준 (lj_32000 기준)

| 구성 | 기대 속도 비 | 비고 |
|------|-------------|------|
| CPU 1코어 | 1× (기준) | |
| CPU 4코어 (OMP) | 3–3.5× | 캐시 경합으로 선형 스케일링 안 됨 |
| MPI np=4 | 3–3.8× | 도메인 분해 통신 오버헤드 포함 |
| CUDA (단일 GPU) | 10–30× | N=32000에서 GPU 활용률 낮을 수 있음 |
| MPI+CUDA np=2 | 20–50× | GPU 간 통신 overlap 시 |

> ⚠️ **N=32000은 CUDA에게 작은 규모**입니다. GPU 가속 효과가 기대보다 낮으면 N을 100K 이상으로 늘려 측정하세요.

---

## 11. 병목 프로파일링

### 11-1. CPU 프로파일링 (gprof / perf)

```bash
# gprof 빌드
g++ -O2 -pg -o md_simulation_cpu_prof <소스파일들> -lyaml-cpp
./md_simulation_cpu_prof examples/lj_32000/input.yaml
gprof md_simulation_cpu_prof gmon.out | head -40

# perf (Linux)
perf stat ./md_simulation_cpu examples/lj_32000/input.yaml 2>&1
perf record -g ./md_simulation_cpu examples/lj_32000/input.yaml
perf report --stdio | head -30
```

**핵심 확인 항목**

| 함수 | 비율 목표 | 비율 높으면 조치 |
|------|-----------|-----------------|
| `force_compute` / `compute_forces` | < 90% | 정상 (force가 병목인 것이 정상) |
| `neighbor_list_build` | > 10% | 갱신 주기 늘리기 (5~20 스텝마다) |
| `apply_pbc` (주기 경계) | > 5% | SIMD 벡터화 또는 조건 분기 제거 |
| `output_write` | > 3% | 출력 빈도 줄이기 |

### 11-2. CUDA 프로파일링 (Nsight Systems / nvprof)

```bash
# Nsight Systems (CUDA 11+)
nsys profile --stats=true \
     -o profile_cuda \
     ./md_simulation_cuda examples/lj_32000/input.yaml
nsys stats profile_cuda.nsys-rep 2>/dev/null || \
  nsys analyze profile_cuda.nsys-rep

# 구버전 nvprof
nvprof --print-gpu-summary ./md_simulation_cuda examples/lj_32000/input.yaml 2>&1

# 커널별 occupancy 확인
nvprof --metrics achieved_occupancy,sm_efficiency \
       ./md_simulation_cuda examples/lj_32000/input.yaml 2>&1
```

**Nsight 결과에서 확인할 항목**

| 지표 | 목표값 | 낮으면 조치 |
|------|--------|------------|
| SM Efficiency | > 80% | 블록 수 늘리기, 작은 커널 fuse |
| Achieved Occupancy | > 50% | 레지스터 수 줄이기, `__launch_bounds__` 추가 |
| Memory Throughput | > 60% peak | coalesced access 확인 |
| H↔D 전송 비율 | < 5% 전체 시간 | 전송 횟수 줄이기, async 전송 사용 |

---

## 12. CPU 최적화

### 12-1. 컴파일러 최적화 플래그

```makefile
# Makefile CPU 타겟 최적화 예시
CXXFLAGS = -O3 -march=native -ffast-math -funroll-loops -fopenmp

# 추가 옵션 (Intel CPU)
CXXFLAGS += -mavx2 -mfma

# 프로파일 기반 최적화 (PGO) – 2단계 빌드
# 1단계: 프로파일 수집 빌드
g++ -O2 -fprofile-generate $(CXXFLAGS) -o md_pgo_train <소스> -lyaml-cpp
./md_pgo_train examples/lj_32000/input.yaml
# 2단계: 프로파일 반영 빌드
g++ -O3 -fprofile-use $(CXXFLAGS) -o md_simulation_cpu <소스> -lyaml-cpp
```

### 12-2. Neighbor List 최적화 (Verlet List)

```cpp
// 현재 O(N²) force loop → Verlet List O(N) 으로 변환
// neighbor_list.h 에 추가

struct NeighborList {
  std::vector<int> head;    // 입자 i의 이웃 시작 인덱스
  std::vector<int> list;    // 이웃 입자 j 번호 목록
  double skin;              // cutoff 확장 (cutoff + skin, 보통 0.3σ)
  int    rebuild_interval;  // 몇 스텝마다 재빌드 (5~20 권장)

  // 재빌드 조건: 모든 입자가 skin/2 이상 이동했을 때
  bool needs_rebuild(const double* disp) {
    return *std::max_element(disp, disp+N) > skin / 2.0;
  }
};
```

```cpp
// force 루프 (Verlet List 사용)
for (int i = 0; i < N; i++) {
  for (int idx = nl.head[i]; idx < nl.head[i+1]; idx++) {
    int j = nl.list[idx];
    // LJ 계산 ...
  }
}
// 빌드는 rebuild_interval 스텝마다만 호출
if (step % nl.rebuild_interval == 0 || nl.needs_rebuild(max_disp))
  nl.build(positions, box_len, cutoff + nl.skin);
```

### 12-3. SIMD 벡터화 확인

```bash
# 벡터화 여부 확인 (gcc)
g++ -O3 -march=native -fopt-info-vec-optimized \
    <소스파일들> -lyaml-cpp 2>&1 | grep "force\|loop"

# 벡터화 실패 이유 확인
g++ -O3 -march=native -fopt-info-vec-missed \
    <소스파일들> 2>&1 | head -30
```

---

## 13. CUDA 커널 최적화

### 13-1. 메모리 접근 패턴 개선 (AoS → SoA)

```cuda
// ❌ Array of Structures (AoS) – coalescing 불리
struct Particle { double x, y, z, vx, vy, vz; };
Particle* d_particles;  // d_particles[i].x 접근 시 stride 발생

// ✅ Structure of Arrays (SoA) – coalesced access
struct ParticlesSoA {
  double *x, *y, *z;    // 위치
  double *vx, *vy, *vz; // 속도
  double *fx, *fy, *fz; // 힘
};
// d_pos.x[i], d_pos.x[i+1] ... 연속 메모리 → 최대 throughput
```

### 13-2. Shared Memory 활용 (타일링)

```cuda
#define TILE_SIZE 256

__global__ void force_kernel_tiled(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    double* fx, double* fy, double* fz,
    int N, double box_len, double cutoff2)
{
  __shared__ double sx[TILE_SIZE], sy[TILE_SIZE], sz[TILE_SIZE];

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  double xi = (i < N) ? x[i] : 0.0;
  double yi = (i < N) ? y[i] : 0.0;
  double zi = (i < N) ? z[i] : 0.0;
  double fxi = 0.0, fyi = 0.0, fzi = 0.0;

  for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; tile++) {
    int j = tile * TILE_SIZE + threadIdx.x;
    sx[threadIdx.x] = (j < N) ? x[j] : 0.0;
    sy[threadIdx.x] = (j < N) ? y[j] : 0.0;
    sz[threadIdx.x] = (j < N) ? z[j] : 0.0;
    __syncthreads();

    for (int k = 0; k < TILE_SIZE && (tile * TILE_SIZE + k) < N; k++) {
      double dx = xi - sx[k];
      double dy = yi - sy[k];
      double dz = zi - sz[k];
      // 최소 이미지 + LJ 계산 ...
    }
    __syncthreads();
  }

  if (i < N) { fx[i] = fxi; fy[i] = fyi; fz[i] = fzi; }
}
```

### 13-3. 커널 최적화 빠른 체크리스트

```cuda
// 1. launch bounds 명시 (레지스터 수 제어)
__launch_bounds__(256, 4)  // 블록당 최대 스레드, SM당 최소 블록
__global__ void force_kernel(...) { ... }

// 2. restrict 힌트 (포인터 aliasing 없음을 컴파일러에 알림)
__global__ void force_kernel(
    const double* __restrict__ x,
    double* __restrict__ fx, ...) { ... }

// 3. fast math 디바이스 함수
// rsqrtf 대신 __drcp_rn (double), rsqrt 사용
double inv_r  = rsqrt(r2);      // 일반
double inv_r2 = __drcp_rn(r2);  // fast (약간 정확도 손실)

// 4. warp divergence 제거 – if/else 최소화
// ❌ 분기 발생
if (r2 < cutoff2) { ... }
// ✅ predicated 실행
double lj = (r2 < cutoff2) ? compute_lj(r2) : 0.0;
```

### 13-4. 비동기 메모리 전송 (CPU-GPU Overlap)

```cuda
// 스트림 2개로 계산과 전송 겹치기
cudaStream_t stream_compute, stream_transfer;
cudaStreamCreate(&stream_compute);
cudaStreamCreate(&stream_transfer);

// 홀수 스텝: A 계산 + B 전송 동시
force_kernel<<<grid, block, 0, stream_compute>>>(d_A_pos, d_A_force, ...);
cudaMemcpyAsync(d_B_pos, h_B_pos, size, cudaMemcpyHostToDevice, stream_transfer);

cudaStreamSynchronize(stream_compute);
cudaStreamSynchronize(stream_transfer);
```

---

## 14. MPI 통신 최적화

### 14-1. 통신 오버헤드 측정

```cpp
// halo exchange 시간 측정
double t_comm_start = MPI_Wtime();
exchange_halo(positions, velocities, comm);
double t_comm = MPI_Wtime() - t_comm_start;

double t_comm_max;
MPI_Reduce(&t_comm, &t_comm_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
if (rank == 0)
  printf("[PERF] step=%d  halo_exchange_max=%.4f ms\n", step, t_comm_max*1e3);
```

**이상적인 통신 비율**

| np | 통신 시간 비율 | 초과 시 조치 |
|----|--------------|-------------|
| 2  | < 5%  | |
| 4  | < 10% | |
| 8  | < 20% | 도메인 분해 방향 재검토 |
| 16 | < 35% | cut-off 줄이기, 비동기 통신 |

### 14-2. 비동기 통신 (계산-통신 overlap)

```cpp
// ❌ 동기 통신 (blocking)
MPI_Sendrecv(send_buf, count, MPI_DOUBLE, dst, 0,
             recv_buf, count, MPI_DOUBLE, src, 0,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
compute_inner_forces();  // 통신 완료 후에야 시작

// ✅ 비동기 통신 (non-blocking)
MPI_Request req[2];
MPI_Isend(send_buf, count, MPI_DOUBLE, dst, 0, MPI_COMM_WORLD, &req[0]);
MPI_Irecv(recv_buf, count, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, &req[1]);
compute_inner_forces();  // 통신과 동시에 내부 입자 계산
MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
compute_boundary_forces();  // 통신 완료 후 경계 입자만 처리
```

### 14-3. MPI 스케일링 효율 계산

```bash
# bench_results.tsv 에서 MPI 스케일링 효율 계산
python3 - <<'EOF'
import csv, sys

data = {}
with open("bench_results.tsv") as f:
    for row in csv.DictReader(f, delimiter='\t'):
        if row['Binary'].startswith('MPI'):
            try:
                data[int(row['Ranks'])] = float(row['Time(s)'])
            except ValueError:
                pass

if 1 in data:
    t1 = data[1]
    print(f"{'np':>4}  {'Time(s)':>10}  {'Speedup':>8}  {'Efficiency':>10}")
    for np in sorted(data):
        sp = t1 / data[np]
        eff = sp / np * 100
        print(f"{np:>4}  {data[np]:>10.3f}  {sp:>8.2f}x  {eff:>9.1f}%")
EOF
```

---

## 15. MPI+CUDA 최적화

### 15-1. GPU-Aware MPI 활성화

```bash
# GPU-Aware MPI 여부 확인
python3 -c "import mpi4py; print(mpi4py.__version__)" 2>/dev/null
ompi_info | grep -i "cuda\|ucx" 2>/dev/null

# GPU-Aware MPI 사용 시 cudaMemcpy 불필요 (GPU 메모리 직접 전달)
export OMPI_MCA_btl=^openib
export OMPI_MCA_pml=ucx
export UCX_TLS=rc,sm,cuda_copy,cuda_ipc,gdr_copy
mpirun -np 2 ./md_simulation_cuda_mpi examples/lj_32000/input.yaml 2>&1
```

```cpp
// GPU-Aware MPI 사용 예 (d_send_buf은 device 포인터)
#ifdef GPU_AWARE_MPI
  MPI_Sendrecv(d_send_buf, count, MPI_DOUBLE, dst, 0,   // device → device
               d_recv_buf, count, MPI_DOUBLE, src, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
#else
  cudaMemcpy(h_send_buf, d_send_buf, bytes, cudaMemcpyDeviceToHost);
  MPI_Sendrecv(h_send_buf, count, MPI_DOUBLE, dst, 0,   // host 경유
               h_recv_buf, count, MPI_DOUBLE, src, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  cudaMemcpy(d_recv_buf, h_recv_buf, bytes, cudaMemcpyHostToDevice);
#endif
```

### 15-2. 단계별 최적화 우선순위

```
MPI+CUDA 최적화 순서:
  1. cudaSetDevice(rank % num_gpus)         ← 필수, 없으면 모든 rank가 GPU 0 사용
  2. SoA 메모리 레이아웃                    ← coalesced access, 10~30% 개선
  3. Neighbor List (CPU+GPU 모두)           ← O(N²) → O(N), 입자 많을수록 효과 큼
  4. Shared memory 타일링                   ← GPU memory bandwidth 포화 방지
  5. 비동기 MPI (Isend/Irecv + 계산 overlap) ← 통신 숨기기
  6. GPU-Aware MPI                          ← H↔D 복사 제거, NVLink 환경에서 극적 효과
  7. CUDA Stream 파이프라이닝               ← 멀티 GPU 로드 밸런싱
```

---

## 16. 성능 개선 반복 루프

```bash
#!/bin/bash
# optimize_loop.sh – 최적화 전/후 자동 비교

INPUT="examples/lj_32000/input.yaml"

get_time() {
  "$@" 2>&1 | grep -oP "Total execution time: \K[0-9]+\.[0-9]+"
}

echo "=== 최적화 전 기준선 측정 ==="
T_CPU_BASE=$(get_time ./md_simulation_cpu $INPUT)
T_CUDA_BASE=$(get_time ./md_simulation_cuda $INPUT)
echo "CPU baseline : ${T_CPU_BASE}s"
echo "CUDA baseline: ${T_CUDA_BASE}s"

echo ""
echo "=== 소스 수정 후 재빌드 & 재측정 ==="
echo "(이 지점에서 Copilot이 소스를 수정하고 아래를 실행)"

make clean && make && make cuda 2>&1 | tail -3

T_CPU_NEW=$(get_time ./md_simulation_cpu $INPUT)
T_CUDA_NEW=$(get_time ./md_simulation_cuda $INPUT)

echo ""
echo "========= 최적화 결과 ========="
python3 - <<EOF
base_cpu  = float("${T_CPU_BASE:-0}")  or 1
base_cuda = float("${T_CUDA_BASE:-0}") or 1
new_cpu   = float("${T_CPU_NEW:-0}")   or 1
new_cuda  = float("${T_CUDA_NEW:-0}")  or 1

print(f"  CPU : {base_cpu:.3f}s → {new_cpu:.3f}s  ({base_cpu/new_cpu:.2f}× 개선)")
print(f"  CUDA: {base_cuda:.3f}s → {new_cuda:.3f}s  ({base_cuda/new_cuda:.2f}× 개선)")
print(f"  CPU vs CUDA 가속비: {new_cpu/new_cuda:.1f}×")
EOF
echo "================================="
```

```bash
chmod +x optimize_loop.sh && bash optimize_loop.sh 2>&1 | tee opt_$(date +%Y%m%d_%H%M%S).log
```

---

## 17. 성능 개선 빠른 참조표

| 최적화 항목 | 대상 | 예상 개선 | 난이도 |
|------------|------|----------|--------|
| `-O3 -march=native -ffast-math` | CPU | 1.5–2× | ⭐ |
| Neighbor List (Verlet) | CPU/GPU | 2–10× (N>10K) | ⭐⭐ |
| OpenMP 병렬화 | CPU | 코어 수 × 0.7–0.9 | ⭐ |
| AoS → SoA 메모리 레이아웃 | CUDA | 1.3–2× | ⭐⭐ |
| Shared memory 타일링 | CUDA | 1.2–1.8× | ⭐⭐⭐ |
| `__launch_bounds__` + restrict | CUDA | 1.1–1.3× | ⭐ |
| 비동기 MPI (Isend/Irecv) | MPI | 1.2–1.5× (np≥8) | ⭐⭐ |
| GPU-Aware MPI | MPI+CUDA | 1.5–3× (NVLink) | ⭐⭐ |
| PGO (프로파일 기반 최적화) | CPU | 1.1–1.3× | ⭐⭐ |
| CUDA Stream 파이프라이닝 | MPI+CUDA | 1.2–1.4× | ⭐⭐⭐ |

> **Copilot 지시 방법**: `optimize_loop.sh`를 실행 → 시간 측정 → 위 표에서 난이도 ⭐ 항목부터 순서대로 적용 → 재측정 → 개선 없으면 다음 항목으로 이동. 개선이 확인된 항목만 유지하고 커밋.
