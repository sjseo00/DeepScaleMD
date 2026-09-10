# 전체 소스코드 구동 과정

> **현재 architecture:** DeepScaleMD는 Cartesian topology, halo communication,
> migration을 standalone `dsDecompX` library에 의존한다. 아래의 legacy
> `domain_decomposition.cpp/.h` 참조는 역사적 설명이다. 지원되는
> integration은 `dsDecompX/docs/integration.md`를 참고한다.

DeepScaleMD의 파일 구성, 클래스 계층, 함수 호출 흐름을 정리한다.

---

## 1. 디렉터리 구조

```
DeepScaleMD/
├── src/                        # C++ 소스 파일
│   ├── main.cpp                # 진입점
│   ├── md_simulation.cpp/.h    # 최상위 시뮬레이션 관리자
│   ├── md_system.cpp/.h        # 원자 데이터 컨테이너 (CPU)
│   ├── integrator.cpp/.h       # 수치 적분기 (Velocity-Verlet 등)
│   ├── force_calculator.cpp/.h # 힘 계산 기반 클래스
│   ├── force_calculator_cuda.cu/.h  # GPU 내장 힘 계산
│   ├── pme_calculator.cpp/.h   # PME 정전기 (cuFFT)
│   ├── neighbor_list*.cpp/.cu  # 이웃 목록 관리
│   ├── domain_decomposition.cpp/.h  # MPI 도메인 분해 (legacy, dsDecompX로 대체됨)
│   ├── mpi_optimizer.cpp/.h    # MPI 집합연산 최적화
│   ├── mlip_runtime.cpp/.h     # C++ TorchScript MLIP 실행기
│   ├── python_force_interface.cpp/.h  # Python 임베딩 인터페이스
│   ├── execution_plan.cpp/.h   # 시뮬레이션 단계 실행 계획
│   ├── md_data_sources.cpp/.h  # 입출력 (XYZ 파싱 등)
│   └── ...
├── python/                     # Python 레이어 (MLIP 어댑터)
│   ├── mlip_interface.py       # MLIPAdapter 기반 클래스 + ADAPTER_REGISTRY
│   ├── ml_forces.py            # C++ PythonForceInterface가 호출하는 진입점
│   ├── uma_adapter.py          # UMA / fairchem 어댑터
│   ├── mace_adapter.py         # MACE-MP 어댑터
│   ├── nequip_adapter.py       # NequIP 어댑터
│   ├── chgnet_adapter.py       # CHGNet 어댑터
│   └── ...
├── include/                    # 공개 헤더
├── bin/                        # 빌드 결과물
│   └── md_simulation_cuda
├── examples/                   # 예시 입력 파일
└── docs/                       # 본 문서들
```

---

## 2. 클래스 계층

```
MDSimulation                          ← 최상위 조율자
├── MDSystem                          ← 원자 데이터 (CPU, SOA)
│   ├── positions[], velocities[], forces[]
│   ├── masses[], charges[], atom_types[]
│   └── topology 정보 (본드, 각도, ...)
│
├── ForceCalculator (abstract base)
│   ├── ForceCalculatorCUDA           ← builtin LJ + PME (GPU)
│   │   ├── NeighborListBuilder       ← Verlet-cell or Morton
│   │   └── PMECalculator             ← cuFFT PME
│   └── MlipRuntime                   ← C++ TorchScript MLIP
│
├── PythonForceInterface              ← Python 임베딩 (python mode)
│   └── (python/ml_forces.py → MLIPAdapter 서브클래스)
│
├── Integrator (abstract base)
│   ├── VelocityVerletIntegrator
│   ├── VerletIntegrator
│   └── LeapFrogIntegrator
│
├── dsDecompX                          ← 도메인 분해 + halo/migration (외부 라이브러리)
│
└── MPIOptimizer                      ← MPI_Iallreduce 배치
```

---

## 3. 실행 흐름 전체 다이어그램

```
main()
  │
  ├─ MPI_Init_thread(MPI_THREAD_MULTIPLE)
  ├─ config_file = argv[1]
  │
  └─ MDSimulation sim
       ├─ sim.initialize(config_file)
       │     │
       │     ├─ parse_simulation_config()          ← YAML 파싱
       │     ├─ MDDataSources::load_coordinates()  ← .xyz 읽기
       │     ├─ MDDataSources::load_topology()     ← .top 읽기
       │     ├─ initialize_velocities()            ← Maxwell-Boltzmann
       │     ├─ setup_domain_decomposition()       ← MPI_Cart_create
       │     ├─ ForceCalculatorCUDA::initialize()  ← GPU 할당 + 커널 준비
       │     ├─ [python mode] PythonForceInterface::initialize()
       │     │     └─ Py_Initialize → import ml_forces.py
      │     ├─ [libtorch mode] MlipRuntime::initialize()
       │     │     └─ torch::jit::load(model.pt)
       │     └─ create_neighbor_list_builder()
       │
       ├─ sim.run()
       │     │
       │     ├─ prepare_first_step()
       │     │     ├─ build_neighbor_list()           ← 최초 이웃 목록
       │     │     └─ compute_forces()                ← 초기 힘
       │     │
       │     └─ for step = 1 .. total_steps:
       │           single_step(step)
       │
       └─ sim.finalize()
             └─ print_timing_breakdown()
```

---

## 4. `single_step()` 내부 상세 흐름

```
single_step(step)
│
├─ [1] 반스텝 속도 + 전체 스텝 위치 업데이트 (GPU)
│     integrator->integrate_on_device(dt/2)
│     → velocity_verlet_half_step_kernel<<<...>>>
│
├─ [2] 이웃 목록 재구축 여부 판단
│     check_displacement_on_device()
│     → thrust::reduce(displacement²) > (verlet_skin/2)²
│     → 조건 충족 시 build_neighbor_list()
│
├─ [3] MPI 하로 교환 (multi-rank인 경우)
│     dsDecompX를 통한 halo exchange
│     → MPI_Isend / MPI_Irecv (ghost atoms)
│
├─ [4] 힘 계산
│     ┌─ [builtin] ForceCalculatorCUDA::compute_forces()
│     │   ├─ zero_forces_kernel<<<...>>>
│     │   ├─ lj_force_kernel<<<...>>>      ← 쌍 포텐셜
│     │   ├─ bond_force_kernel<<<...>>>
│     │   ├─ angle_force_kernel<<<...>>>
│     │   └─ PMECalculator::compute_pme() ← 정전기 (cuFFT)
│     │
│     ├─ [python] PythonForceInterface::compute_ml_forces()
│     │   ├─ GPU → CPU 위치 복사 (pinned buffer)
│     │   ├─ PyObject_CallMethod("calculate_forces")
│     │   │   └─ ml_forces.py → MLIPAdapter.calculate()
│     │   │       ├─ UMAAdapter: FAIRChemCalculator.calculate()
│     │   │       ├─ MACEAdapter: mace_calc.get_forces()
│     │   │       └─ ...
│     │   └─ CPU → GPU 힘 복사
│     │
│     └─ [libtorch] MlipRuntime::compute_forces()
│         ├─ positions → torch::Tensor (GPU)
│         ├─ model_.forward({inputs}) → energy, forces
│         └─ forces → GPU buffer 직접 사용
│
├─ [5] MPI 힘 리덕션 (multi-rank인 경우)
│     MPIOptimizer::allreduce_forces()
│     → MPI_Iallreduce 배치 (non-blocking)
│
├─ [6] 나머지 반스텝 속도 업데이트
│     integrator->complete_velocity_update_on_device(dt/2)
│     → velocity_verlet_second_half_kernel<<<...>>>
│
├─ [7] 온도 제어
│     thermostat->apply_device()
│     → langevin: stochastic_kick_kernel<<<...>>>
│     → nose_hoover: xi_chain 업데이트 + velocity rescale
│
├─ [8] 압력 제어 (NPT)
│     barostat->apply_device()
│     → box_size 업데이트 + position rescale
│
└─ [9] 출력 (step % output_freq == 0)
      output_results(step)
      → KE, PE, T, P 계산 후 콘솔 출력
      → trajectory_output_freq마다 XYZ 기록
```

---

## 5. 초기화 시퀀스 (`initialize()`)

```
parse_simulation_config()     src/md_simulation.cpp
  │  YAML → config struct 전체 채우기
  │
load_coordinates()            src/md_data_sources.cpp
  │  .xyz → MDSystem::positions[], atom_symbols[]
  │
load_topology()               src/md_data_sources.cpp
  │  .top → masses, charges, LJ params, bonds, angles
  │
initialize_velocities()       src/md_simulation.cpp
  │  Maxwell-Boltzmann 분포 → velocities[]
  │  v_cm 제거
  │
setup_domain_decomposition()  src/domain_decomposition.cpp
  │  MPI_Cart_create (3D)
  │  각 rank의 서브도메인 범위 계산
  │  → local atom selection
  │
ForceCalculatorCUDA::initialize()  src/force_calculator_cuda.cu
  │  cudaMalloc (positions_d, forces_d, ...)
  │  cudaMemcpy H→D (초기 데이터)
  │  ConstantMemory 상수 등록 (__constant__)
  │
PythonForceInterface::initialize()  src/python_force_interface.cpp
  │  Py_Initialize()
  │  PyRun_SimpleString("import sys; sys.path.insert(...)")
  │  import ml_forces
  │  adapter = create_adapter(model_type, ...)
  │
  └─ create_neighbor_list_builder()  src/neighbor_list*.cpp
       verlet_cell_list or morton_code 인스턴스 생성
```

---

## 6. 파일별 역할 요약

| 파일 | 클래스/함수 | 역할 |
|------|-----------|------|
| `main.cpp` | `main()` | MPI 초기화, 진입점 |
| `md_simulation.cpp` | `MDSimulation` | 전체 시뮬레이션 흐름 제어 |
| `md_system.cpp` | `MDSystem` | CPU 원자 데이터 컨테이너 |
| `md_data_sources.cpp` | `MDDataSources` | XYZ/TOP 파일 파싱 |
| `integrator.cpp` | `*Integrator` | 뉴턴 운동 방정식 적분 |
| `force_calculator_cuda.cu` | `ForceCalculatorCUDA` | GPU LJ, bond, angle 힘 |
| `neighbor_list_kernels_optimized.cu` | kernels | Verlet-cell 이웃 목록 GPU 커널 |
| `pme_calculator.cpp` | `PMECalculator` | PME 정전기 (cuFFT) |
| `mpi_optimizer.cpp` | `MPIOptimizer` | MPI_Iallreduce 배치 최적화 |
| `mlip_runtime.cpp` | `MlipRuntime` | TorchScript 모델 로드/추론 |
| `python_force_interface.cpp` | `PythonForceInterface` | Python 임베딩 MLIP 호출 |
| `execution_plan.cpp` | `ExecutionPlan` | 단계별 실행 계획 추상화 |
| `python/ml_forces.py` | `calculate_forces()` | C++에서 호출되는 Python 진입점 |
| `python/mlip_interface.py` | `MLIPAdapter`, `ADAPTER_REGISTRY` | 어댑터 기반 클래스 + 등록 |
| `python/uma_adapter.py` | `UMAAdapter` | UMA/fairchem 모델 래퍼 |
| `python/mace_adapter.py` | `MACEAdapter` | MACE-MP 래퍼 |

---

## 7. 데이터 흐름 (GPU 모드)

```
                    CPU (Host)
┌─────────────────────────────────┐
│  MDSystem                       │
│  positions[]  velocities[]      │──H→D 초기화──┐
│  forces[]     masses[]          │              │
└─────────────────────────────────┘              │
                                                 ▼
                    GPU (Device)
┌─────────────────────────────────────────────────┐
│  SOA Buffers (float3*)                          │
│  positions_d[]  velocities_d[]  forces_d[]      │
│  masses_d[]     charges_d[]                     │
│                                                 │
│  Neighbor List Buffer (int*)                    │
│  neighbor_list_d[]  num_neighbors_d[]           │
│                                                 │
│  [builtin] LJ kernel → PME kernel               │
│  [libtorch] torch::Tensor (shared mem)      │
│  [python] D→H copy → Python → H→D copy          │
└─────────────────────────────────────────────────┘
         │                         │
    output_freq              D→H 복사
    trajectory              (positions for XYZ)
```

---

## 8. MPI 멀티 랭크 실행 흐름

```
Rank 0                          Rank 1 ... N
  │                               │
  ├─ 서브도메인 [0, Lx/N)         ├─ 서브도메인 [Lx/N, ...]
  │                               │
  ├─ local atom 힘 계산            ├─ local atom 힘 계산
  │                               │
  ├─ halo exchange (ghost atoms)──┤  MPI_Isend/Irecv
  │  dsDecompX                     │  cutoff 거리 내 경계 원자
  │                               │
  ├─ 경계 힘 계산                  ├─ 경계 힘 계산
  │                               │
  └─ MPIOptimizer::allreduce_forces()
       MPI_Iallreduce (batched, non-blocking)
       → 모든 랭크에 합산된 힘 전달
```

---

## 9. 타이밍 출력 (finalize)

시뮬레이션 완료 후 각 커널/함수별 소요 시간을 출력한다:

```
=== Timing Breakdown ===
Neighbor list build:    0.123 s ( 12.3%)
Force calculation:      0.456 s ( 45.6%)
  LJ kernel:            0.234 s
  PME:                  0.222 s
Integration:            0.045 s (  4.5%)
MPI communication:      0.089 s (  8.9%)
Python MLIP calls:      0.287 s ( 28.7%)
Total wall time:        1.000 s
Performance:            10.0 ns/day  (96-atom water)
```
