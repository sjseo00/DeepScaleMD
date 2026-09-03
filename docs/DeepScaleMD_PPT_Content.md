cat > /home1/kisti/DeepScaleMD/DeepScaleMD_PPT_Content.md << 'ENDOFFILE'
# DeepScaleMD — 발표 자료

## 핵심 메시지 3가지
1. **GPU 최적화 설계** (+ MLP 가속화에도 직접 기여)
2. **대규모 노드에서 효율적으로 동작** (MPI 스케일링)
3. **기존 코드보다 빠르게 동작** (LAMMPS 대비 측정 가능)

---

## 슬라이드 1: 표지

**제목:** DeepScaleMD  
**부제:** GPU-Optimized Molecular Dynamics for Large-Scale Machine Learning Potential Simulations  
**배경:** C++ / CUDA / MPI — 단일 코드베이스, 멀티 GPU, ML Potential 통합

---

## 슬라이드 2: 기존 코드의 한계 — 왜 새로 만들었나?

### GROMACS / LAMMPS의 공통 문제

| 문제 | 현상 |
|------|------|
| GPU ↔ CPU 데이터 왕복 | 매 스텝 cudaMemcpy 발생 → 수십 μs 낭비 |
| 커널 런칭 오버헤드 | CPU가 매 스텝 드라이버에 명령 전달 |
| MPI 통신 블로킹 | Halo Exchange 동안 GPU 계산 정지 |
| ML Potential 통합 없음 | 별도 플러그인, Python 브릿지 없음 |
| 메모리 레이아웃 비효율 | AoS 구조 → GPU 캐시 미스 |

### DeepScaleMD의 답

처음부터 **GPU-first, MPI-scalable, MLP-ready** 로 설계하여 위 문제를 모두 구조적으로 해결

---

## 슬라이드 3: GPU 최적화 ① — SoA 메모리 레이아웃

### AoS (Array of Structs) — 기존 방식의 문제

파티클 데이터를 `{x, y, z, vx, vy, vz, fx, fy, fz, mass, ...}` 구조체로 묶어서 배열로 저장하면, GPU 스레드 32개가 각자의 파티클 x좌표를 읽을 때 메모리상에서 건너뛰며 접근하게 됩니다. 이를 **비연속(Non-coalesced) 접근**이라 하며 캐시 미스가 폭발적으로 발생합니다.

### SoA (Structure of Arrays) — DeepScaleMD 방식

x 좌표끼리, y 좌표끼리, z 좌표끼리 별도 배열로 분리하여 저장합니다. GPU 스레드 32개가 x 배열의 연속된 32개 원소를 동시에 읽으면 **단 한 번의 메모리 트랜잭션(128바이트 캐시 라인)**으로 처리됩니다.

### 효과

- GPU L1/L2 캐시 적중률 최대화
- 메모리 대역폭 활용률이 이론적 최대치에 근접
- 힘 계산·적분 커널 전체에 걸쳐 처리량 직접 향상

---

## 슬라이드 4: GPU 최적화 ② — Morton Z-Order 정렬 + GPU 이웃 리스트

### 문제: 공간 근접 파티클이 메모리상에서 멀리 떨어짐

시뮬레이션 중 파티클은 계속 이동하므로, 공간적으로 가까운 파티클이 메모리 배열에서는 멀리 흩어져 있습니다. 이웃 탐색 시 이 파티클들의 데이터를 읽을 때 캐시 미스가 연쇄 발생합니다.

### 해법: Morton Z-Order 코드로 공간 locality 보존

각 파티클의 3D 격자 좌표 (cx, cy, cz)를 **비트 인터리빙**하여 64비트 Morton 코드를 계산합니다. 이 코드로 파티클 배열을 GPU에서 정렬(Thrust::sort_by_key)하면, **공간적으로 인접한 파티클이 메모리상에서도 인접**하게 됩니다. 이후 셀 범위를 cell_start/cell_end 배열로 O(1)에 조회할 수 있어 이웃 탐색이 빠릅니다.

### Warp-per-Particle 전략

GPU의 Warp(32개 스레드)가 파티클 1개를 협력하여 처리합니다. 27개 이웃 셀(3×3×3 stencil)을 32개 레인이 분담하여 병렬 탐색하므로, 파티클당 이웃 탐색 속도가 최대 32배 향상됩니다.

### GPU-side 변위 판정

이웃 리스트 재빌드 여부를 결정하기 위해 파티클 최대 변위를 CPU로 내려받지 않고, **GPU에서 직접 Thrust 리덕션으로 계산**합니다. Verlet skin의 절반을 초과한 경우에만 재빌드하여 불필요한 rebuild를 최소화합니다.

---

## 슬라이드 5: GPU 최적화 ③ — Kernel Fusion (커널 융합)

### 문제: 동일 데이터를 반복해서 글로벌 메모리에서 로드

Velocity Verlet 적분기는 두 단계로 나뉩니다. 첫 번째 단계에서 위치·속도 데이터를 글로벌 메모리에서 읽고, 두 번째 단계에서 같은 데이터를 다시 읽습니다. GPU 글로벌 메모리 접근은 L1 캐시 히트 대비 수십~수백 배 느리기 때문에 이 중복 접근이 큰 낭비입니다.

### 해법: 하나의 Fused 커널로 통합

위치, 속도, 힘 데이터를 **레지스터에 단 한 번 로드**하고, 레지스터 내에서 가속도 계산 → 속도 업데이트 → 위치 업데이트를 모두 수행한 뒤 **한 번만 글로벌 메모리에 기록**합니다.

- 글로벌 메모리 접근: **읽기 2회 + 쓰기 2회 → 읽기 1회 + 쓰기 1회**
- `__restrict__` 키워드로 컴파일러 alias analysis 활성화 → 레지스터 재사용 최적화

---

## 슬라이드 6: GPU 최적화 ④ — GPU 상주 실행 + CUDA Graphs

### GPU-Resident 실행: 데이터는 GPU에서 떠나지 않는다

위치·속도·힘 배열이 시뮬레이션 전 기간 동안 GPU 메모리에 상주합니다. 온도 조절(Thermostat)과 압력 조절(Barostat) 연산도 `apply_device()` API를 통해 GPU에서 직접 실행됩니다. CPU로의 복사는 trajectory 파일 출력 시에만 발생합니다.

### CUDA Graphs: 반복 루프의 CPU 오버헤드를 제로로

MD 시뮬레이션은 매 스텝 동일한 커널 시퀀스를 반복합니다. 기존 방식은 CPU가 매 스텝 드라이버에 커널 발사 명령을 전달하는 오버헤드(수 μs × 커널 수 × 스텝 수)가 누적됩니다.

CUDA Graphs를 사용하면 전체 시뮬레이션 루프를 **워밍업 후 그래프로 캡처**하고, 이후 매 스텝은 단 한 번의 `cudaGraphLaunch` 호출로 실행됩니다. CUDA 드라이버가 커널 간 의존성을 분석하여 **독립적인 커널들을 자동으로 병렬 실행**합니다. 시스템 변화에 대응하기 위해 100스텝마다 재캡처합니다.

---

## 슬라이드 7: GPU 최적화 ⑤ — 혼합 정밀도 (Mixed Precision)

### 전략: 계산 속도와 물리 정확도를 동시에 잡는다

위치 데이터를 **float(FP32, 4바이트)**로 저장하면 같은 캐시 라인에 double 대비 2배 많은 파티클 데이터가 들어갑니다. FP32 ALU 처리량은 최신 GPU에서 FP64의 2~32배에 달합니다.

힘과 에너지 누산은 **double(FP64)**로 유지하여 에너지 보존 정확도를 보장합니다. 이를 통해 성능과 정확도를 동시에 달성합니다.

### 추가 캐시 최적화

- **`__ldg()` Read-Only Cache**: 이웃 리스트, 파티클 타입 등 반복 읽기 전용 데이터를 L1 read-only 캐시에 고정하여 L2 접근을 줄입니다
- **`__shfl_down_sync` Warp Reduction**: Warp 내 32개 스레드가 에너지/비리얼을 셔플 연산으로 합산하여, atomicAdd 호출 횟수를 32분의 1로 줄입니다

---

## 슬라이드 8: 대규모 MPI 스케일링

### 3D Domain Decomposition (공간 분할)

전체 시뮬레이션 박스를 3D 그리드로 분할하여 각 MPI 랭크가 자신의 서브도메인만 계산합니다. 분할 비율은 코드가 자동으로 최적 탐색합니다 — 정육면체에 가장 가까운 비율을 선택하여 랭크 간 파티클 수 불균형을 최소화합니다. MPI Cartesian Communicator로 주기적 경계를 포함한 26방향 이웃 랭크 통신을 관리합니다.

### 통신-계산 오버랩 (Multi-Stream)

멀티 GPU 환경에서 경계 파티클 데이터(Halo)를 이웃 랭크와 교환하는 동안 GPU 계산이 멈추는 것이 큰 낭비입니다. DeepScaleMD는 파티클을 **내부(Interior)** 와 **경계(Boundary)** 로 분류한 뒤, 세 개의 독립 CUDA 스트림에 올립니다.

- **통신 스트림**: MPI Halo Exchange 비동기 실행
- **내부 스트림**: 통신과 동시에 내부 파티클 힘 계산 수행
- **경계 스트림**: 통신 완료 이벤트를 받은 후 경계 파티클 힘 계산

CUDA Event로 스트림 간 동기화를 제어하여, **MPI 통신 시간이 GPU 계산에 완전히 숨겨집니다.**

### CUDA-aware MPI + NCCL

GPU 버퍼를 CPU로 복사하지 않고 직접 MPI에 전달합니다. NCCL(NVIDIA Collective Communications Library)을 통해 GPU 간 AllReduce를 CPU 개입 없이 링(ring) 토폴로지로 처리합니다. 싱글 랭크 환경에서는 MPI 호출 자체를 생략하는 Fast Path가 있어 단일 노드에서도 오버헤드가 없습니다.

### OpenMP 자동 튜닝

MPI 랭크 수에 맞게 랭크당 OpenMP 스레드 수를 자동 계산합니다 (전체 코어 수 ÷ MPI 랭크 수). `OMP_NUM_THREADS`를 명시적으로 설정한 경우 이를 우선 존중합니다. 이로써 오버서브스크립션(과다 스레드 경쟁)을 방지합니다.

---

## 슬라이드 9: LAMMPS 스타일 성능 분석 내장

시뮬레이션 종료 후 LAMMPS와 동일한 형식의 타이밍 분석을 자동 출력합니다.

- **Section별 분류**: Pair(힘 계산), Neigh(이웃 리스트), Comm(MPI 통신), Output, Modify
- **min/avg/max 시간**: 모든 MPI 랭크에서 수집하여 로드 불균형 즉시 파악
- **%varavg**: 랭크 간 시간 편차 — 값이 클수록 특정 랭크가 병목임을 나타냄
- **%total**: 전체 시간 대비 각 구간의 비율

이를 통해 **어느 구간이 병목인지 정량적으로 진단**하고 최적화 방향을 결정할 수 있습니다.

---

## 슬라이드 10: ML Potential 가속화 — GPU 최적화의 시너지 효과

### MLP-MD 파이프라인에서 각 최적화가 작동하는 위치

GPU-Resident로 위치 데이터가 이미 GPU에 있으므로, ML inference 전 별도의 H2D 복사가 없습니다. Multi-Stream 오버랩 덕분에 Halo 통신이 진행되는 동안 내부 원자들의 ML 전처리를 동시에 수행합니다. CUDA-aware MPI를 통해 ML inference 결과인 힘 텐서를 GPU에서 바로 AllReduce합니다. CUDA Graphs가 ML 전후처리 커널들의 런칭 오버헤드를 흡수합니다.

### MLP 고유 특성과 최적화의 연결

| MLP 특성 | 대응하는 DeepScaleMD 최적화 | 이유 |
|---------|---------------------------|------|
| GNN cutoff 크다 (6~8 Å) | Multi-Stream 오버랩 | cutoff가 클수록 Halo 통신량 증가 → 오버랩 효과 더 큼 |
| 전체 원자 정보 필요 | NCCL AllReduce | 멀티 GPU 간 힘 합산을 CPU 없이 처리 |
| float32 inference | Mixed Precision (FP32) | 데이터 타입 일치 → 타입 변환 오버헤드 제거, Tensor Core 활용 |
| 매 스텝 반복 호출 | CUDA Graphs | ML 전후처리 커널 런칭 오버헤드 누적 방지 |

---

## 슬라이드 11: 요약 — 최적화 전체 지도

### 메모리 최적화
- **SoA 레이아웃**: GPU 메모리 접근 완전 coalesced
- **Morton Z-Order 정렬**: 공간 locality 보존 → 이웃 탐색 캐시 효율
- **혼합 정밀도 FP32/FP64**: 캐시 용량 2배 + ALU 처리량 2~32배
- **`__ldg()` Read-Only Cache**: 반복 읽기 데이터 L1 캐시 고정

### 계산 최적화
- **Kernel Fusion**: 글로벌 메모리 접근 횟수 절반으로 감소
- **CUDA Graphs**: CPU 드라이버 런칭 오버헤드 제로화
- **GPU-Resident 실행**: H2D/D2H 복사 최소화
- **Warp-per-Particle**: 파티클당 이웃 탐색 32배 병렬화

### 통신 최적화
- **3D Domain Decomposition**: 최적 격자 자동 탐색, 로드 균형
- **Multi-Stream Overlap**: MPI 통신 시간을 GPU 계산으로 완전 은닉
- **CUDA-aware MPI + NCCL**: GPU간 직접 통신, CPU 개입 없음
- **비동기 MPI Reduce**: 통신 대기 시간 없이 배치 처리

### 결론

> DeepScaleMD의 모든 GPU 최적화는 독립적으로 작동하면서도 ML Potential 실행 파이프라인과 직접 연결됩니다.  
> 전통 MD와 MLP-MD 모두에서 **대규모 노드 스케일링 + 최고 성능**을 단일 코드베이스로 달성합니다.

---

*이 문서는 DeepScaleMD 소스코드 (`src/`, `include/`) 직접 분석을 기반으로 작성되었습니다.*
ENDOFFILE