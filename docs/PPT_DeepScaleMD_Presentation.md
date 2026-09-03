# DeepScaleMD — PPT 발표 자료

---

## 발표 메타데이터

- **발표 제목**: DeepScaleMD: GPU-Optimized Molecular Dynamics for Large-Scale Machine Learning Potential Simulations
- **발표 대상**: 슈퍼컴퓨팅 연구자, 계산화학/물리 연구자, HPC 엔지니어
- **발표 시간**: 20~30분 (슬라이드 13장)
- **핵심 메시지**:
  - 기존 MD 코드(GROMACS/LAMMPS)의 GPU 활용 한계를 구조적으로 해결
  - ML Potential(MLP) 실행에 최적화된 파이프라인 내장
  - 수천 노드 MPI 스케일링 + 최고 GPU 효율을 단일 코드베이스로 달성

---

## 슬라이드 1 — 표지

**제목**: DeepScaleMD  
**부제**: GPU-Optimized Molecular Dynamics  
**하단 태그**: C++ · CUDA · MPI · ML Potential Ready

**화자 노트**:
- 기존 MD 코드의 GPU 비효율과 ML Potential 통합 문제를 동시에 해결하기 위해 처음부터 재설계한 분자동역학 시뮬레이션 엔진

---

## 슬라이드 2 — 왜 새로 만들었나? 기존 코드의 한계

**제목**: GROMACS / LAMMPS로는 부족한 이유

**기존 코드의 5대 문제점**

| 문제 영역 | 기존 GROMACS / LAMMPS | DeepScaleMD |
|-----------|----------------------|-------------|
| GPU ↔ CPU 데이터 이동 | 매 스텝 cudaMemcpy → 수십 μs 낭비 | GPU-Resident: 데이터 GPU 상주 |
| 커널 런칭 오버헤드 | CPU가 매 스텝 드라이버에 명령 전달 | CUDA Graphs로 오버헤드 제로화 |
| MPI 통신 블로킹 | Halo Exchange 동안 GPU 계산 정지 | Multi-Stream으로 통신·계산 오버랩 |
| ML Potential 통합 | 별도 플러그인, Python 브릿지 (느림) | 네이티브 MLP 파이프라인 내장 |
| 메모리 레이아웃 | AoS 구조 → GPU 캐시 미스 폭발 | SoA 구조 → 완전 coalesced 접근 |

**설계 철학**: GPU-first, MPI-scalable, MLP-ready

> **[MPI 통신 블로킹 보충 설명]**  
> 멀티 GPU 환경에서 각 GPU는 자신의 서브도메인 경계에 있는 파티클 정보를 이웃 GPU와 교환해야 함 (Halo Exchange).  
> 기존 코드는 이 통신이 완료될 때까지 GPU 계산 전체가 멈춤 → 통신 시간이 그대로 낭비.  
> DeepScaleMD는 통신과 계산을 3개의 독립 CUDA 스트림으로 동시에 진행하여 통신 대기 시간을 0으로 만듦.

**화자 노트**:
- GROMACS·LAMMPS는 CPU 시대 설계를 GPU에 맞게 확장한 구조 → 근본적 한계 존재
- DeepScaleMD는 플러그인 추가가 아닌 코드 재설계로 해결

---

## 슬라이드 3 — GPU 최적화 ① SoA 메모리 레이아웃

**제목**: 메모리 접근 패턴이 성능을 결정한다

**문제: AoS (Array of Structs) — 기존 방식**
- 파티클 데이터를 좌표·속도·힘을 한 구조체로 묶어 배열로 저장
- GPU 스레드 32개가 각자의 x좌표를 읽으면 메모리 32곳을 산발적으로 접근
- 결과: 캐시 미스 폭발, 메모리 대역폭 낭비

**해법: SoA (Structure of Arrays) — DeepScaleMD 방식**
- x 좌표끼리, y 좌표끼리, z 좌표끼리 별도 배열로 분리 저장
- GPU 스레드 32개가 x 배열의 연속 32개 원소를 동시에 읽음
- 단 1번의 메모리 트랜잭션(128바이트 캐시 라인)으로 처리 → 완전 coalesced 접근

**효과**
- GPU L1/L2 캐시 적중률 최대화
- 메모리 대역폭 이론적 최대치에 근접
- 힘 계산·적분 커널 전체 처리량 직접 향상

**화자 노트**:
- 가장 기본이면서 가장 중요한 최적화
- 모든 CUDA 커널이 SoA 레이아웃을 전제로 설계 → 이후 모든 최적화의 토대

---

## 슬라이드 4 — GPU 최적화 ② Morton Z-Order + GPU 이웃 리스트

**제목**: 공간 가까운 원자는 메모리에서도 가까워야 한다

**문제**
- 시뮬레이션 중 원자 계속 이동 → 공간 인접 원자가 메모리에서 멀리 흩어짐
- 이웃 탐색 시 분산된 메모리를 무작위로 읽음 → 캐시 미스 연쇄 발생

**해법: Morton Z-Order 코드 (공간 충전 곡선)**
- 3D 격자 좌표를 비트 인터리빙하여 64비트 Morton 코드 계산
- GPU에서 Morton 코드 기준으로 파티클 배열 정렬
- 공간 인접 원자 = 메모리 인접 원자 → 이웃 탐색 캐시 효율 극대화
- cell_start / cell_end 배열로 O(1) 셀 조회 가능

**Warp-per-Particle 전략**
- GPU Warp(32스레드)가 원자 1개를 협력 처리
- 27개 이웃 셀(3×3×3)을 32레인이 분담하여 병렬 탐색
- 이웃 탐색 속도 최대 32배 향상

**GPU-side 변위 판정 (이웃 리스트 재빌드 시점 결정)**
- 매 스텝마다 이웃 리스트를 다시 만들면 비용이 큼 → 원자가 충분히 이동했을 때만 재빌드해야 함
- 기존 방식: 원자 최대 변위를 CPU로 내려받아 판단 → 불필요한 GPU↔CPU 복사 발생
- DeepScaleMD: GPU 내부에서 Thrust 리덕션으로 전체 원자 최대 변위를 직접 계산
- Verlet skin(안전 여유 거리)의 절반을 초과한 경우에만 재빌드 → 불필요한 rebuild 최소화
- CPU 개입 없이 GPU 단독으로 판단·실행 → 지연 없음

**화자 노트**:
- Morton Z-Order는 특히 ML Potential에 유리
- GNN 기반 MLP는 cutoff 6~8Å로 크기 때문에 이웃 원자 수가 많음 → 캐시 효율 향상 효과 더 큼

---

## 슬라이드 5 — GPU 최적화 ③ Kernel Fusion (커널 융합)

**제목**: 같은 데이터를 두 번 읽지 않는다

**문제: Velocity Verlet의 중복 메모리 접근**
- Step 1(Half-kick): 위치·속도를 글로벌 메모리에서 읽고 속도 업데이트 후 씀
- Step 2(Full-kick): 같은 위치·속도를 글로벌 메모리에서 다시 읽고 위치 업데이트 후 씀
- 같은 데이터를 2번 읽음 → 글로벌 메모리 접근: 수백 사이클/번

**해법: Fused Integrator 커널**
- 위치·속도·힘 데이터를 레지스터에 단 1회 로드
- 레지스터 내에서 가속도 계산 → 속도 업데이트 → 위치 업데이트 모두 처리
- 글로벌 메모리에 1회만 기록

| 항목 | 기존 | Kernel Fusion |
|------|------|---------------|
| 글로벌 메모리 읽기 | 2회 | 1회 |
| 글로벌 메모리 쓰기 | 2회 | 1회 |

**화자 노트**:
- GPU 메모리 대역폭은 연산 속도보다 훨씬 먼저 포화됨
- 커널 융합은 "memory-bound" 병목을 직접 공격하는 최적화

---

## 슬라이드 6 — GPU 최적화 ④ GPU-Resident 실행 + CUDA Graphs

**제목**: CPU를 거치지 않는 시뮬레이션 루프

**GPU-Resident 실행**
- 위치·속도·힘 배열이 시뮬레이션 전 기간 GPU 메모리에 상주
- Thermostat·Barostat 연산도 GPU에서 직접 실행 (apply_device API)
- CPU로의 복사는 trajectory 파일 출력 시에만 발생

**기존 방식과의 차이**
- 기존: GPU 계산 → CPU 복사 → Thermostat/Barostat 처리 → GPU 복사 반복
- DeepScaleMD: GPU 메모리 내에서 모든 연산 완결, CPU 개입 없음

**CUDA Graphs: CPU 런칭 오버헤드 제거**
- 기존: CPU가 매 스텝마다 커널 발사 명령을 드라이버에 전달 → 수 μs × 커널 수 × 스텝 수 누적
- DeepScaleMD: 워밍업 스텝에서 전체 루프를 그래프로 1회 캡처 → 이후 매 스텝 cudaGraphLaunch 1회 호출로 처리
- CUDA 드라이버가 커널 간 의존성을 분석하여 독립 커널 자동 병렬 실행
- 100스텝마다 재캡처 (시스템 변화 대응)

> **[CUDA Graphs와 ML Potential 보충 설명]**  
> MD 시뮬레이션은 매 스텝 동일한 커널 시퀀스를 반복함.  
> ML Potential 사용 시: 전처리(이웃 리스트 → Graph 변환) → GNN inference → 후처리(힘 배분) 순서가 매 스텝 고정됨.  
> CUDA Graphs가 이 반복 구조를 한 번 캡처해두면, 매 스텝 CPU가 커널을 하나씩 발사하는 오버헤드(수 μs × 커널 수)가 완전히 사라짐.  
> ML 모델은 전통 힘장보다 커널 수가 많기 때문에 효과가 더 큼.

**화자 노트**:
- CUDA Graphs는 특히 ML Potential 사용 시 효과 큼
- 전처리 → inference → 후처리로 이어지는 커널 시퀀스가 매 스텝 동일하게 반복되기 때문

---

## 슬라이드 7 — GPU 최적화 ⑤ 혼합 정밀도 + 캐시 최적화

**제목**: 빠르고 정확하게 — 정밀도의 전략적 배분

**혼합 정밀도 (Mixed Precision) 전략**

| 데이터 | 정밀도 | 이유 |
|--------|--------|------|
| 위치 (x, y, z) | FP32 | 캐시 라인에 2배 더 많은 입자 적재, ALU 처리량 2~32배 |
| 힘 누산, 에너지 합산 | FP64 | 수치 오차 누적 방지, 에너지 보존 보장 |
| ML inference 입출력 | FP32 | PyTorch Tensor Core 활용, 타입 변환 오버헤드 제거 |

**추가 캐시 최적화 기법**

- **\_\_ldg() Read-Only Cache**
  - GPU 글로벌 메모리 접근에는 L1 캐시와 별도로 읽기 전용(Read-Only) 캐시가 존재함
  - \_\_ldg() 명령어로 데이터를 읽으면 이 캐시를 경유하여 같은 데이터를 반복 읽을 때 L2 접근을 건너뜀
  - 이웃 리스트·파티클 타입처럼 매 스텝 변하지 않고 반복 읽히는 데이터에 적용 → L2 접근 횟수 감소, 캐시 효율 향상

- **\_\_shfl\_down\_sync Warp Reduction**
  - GPU에서 에너지나 압력(비리얼)을 계산할 때 전체 원자의 값을 모두 합산해야 함
  - 기존 방식: 각 스레드가 atomicAdd로 전역 변수에 하나씩 더함 → 충돌(contention) 발생, 직렬화됨
  - Warp Reduction: Warp 내 32개 스레드가 셔플 명령어로 옆 레인의 값을 직접 레지스터에서 읽어 절반씩 합산 반복
  - atomicAdd 호출 횟수가 32개 → 1개로 줄어 충돌 32분의 1 수준으로 감소 → 합산 속도 대폭 향상

**화자 노트**:
- Mixed Precision은 MLP-MD에서 특히 중요
- MACE·NequIP 등 GNN 기반 모델은 기본적으로 FP32 inference → 타입 변환 없이 직접 전달 가능

---

## 슬라이드 8 — MPI 대규모 스케일링

**제목**: 수천 GPU에서도 효율을 잃지 않는 통신 설계

**3D Domain Decomposition**
- 전체 박스를 Nx × Ny × Nz 격자로 자동 분할
- 각 MPI 랭크가 독립 서브도메인만 계산
- 최적 분할 비율 자동 탐색 (정육면체에 가장 가까운 비율)
- MPI Cartesian Communicator로 26방향 이웃·주기 경계 통신 관리

**Multi-Stream 통신-계산 오버랩 (핵심)**

> **[Halo Exchange 보충 설명]**  
> 각 GPU는 자신의 서브도메인 경계 근처에 있는 원자 정보를 이웃 GPU와 주고받아야 함 → 이를 Halo Exchange라 함.  
> 기존: Halo Exchange가 끝날 때까지 GPU 계산 전체 정지 → 통신 시간이 순수 낭비.  
> DeepScaleMD: 원자를 경계(Boundary)와 내부(Interior)로 나누어, 경계 데이터 통신이 진행되는 동안 내부 원자 힘 계산을 동시에 수행.  
> 통신과 계산이 겹쳐서 실행되므로 통신 대기 시간이 GPU 계산 시간 속으로 숨겨짐.

- 파티클을 내부(Interior)와 경계(Boundary)로 분류 후 3개 CUDA 스트림에 배분
  - 통신 스트림: MPI Halo Exchange 비동기 실행
  - 내부 스트림: 통신과 동시에 내부 원자 힘 계산
  - 경계 스트림: 통신 완료 이벤트 수신 후 경계 힘 계산
- 결과: MPI 통신 시간이 GPU 계산에 완전히 숨겨짐

**CUDA-aware MPI + NCCL**
- GPU 버퍼를 CPU로 복사하지 않고 직접 MPI 전달
- NCCL AllReduce: GPU 간 링(ring) 토폴로지로 집합 통신 → CPU 개입 없음
- 싱글 랭크 환경: MPI 호출 자체를 생략하는 Fast Path 존재 → 단일 노드 오버헤드 없음

**OpenMP 자동 튜닝**
- 랭크당 스레드 수 = 전체 코어 수 ÷ MPI 랭크 수 (자동 계산)
- OMP_NUM_THREADS 명시 시 우선 적용 → 오버서브스크립션 방지

**화자 노트**:
- Multi-Stream 오버랩은 MLP에서 더욱 효과적
- MLP cutoff가 크기 때문에 Halo 통신량이 전통 LJ보다 훨씬 많음 → 통신이 길수록 오버랩 이점 증가

---

## 슬라이드 9 — ML Potential 가속화: GPU 최적화의 시너지

**제목**: MLP-MD의 모든 병목을 GPU 최적화로 직접 해결

**MLP-MD 실행 파이프라인 (매 MD 스텝)**
- ① 원자 좌표: GPU에 이미 상주 (GPU-Resident) → H2D 복사 없음
- ② 이웃 리스트 생성: Morton Z-Order + Warp-per-Particle 병렬 탐색
- ③ Graph 구성: CUDA Graphs로 ML 전처리 커널 런칭 오버헤드 흡수 / Halo 통신과 동시 진행 (Multi-Stream)
- ④ GNN Inference: FP32 Tensor Core 활용 (Mixed Precision)
- ⑤ 힘 배분: NCCL GPU AllReduce → CPU 개입 없음
- ⑥ 적분: Kernel Fusion으로 메모리 접근 최소화

**MLP 특성별 대응 최적화**

| MLP 고유 특성 | 발생하는 병목 | DeepScaleMD 대응 | 효과 |
|--------------|-------------|-----------------|------|
| GNN cutoff 크다 (6~8 Å) | Halo 통신량 증가 | Multi-Stream 오버랩 | 통신 대기 시간 → 0 |
| 전체 원자 힘 합산 필요 | 멀티 GPU AllReduce | NCCL GPU-direct | CPU 왕복 제거 |
| FP32 inference 기본 | 타입 변환 오버헤드 | Mixed Precision (FP32) | Tensor Core 활용 |
| 매 스텝 동일 커널 시퀀스 | CPU 런칭 오버헤드 누적 | CUDA Graphs | 오버헤드 제로 |
| Graph 구성 = 이웃 탐색 | 캐시 미스 | Morton Z-Order + SoA | 이웃 탐색 캐시 효율 극대화 |

**화자 노트**:
- LAMMPS에서 MACE 실행 시: Python 브릿지를 통해 매 스텝 CPU↔GPU 왕복 발생
- DeepScaleMD: 이 왕복 경로 자체가 없음 → 모든 연산이 GPU에서 완결

---

## 슬라이드 10 — LAMMPS 스타일 성능 분석 내장

**제목**: 병목을 수치로 보여주는 내장 프로파일러

**자동 출력 타이밍 분석 항목**
- Section별 분류: Pair(힘 계산), Neigh(이웃 리스트), Comm(MPI 통신), Output, Modify
- min / avg / max 시간: 모든 MPI 랭크에서 수집 → 로드 불균형 즉시 파악
- %varavg: 랭크 간 시간 편차 (클수록 특정 랭크가 병목)
- %total: 전체 시간 대비 각 구간의 비율
- LAMMPS와 동일한 출력 형식 → 기존 사용자 학습 비용 없음

**활성화 방법**
- 환경변수 MD_TIMING=1 설정 후 실행

**화자 노트**:
- MLP 사용 시 Pair(= ML inference) 비율과 Comm(= Halo Exchange) 비율을 정량적으로 확인 가능
- 최적화 방향 결정에 직접 활용 가능

---

## 슬라이드 11 — 지원 시뮬레이션 기법 전체 목록

**제목**: 단일 코드로 전통 MD부터 MLP-MD까지

**적분기 (Integrators)**
- Verlet, Velocity-Verlet, Leapfrog, r-RESPA (다중 시간 간격)

**온도 조절 (Thermostats)**
- Nosé-Hoover, Langevin, Andersen

**압력 조절 (Barostats)**
- Parrinello-Rahman, Berendsen (Andersen)

**힘장 (Force Fields)**
- Lennard-Jones, Electrostatic (PME 장거리 정전기 지원)
- ML Potential: PyTorch / TensorFlow 기반 모델 (MACE 등)
- 결합 항목: Bonds, Angles, Dihedrals (Python 인터페이스)

**하드웨어 지원**

| 빌드 타입 | 대상 |
|----------|------|
| CPU-only | 모든 클러스터 |
| CUDA | NVIDIA GPU |
| SYCL | 멀티벤더 GPU (Intel, AMD) |

**화자 노트**:
- r-RESPA: ML Potential과 전통 힘장을 서로 다른 시간 간격으로 실행 가능
- MLP의 높은 계산 비용을 줄이면서 정확도 유지

---

## 슬라이드 12 — 최적화 전체 지도 (요약)

**제목**: 모든 최적화가 MLP-MD로 연결된다

**메모리 최적화**

| 기법 | 효과 |
|------|------|
| SoA 레이아웃 | GPU 메모리 접근 완전 coalesced |
| Morton Z-Order 정렬 | 공간 locality → 이웃 탐색 캐시 효율 극대화 |
| FP32/FP64 혼합 정밀도 | 캐시 용량 2배 + ALU 처리량 2~32배 |
| __ldg() Read-Only Cache | 반복 읽기 데이터 L1 캐시 고정 |

**계산 최적화**

| 기법 | 효과 |
|------|------|
| Kernel Fusion | 글로벌 메모리 접근 횟수 절반 |
| CUDA Graphs | CPU 드라이버 런칭 오버헤드 제로 |
| GPU-Resident 실행 | H2D/D2H 복사 최소화 |
| Warp-per-Particle | 이웃 탐색 32배 병렬화 |

**통신 최적화**

| 기법 | 효과 |
|------|------|
| 3D Domain Decomposition | 최적 격자 자동 탐색, 로드 균형 |
| Multi-Stream Overlap | MPI 통신 시간 GPU 계산으로 완전 은닉 |
| CUDA-aware MPI + NCCL | GPU 간 직접 통신, CPU 개입 없음 |
| 비동기 MPI Reduce | 통신 대기 없이 배치 처리 |

**화자 노트**:
- 각 최적화는 독립적으로 동작하면서도 MLP 파이프라인의 각 단계와 1:1 대응
- 전통 MD에서 좋은 코드는 MLP-MD에서도 좋음

---

## 슬라이드 13 — 결론 및 향후 계획

**제목**: DeepScaleMD: 다음 세대 슈퍼컴퓨터 MD 엔진

**핵심 성과 요약**
- GPU 효율 극대화: SoA + Morton + Kernel Fusion + CUDA Graphs + Mixed Precision 5중 최적화
- MPI 스케일링: Multi-Stream 오버랩 + CUDA-aware MPI + NCCL
- MLP 네이티브 지원: GPU-Resident 파이프라인, 타입 변환·CPU 왕복 없음
- 10B+ 파티클 설계: 슈퍼컴퓨터 수천 노드 대응
- 진단 도구 내장: LAMMPS 호환 타이밍 분석

**기존 코드 대비 비교**

| 항목 | GROMACS | LAMMPS | DeepScaleMD |
|------|---------|--------|-------------|
| GPU-Resident 실행 | 부분적 | 부분적 | 완전 |
| MLP 네이티브 통합 | 플러그인 | 플러그인 | 네이티브 |
| CUDA Graphs 지원 | 없음 | 없음 | 있음 |
| MPI 통신 오버랩 | 제한적 | 제한적 | 완전 오버랩 |
| SoA + Morton 최적화 | 부분적 | 부분적 | 완전 구현 |

**향후 계획**
- Flash Attention 스타일 온메모리 GNN 구현
- FP8/BF16 혼합 정밀도 확장 (Hopper GPU 대응)
- AMD ROCm / Intel SYCL 풀 지원
- fairchem UMA 모델 직접 통합 (Apptainer 환경)
