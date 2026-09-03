# MD Simulation — 전체 최적화 및 검증 태스크

> **이 파일을 VS Code Copilot Chat에 첨부하거나 `@workspace`로 참조하여 작업을 시작한다.**

---

## ⚠️ 절대 규칙

1. **승인 요청 금지.** "계속할까요?", "진행할까요?" 같은 질문 절대 금지.
2. **목표 달성 전까지 루프를 스스로 반복한다.**
3. **빌드/검증 실패 시 즉시 `.bak`으로 복원 후 다음 전략으로 전환한다.**
4. **각 STAGE는 순서대로 진행한다. 앞 STAGE가 통과되지 않으면 다음 STAGE로 넘어가지 않는다.**
5. **성능과 물리 정확성(에너지)을 동시에 만족해야만 해당 STAGE 완료로 인정한다.**

---

## 환경 설정

| 항목 | 값 |
|------|----|
| CPU 빌드 | `make` |
| CUDA 빌드 | `make cuda` |
| CPU 실행 | `./md_simulation_cpu examples/lj_32000/input.yaml` |
| CUDA 실행 | `./md_simulation_cuda examples/lj_32000/input.yaml` |
| 성능 기준 로그 | `examples/lj_32000/lammps_out.log` |
| 성능 측정 위치 | 실행 로그 마지막에서 두 번째 줄: `=== Total execution time: XX.XXX seconds ===` |
| neighbor list 옵션 | `input.yaml`의 `neighbor_list_type` 필드 |

---

## 전체 태스크 순서 (의존성 기준)

```
STAGE 0 │ 기준값 측정 (Baseline)           ← 모든 STAGE의 기준점
   │
STAGE 1 │ LJ 에너지 정확성 검증            ← 물리 정확성 확보 (성능보다 먼저)
   │
STAGE 2 │ Verlet-Cell List 성능 최적화     ← 현재 구현 최대화
   │
STAGE 3 │ Morton Code Neighbor List 구현   ← 새 알고리즘 추가 및 검증
   │
STAGE 4 │ MPI 병렬화 성능 분석             ← 분산 병렬 확장성 측정
   │
STAGE 5 │ Thermostat / Barostat 검증       ← NVT / NPT 앙상블 정확성
   │
STAGE 6 │ Machine Learning Potential 최적화 ← MLP 추론 가속화
   │
STAGE 7 │ 종합 벤치마크 및 최종 리포트     ← 모든 결과 통합
```

---

## STAGE 0: 기준값 측정 (Baseline) — 최초 1회

### 목적
모든 이후 STAGE에서 비교 기준이 되는 LAMMPS 참조값과 현재 코드의 초기 상태를 기록한다.

### 실행 순서

```bash
# 1. LAMMPS 기준 로그 확인
cat examples/lj_32000/lammps_out.log

# 2. CPU baseline
make 2>&1 | tail -5
./md_simulation_cpu examples/lj_32000/input.yaml 2>&1 | tee /tmp/cpu_baseline.txt

# 3. CUDA baseline
make cuda 2>&1 | tail -5
./md_simulation_cuda examples/lj_32000/input.yaml 2>&1 | tee /tmp/cuda_baseline.txt
```

### 추출 및 기록 항목

LAMMPS 로그 thermo 출력의 **마지막 Step 행**에서 추출한다.
헤더(`Step Temp E_pair TotEng KinEng ...`)를 먼저 확인하고 컬럼을 매핑한다.

```
[STAGE 0 — BASELINE RECORD]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAMMPS 기준
  REF_TIME          = _____ s
  REF_TOTAL_ENERGY  = _____  (eV 또는 kcal/mol, 단위 확인)
  REF_POT_ENERGY    = _____
  REF_KIN_ENERGY    = _____
  REF_TEMPERATURE   = _____  K
  REF_PRESSURE      = _____  (NVT/NPT인 경우)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CPU 현재
  CPU_TIME          = _____ s  (LAMMPS 대비 _____x)
  CPU_TOTAL_ENERGY  = _____    (오차 _____%)
  CPU_POT_ENERGY    = _____    (오차 _____%)
  CPU_KIN_ENERGY    = _____    (오차 _____%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CUDA 현재
  CUDA_TIME         = _____ s  (LAMMPS 대비 _____x)
  CUDA_TOTAL_ENERGY = _____    (오차 _____%)
  CUDA_POT_ENERGY   = _____    (오차 _____%)
  CUDA_KIN_ENERGY   = _____    (오차 _____%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 완료 조건
위 항목이 모두 기록되면 STAGE 0 완료 → 즉시 STAGE 1 시작.

---

## STAGE 1: LJ 에너지 정확성 검증

### 목적
이후 모든 최적화의 전제 조건인 **물리 정확성**을 먼저 확보한다.
성능 최적화는 이 단계를 통과한 이후에만 의미가 있다.

### 검증 기준

```
에너지 오차 = |현재값 - LAMMPS기준값| / |LAMMPS기준값| × 100 (%)

✅ PASS (모두 만족):
  Total Energy 오차    ≤ 1.0%
  Potential Energy 오차 ≤ 1.0%
  Kinetic Energy 오차   ≤ 2.0%   ← 속도 초기화 방식 차이 허용

❌ FAIL (하나라도 해당 시 코드 수정 후 재검증):
  허용 오차 초과
  NaN 또는 Inf 출력
  에너지 단조 발산 (드리프트)
```

### 검증 스크립트 (`scripts/validate_energy.sh`)

이 스크립트를 프로젝트에 생성하고, 이후 모든 STAGE에서 재사용한다.
`REF_*` 값은 STAGE 0에서 파싱한 값으로 채운다.

```bash
#!/bin/bash
# scripts/validate_energy.sh
# 사용법: ./scripts/validate_energy.sh cpu|cuda [input.yaml 경로]

MODE=${1:-cpu}
INPUT=${2:-examples/lj_32000/input.yaml}

REF_TOT=__FILL_FROM_STAGE0__
REF_POT=__FILL_FROM_STAGE0__
REF_KIN=__FILL_FROM_STAGE0__
REF_TIME=__FILL_FROM_STAGE0__

BIN="./md_simulation_${MODE}"
echo "=== VALIDATION: ${MODE} | ${INPUT} ==="

run_out=$($BIN $INPUT 2>&1)
echo "$run_out" | tail -20

# 실제 출력 포맷에 맞게 grep 패턴 수정
CUR_TOT=$(echo  "$run_out" | grep -i "total.energy"      | tail -1 | awk '{print $NF}')
CUR_POT=$(echo  "$run_out" | grep -i "potential"         | tail -1 | awk '{print $NF}')
CUR_KIN=$(echo  "$run_out" | grep -i "kinetic"           | tail -1 | awk '{print $NF}')
CUR_TIME=$(echo "$run_out" | grep  "Total execution time" | awk -F': ' '{print $2}' | awk '{print $1}')

python3 - <<PYEOF
import sys, math

ref_tot, ref_pot, ref_kin, ref_time = $REF_TOT, $REF_POT, $REF_KIN, $REF_TIME

def sf(s):
    try: return float(s)
    except: return float('nan')

cur_tot  = sf("$CUR_TOT")
cur_pot  = sf("$CUR_POT")
cur_kin  = sf("$CUR_KIN")
cur_time = sf("$CUR_TIME")

def err(c, r): return abs(c-r)/abs(r)*100 if r!=0 else float('inf')

e_tot = err(cur_tot, ref_tot)
e_pot = err(cur_pot, ref_pot)
e_kin = err(cur_kin, ref_kin)
ratio = cur_time / ref_time if ref_time > 0 else float('inf')

W=55
print(f"\n{'─'*W}")
print(f"  {'항목':<22} {'현재':>10} {'기준':>10} {'오차':>7}  판정")
print(f"{'─'*W}")
print(f"  {'Total  Energy':<22} {cur_tot:>10.4f} {ref_tot:>10.4f} {e_tot:>6.2f}%  {'✅' if e_tot<=1.0 else '❌'}")
print(f"  {'Potential Energy':<22} {cur_pot:>10.4f} {ref_pot:>10.4f} {e_pot:>6.2f}%  {'✅' if e_pot<=1.0 else '❌'}")
print(f"  {'Kinetic  Energy':<22} {cur_kin:>10.4f} {ref_kin:>10.4f} {e_kin:>6.2f}%  {'✅' if e_kin<=2.0 else '❌'}")
print(f"  {'실행 시간':<22} {cur_time:>9.3f}s {ref_time:>9.3f}s {ratio:>6.2f}x  {'✅' if ratio<=1.2 else ('🔶' if ratio<=2.0 else '🔴')}")
print(f"{'─'*W}")

nan_ok     = not any(math.isnan(v) for v in [cur_tot,cur_pot,cur_kin])
energy_ok  = e_tot<=1.0 and e_pot<=1.0 and e_kin<=2.0
perf_ok    = ratio<=1.2

if not nan_ok:
    print("  → ❌ NaN/Inf 감지"); sys.exit(2)
elif not energy_ok:
    print("  → ❌ ENERGY FAIL"); sys.exit(1)
elif not perf_ok:
    print("  → 🔶 ENERGY PASS / PERF NOT YET"); sys.exit(3)
else:
    print("  → ✅ ALL PASS"); sys.exit(0)
PYEOF
```

### 에너지 오차가 있을 경우 체크리스트

오차가 허용 범위를 초과하면 아래 순서로 원인을 찾아 수정한다.

1. cutoff 거리 설정 확인 (LAMMPS `pair_coeff` 와 비교)
2. LJ 파라미터 (ε, σ) 단위 및 값 확인
3. Periodic boundary condition 적용 방식 확인
4. Long-range correction 포함 여부 확인
5. neighbor list skin distance 값 확인

### 완료 조건
`validate_energy.sh cpu`와 `validate_energy.sh cuda` 모두 exit 0 또는 exit 3 (에너지 OK) → STAGE 2 시작.

---

## STAGE 2: Verlet-Cell List 성능 최적화

### 목적
현재 구현(`neighbor_list_type: "verlet_cell_list"`)의 성능을 LAMMPS 수준으로 끌어올린다.
**에너지 검증을 매 반복 수행하며, 실패 시 즉시 복원한다.**

### 자동 반복 루프

```
LOOP:
  1. 소스코드 분석 → 병목 식별
  2. 파일 백업 (*.bak)
  3. 최적화 전략 하나 적용
  4. make (실패 시 복원 → 다음 전략)
  5. validate_energy.sh 실행
     - exit 1/2: 복원 → 다음 전략
     - exit 3:   에너지 OK, 성능 미달 → 이력 기록 → LOOP
     - exit 0:   완료
```

### CPU 최적화 전략 (우선순위 순)

| # | 전략 | 예상 효과 | input.yaml 옵션 |
|---|------|-----------|----------------|
| 1 | 컴파일러 플래그: `-O3 -march=native -ffast-math -funroll-loops -fopenmp` | 20~50% | — |
| 2 | OpenMP: force 루프 `#pragma omp parallel for reduction` | 코어 수 비례 | — |
| 3 | Cell list 셀 크기 튜닝 (cutoff 정확히 맞추기) | 10~30% | — |
| 4 | Newton 3rd law (force 계산량 절반) | 40~50% | — |
| 5 | SoA 데이터 레이아웃 + `alignas(64)` | 10~30% | — |
| 6 | `#pragma omp simd` 또는 AVX2 intrinsics | 2~4x | — |
| 7 | Verlet skin distance 최적화 | 5~20% | — |

### CUDA 최적화 전략 (우선순위 순)

| # | 전략 | 예상 효과 |
|---|------|-----------|
| 1 | Block size 튜닝 (128 / 256 테스트) | 10~30% |
| 2 | Coalesced memory access (SoA on GPU) | 20~50% |
| 3 | Shared memory tile-based neighbor 탐색 | 20~40% |
| 4 | `__ldg()` read-only cache hint | 5~15% |
| 5 | GPU-side cell list 구축 (CPU→GPU 전송 제거) | 30~60% |
| 6 | Warp divergence 최소화 | 10~20% |

### 이력 출력 형식 (매 반복, 승인 없이 즉시 다음 반복)

```
════════════════════════════════════════════════════
[STAGE 2 — ITERATION N]
전략      : (적용한 내용)
빌드      : SUCCESS / FAIL
에너지 검증:
  TotEng  : _____ → 오차 _____% ✅/❌
  PotEng  : _____ → 오차 _____% ✅/❌
  KinEng  : _____ → 오차 _____% ✅/❌
CPU  성능  : _____ s  (이전 대비 +_____%)  LAMMPS _____x ✅/🔶/🔴
CUDA 성능  : _____ s  (이전 대비 +_____%)  LAMMPS _____x ✅/🔶/🔴
상태      : DONE ✅ / CONTINUE 🔄
다음 전략  : _______________
════════════════════════════════════════════════════
→ 승인 없이 즉시 다음 반복 또는 STAGE 3 시작
```

### 완료 조건
```
✅ Energy 오차 모두 통과
✅ CPU 또는 CUDA 실행시간 ≤ LAMMPS × 1.2
```

---

## STAGE 3: Morton Code Neighbor List 구현 및 검증

### 목적
`neighbor_list_type: "morton_code"` 옵션으로 동작하는 Morton Code 기반 neighbor list를
**올바르게 구현하고 Verlet-Cell 대비 성능을 비교한다.**

### 구현 요구사항

#### 3-1. Morton Code (Z-curve) 인코딩

3D 좌표 → Morton code 변환을 구현한다.

```cpp
// 각 축 좌표를 정수 격자로 변환 후 비트 인터리빙
// 좌표 정규화: cell_idx = (pos - box_min) / cell_size
// 각 축 10비트 → 30비트 Morton code (10억 개 셀 대응)

inline uint32_t expand_bits(uint32_t v) {
    // 10비트 정수를 30비트로 확장 (비트 사이에 00 삽입)
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v <<  8)) & 0x0300F00F;
    v = (v | (v <<  4)) & 0x030C30C3;
    v = (v | (v <<  2)) & 0x09249249;
    return v;
}

inline uint32_t morton3D(uint32_t x, uint32_t y, uint32_t z) {
    return expand_bits(x) | (expand_bits(y) << 1) | (expand_bits(z) << 2);
}
```

#### 3-2. 구현 단계

```
Step 1: 모든 원자에 대해 Morton code 계산
Step 2: Morton code 기준으로 원자 정렬 (radix sort 또는 std::sort)
Step 3: 정렬된 원자 배열을 기반으로 neighbor list 구축
         - 공간 지역성(spatial locality) 덕에 캐시 효율 향상
Step 4: 정렬 후 원래 인덱스 매핑 테이블 유지
Step 5: force 계산은 기존 루프 재사용 (인덱스만 매핑)
```

#### 3-3. CUDA 구현 추가 요구사항

```
- thrust::sort_by_key 사용: Morton code(key) + atom index(value) 정렬
- 정렬 후 GPU 메모리에서 직접 neighbor list 구축
- CPU→GPU 데이터 전송은 초기화 시 1회만 수행
```

### 검증 순서

```bash
# 1. input.yaml에서 morton_code 활성화
#    neighbor_list_type: "morton_code"  ← 주석 해제
#    neighbor_list_type: "verlet_cell_list"  ← 주석 처리

# 2. 빌드 및 에너지 검증
make && ./scripts/validate_energy.sh cpu
make cuda && ./scripts/validate_energy.sh cuda

# 3. Verlet-Cell 대비 성능 비교
echo "=== verlet_cell_list ===" 
# input.yaml verlet_cell_list 활성화 후 실행
./md_simulation_cpu examples/lj_32000/input.yaml | grep "execution time"

echo "=== morton_code ==="
# input.yaml morton_code 활성화 후 실행
./md_simulation_cpu examples/lj_32000/input.yaml | grep "execution time"
```

### 결과 기록 형식

```
[STAGE 3 — MORTON CODE 결과]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
에너지 검증 (morton_code)
  TotEng 오차: _____%  ✅/❌
  PotEng 오차: _____%  ✅/❌
  KinEng 오차: _____%  ✅/❌
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
성능 비교 (CPU)
  verlet_cell_list : _____ s
  morton_code      : _____ s  (_____ % 차이)
  LAMMPS 기준      : _____ s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
성능 비교 (CUDA)
  verlet_cell_list : _____ s
  morton_code      : _____ s  (_____ % 차이)
  LAMMPS 기준      : _____ s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
결론: morton_code가 더 빠른 조건: (원자 수 / 시스템 크기 등)
```

### 완료 조건
```
✅ morton_code 에너지 검증 통과 (STAGE 1과 동일 기준)
✅ 두 방식 모두 정상 동작 확인
✅ 성능 비교 결과 기록 완료
```

---

## STAGE 4: MPI 병렬화 성능 분석

### 목적
MPI 프로세스 수에 따른 **확장성(scaling)** 을 측정하고 병목을 분석한다.

### 전제 조건
- MPI 빌드가 이미 지원되거나 Makefile에 MPI 빌드 타겟이 있어야 함
- 없으면 먼저 `make mpi` 또는 `mpicxx` 기반 빌드 추가

### 측정 시나리오

```bash
# Strong scaling: 같은 문제 크기, 프로세스 수 증가
for NP in 1 2 4 8 16; do
    echo "=== NP=$NP ==="
    mpirun -np $NP ./md_simulation_mpi examples/lj_32000/input.yaml 2>&1 \
        | grep "execution time"
done

# Weak scaling: 프로세스당 원자 수 일정 (시스템 크기 증가)
# examples/lj_*000/input.yaml 파일이 있으면 각각 실행
```

### 분석 항목

| 항목 | 설명 |
|------|------|
| Strong scaling efficiency | `T(1) / (N × T(N))` × 100% |
| Communication overhead | force 계산 시간 중 halo exchange 비율 |
| Load imbalance | 프로세스 간 원자 수 분포 |
| 병목 위치 | `MPI_Barrier`, `MPI_Allreduce` 호출 빈도 및 대기시간 |

### 결과 기록 형식

```
[STAGE 4 — MPI SCALING 결과]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Strong Scaling (lj_32000)
  NP=1  : _____ s  (efficiency 100%)
  NP=2  : _____ s  (efficiency ____%)
  NP=4  : _____ s  (efficiency ____%)
  NP=8  : _____ s  (efficiency ____%)
  NP=16 : _____ s  (efficiency ____%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
병목 분석
  주요 병목: _______________
  개선 방향: _______________
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### MPI 성능이 기대 이하일 경우 체크리스트

1. 도메인 분해(domain decomposition) 방식 확인
2. Ghost atom 교환(halo exchange) 최소화
3. `MPI_Allreduce` 호출 빈도 최소화 (에너지 집계를 매 step마다 하지 않도록)
4. 비동기 통신(`MPI_Isend/Irecv`) 적용 여부 확인
5. 메모리 레이아웃의 통신 친화성 확인

### 완료 조건
```
✅ NP=1 대비 NP=4에서 scaling efficiency ≥ 70%
✅ 에너지 검증 통과 (MPI 버전도 동일 기준)
✅ 병목 분석 결과 기록 완료
```

---

## STAGE 5: Thermostat / Barostat 검증

### 목적
NVT → NPT 순서로 검증한다.
**NVT가 완전히 통과된 후에만 NPT를 시작한다.**
에러가 발생하면 소스 코드를 수정하고, 필요하면 중간 로그를 추가해서라도
두 앙상블이 정상 동작할 때까지 자동으로 반복한다.

### ⚠️ STAGE 5 절대 규칙
1. **NVT 통과 전에 NPT를 시작하지 않는다.**
2. **에러 발생 시 승인 없이 즉시 원인을 분석하고 코드를 수정한다.**
3. **원인 불명이면 중간 로그를 추가하여 상태를 파악한 후 수정한다.**
4. **수정 → 빌드 → 검증 루프를 목표 달성까지 스스로 반복한다.**

---

### 허용 오차 기준 (전 단계에서 공통 적용)

```
NVT 통과 조건:
  평균 온도  오차  ≤ 3.0%   (target_temperature 기준)
  에너지 드리프트  없음     (NaN/Inf 없음, 단조 증가/감소 없음)

NPT 통과 조건:
  평균 온도  오차  ≤ 3.0%   (target_temperature 기준)
  평균 압력  오차  ≤ 5.0%   (target_pressure 기준, 전체 trajectory 평균)
  박스 부피         발산 없음 (부피가 합리적 범위 내 수렴)
  에너지 드리프트  없음
```

---

### 5-1. NVT 테스트

#### 5-1-A. 입력 파일 준비

`examples/lj_32000/input_nvt.yaml`을 생성한다.
파일이 없으면 `examples/lj_32000/input.yaml`을 복사한 뒤 아래 항목을 수정한다.

```yaml
ensemble: "NVT"
thermostat: "nose_hoover"
target_temperature: 300.0    # LAMMPS fix nvt temp 값과 동일하게 맞출 것
thermostat_damping: 100.0    # LAMMPS Tdamp (fs 단위)
steps: 10000                 # 충분한 equilibration을 위해 최소 10000 step
thermo_output_interval: 100  # 온도/에너지를 100 step마다 출력 (파싱용)
```

#### 5-1-B. NVT 실행 및 로그 저장

```bash
make 2>&1 | tail -3
./md_simulation_cpu examples/lj_32000/input_nvt.yaml 2>&1 | tee /tmp/nvt_out.txt
echo "종료 코드: $?"
tail -30 /tmp/nvt_out.txt
```

#### 5-1-C. NVT 자동 검증 스크립트

아래 스크립트를 `scripts/validate_nvt.sh`로 저장하고 실행한다.

```bash
#!/bin/bash
# scripts/validate_nvt.sh
TARGET_TEMP=300.0   # input_nvt.yaml의 target_temperature 값
LOG=/tmp/nvt_out.txt

echo "=== [STAGE 5-1] NVT 검증 ==="

if [ ! -f "$LOG" ]; then
    echo "❌ 로그 파일 없음: $LOG"
    exit 1
fi

python3 - <<PYEOF
import sys, math, re

TARGET_TEMP = $TARGET_TEMP
log = open("$LOG").read()

# ── NaN / Inf 체크 ──────────────────────────────────────
if re.search(r'\bnan\b|\binf\b', log, re.IGNORECASE):
    print("❌ FAIL: NaN/Inf 감지")
    print("[DEBUG] 해당 줄:")
    for line in log.splitlines():
        if re.search(r'\bnan\b|\binf\b', line, re.IGNORECASE):
            print(f"  > {line}")
    sys.exit(2)

# ── 온도 값 파싱 ─────────────────────────────────────────
# 출력 포맷 예: "Step=100 Temp=299.5 ..." 또는 "100  299.5  ..."
# 아래 패턴은 일반적인 경우를 커버; 포맷이 다르면 자동 조정
temp_values = re.findall(
    r'[Tt]emp(?:erature)?\s*[=:]\s*([\d.+-]+)', log
)
if not temp_values:
    # 두 번째 시도: thermo 테이블 형식 (컬럼 순서 추정)
    temp_values = re.findall(
        r'^\s*\d+\s+([\d.+-]+)', log, re.MULTILINE
    )

if not temp_values:
    print("⚠ 온도 값을 파싱할 수 없음 — 로그 포맷을 확인하라")
    print("[DEBUG] 로그 마지막 20줄:")
    for line in log.splitlines()[-20:]:
        print(f"  | {line}")
    sys.exit(3)

temps = [float(t) for t in temp_values]
# equilibration 전반부 20% 제외 후 평균
skip = max(1, len(temps) // 5)
temps_eq = temps[skip:]
avg_temp = sum(temps_eq) / len(temps_eq)
err_temp = abs(avg_temp - TARGET_TEMP) / TARGET_TEMP * 100

# ── 에너지 드리프트 체크 ─────────────────────────────────
energy_values = re.findall(
    r'[Tt]ot(?:al)?\s*[Ee]nergy\s*[=:]\s*([\d.+eE-]+)', log
)
if not energy_values:
    energy_values = re.findall(
        r'^\s*\d+\s+[\d.+-]+\s+([\d.+eE-]+)', log, re.MULTILINE
    )

drift_ok = True
if len(energy_values) >= 10:
    energies = [float(e) for e in energy_values]
    eq_energies = energies[len(energies)//5:]
    e_min, e_max = min(eq_energies), max(eq_energies)
    e_range = e_max - e_min
    e_mean  = sum(eq_energies) / len(eq_energies)
    drift_ratio = e_range / abs(e_mean) * 100 if e_mean != 0 else float('inf')
    drift_ok = drift_ratio < 5.0   # 평균 대비 5% 이상 변동이면 드리프트로 판단
else:
    drift_ratio = 0.0

W = 55
print(f"\n{'─'*W}")
print(f"  {'항목':<22} {'측정값':>12} {'기준':>10} {'오차':>7}  판정")
print(f"{'─'*W}")
print(f"  {'평균 온도 (K)':<22} {avg_temp:>12.2f} {TARGET_TEMP:>10.1f} {err_temp:>6.2f}%  {'✅' if err_temp<=3.0 else '❌'}")
if len(energy_values) >= 10:
    print(f"  {'에너지 드리프트':<22} {'range=' + f'{e_range:.4f}':>12} {'< 5%':>10} {drift_ratio:>6.2f}%  {'✅' if drift_ok else '❌'}")
print(f"{'─'*W}")

temp_ok   = err_temp <= 3.0
all_ok    = temp_ok and drift_ok

if not temp_ok:
    print(f"  → ❌ 온도 오차 {err_temp:.2f}% > 3.0%")
    print(f"     [DEBUG] 온도 샘플 수: {len(temps_eq)}, 범위: {min(temps_eq):.2f}~{max(temps_eq):.2f} K")
if not drift_ok:
    print(f"  → ❌ 에너지 드리프트 {drift_ratio:.2f}% > 5.0%")

if all_ok:
    print("  → ✅ NVT PASS")
    sys.exit(0)
else:
    sys.exit(1)
PYEOF
```

```bash
chmod +x scripts/validate_nvt.sh
bash scripts/validate_nvt.sh
```

#### 5-1-D. NVT 실패 시 자동 디버깅 루프

검증이 실패하면 아래 순서로 원인을 찾아 **승인 없이 즉시** 수정하고 재실행한다.

```
NVT 검증 실패
  │
  ├─ [case A] NaN / Inf 발생
  │     원인: 시간 간격(dt) 너무 큼, 초기 원자 배치 충돌, 힘 계산 오류
  │     조치:
  │       1. input_nvt.yaml의 dt를 절반으로 줄임 (예: 1.0 fs → 0.5 fs)
  │       2. 소스 코드에서 Nosé-Hoover 체인 변수 초기화 확인
  │       3. 힘 계산 직전/직후에 DEBUG 로그 추가:
  │            fprintf(stderr, "[DEBUG NVT step=%d] F_max=%.6f\n", step, f_max);
  │       4. 재빌드 → 재실행 → 로그 확인 → 원인 수정
  │
  ├─ [case B] 온도 수렴 안 됨 (오차 > 3%)
  │     원인: thermostat_damping 값 부적절, 체인 길이 부족, 구현 버그
  │     조치:
  │       1. thermostat_damping을 20배 증가 (100 → 2000 fs) 로 재시도
  │          → 여전히 실패하면 10배 감소 (100 → 10 fs) 로 재시도
  │       2. 소스 코드에서 Nosé-Hoover 업데이트 순서 확인
  │          (LAMMPS 기준: velocity-Verlet + half-step thermostat 방식)
  │       3. 체인 변수 개수(chain length)가 1 이상인지 확인
  │       4. 온도 계산 수식 확인: T = 2 * KE / (3 * N * k_B)
  │       5. DEBUG 로그 추가:
  │            fprintf(stderr, "[DEBUG NVT] step=%d T=%.4f xi=%.6f\n",
  │                    step, temperature, xi);
  │       6. 재빌드 → 재실행 → validate_nvt.sh
  │
  ├─ [case C] 에너지 드리프트
  │     원인: 적분기 에너지 보존 불량, 체인 변수 누적 오차
  │     조치:
  │       1. dt를 절반으로 줄여서 재시도
  │       2. 확장 해밀토니안(H_ext) 보존 여부를 로그로 출력:
  │            fprintf(stderr, "[DEBUG NVT] H_ext=%.8f\n", H_kinetic + H_potential + H_thermostat);
  │       3. 적분기를 Suzuki-Yoshida 방식으로 교체 검토
  │
  └─ [case D] 파싱 실패 (온도/에너지 값을 못 읽음)
        원인: 출력 포맷이 validate_nvt.sh의 grep 패턴과 불일치
        조치:
          1. 실제 출력 포맷 확인: cat /tmp/nvt_out.txt | head -30
          2. validate_nvt.sh의 정규식 패턴을 실제 포맷에 맞게 수정
          3. 수정 후 재실행
```

모든 조치는 `.bak` 백업 후 적용하며, 빌드 실패 시 즉시 복원 후 다른 조치 시도.

#### 5-1-E. NVT 완료 조건

```
✅ 평균 온도 오차 ≤ 3.0%
✅ NaN/Inf 없음
✅ 에너지 드리프트 없음 (변동 폭 < 5%)
→ 위 3가지 모두 만족 시 즉시 5-2 (NPT) 시작
```

---

### 5-2. NPT 테스트 (NVT 통과 후에만 시작)

NPT는 내부적으로 NVT(thermostat) + 부피 조절(barostat)이 결합된 앙상블이다.
**NVT가 정상 동작함을 확인한 뒤 NPT를 시작해야** thermostat 버그와 barostat 버그를 분리할 수 있다.

#### 5-2-A. NVT 결과로 NPT 초기 상태 구성

NPT를 바로 cold-start하면 초기 압력 스파이크로 시뮬레이션이 불안정해질 수 있다.
NVT equilibration 결과(좌표 + 속도)를 NPT 시작점으로 사용한다.

```yaml
# examples/lj_32000/input_npt.yaml
ensemble: "NPT"
thermostat: "nose_hoover"
barostat: "parrinello_rahman"    # 또는 "berendsen" — 구현된 것 사용
target_temperature: 300.0
target_pressure: 1.0             # bar, LAMMPS와 단위 통일 필수
thermostat_damping: 100.0        # NVT에서 통과한 값 그대로 사용
barostat_damping: 1000.0         # thermostat_damping의 10배 권장
steps: 20000                     # NPT는 equilibration이 더 오래 걸림
thermo_output_interval: 100
pressure_output: true            # 압력 출력 반드시 활성화

# NVT 결과를 초기 상태로 사용 (구현에 따라 필드명 다를 수 있음)
# restart_file: "output/nvt_final.restart"   ← 코드가 restart 지원 시
# 또는 NVT 마지막 frame의 좌표/속도를 읽는 방식으로 대체
```

```bash
# NVT 완료 확인 후 NPT 실행
make 2>&1 | tail -3
./md_simulation_cpu examples/lj_32000/input_npt.yaml 2>&1 | tee /tmp/npt_out.txt
echo "종료 코드: $?"
tail -30 /tmp/npt_out.txt
```

#### 5-2-B. NPT 자동 검증 스크립트

```bash
#!/bin/bash
# scripts/validate_npt.sh
TARGET_TEMP=300.0
TARGET_PRES=1.0    # bar (input_npt.yaml의 target_pressure 값)
LOG=/tmp/npt_out.txt

echo "=== [STAGE 5-2] NPT 검증 ==="

if [ ! -f "$LOG" ]; then
    echo "❌ 로그 파일 없음: $LOG"
    exit 1
fi

python3 - <<PYEOF
import sys, math, re

TARGET_TEMP = $TARGET_TEMP
TARGET_PRES = $TARGET_PRES
log = open("$LOG").read()

# ── NaN / Inf 체크 ──────────────────────────────────────
if re.search(r'\bnan\b|\binf\b', log, re.IGNORECASE):
    print("❌ FAIL: NaN/Inf 감지")
    for line in log.splitlines():
        if re.search(r'\bnan\b|\binf\b', line, re.IGNORECASE):
            print(f"  [DEBUG] > {line}")
    sys.exit(2)

# ── 온도 파싱 ────────────────────────────────────────────
temp_values = re.findall(r'[Tt]emp(?:erature)?\s*[=:]\s*([\d.+-]+)', log)
if not temp_values:
    temp_values = re.findall(r'^\s*\d+\s+([\d.+-]+)', log, re.MULTILINE)

# ── 압력 파싱 ────────────────────────────────────────────
pres_values = re.findall(r'[Pp]res(?:sure)?\s*[=:]\s*([\d.+eE-]+)', log)
if not pres_values:
    # 세 번째 컬럼 시도 (Step Temp Press ... 포맷)
    pres_values = re.findall(r'^\s*\d+\s+[\d.+-]+\s+([\d.+eE-]+)', log, re.MULTILINE)

# ── 부피 파싱 ────────────────────────────────────────────
vol_values  = re.findall(r'[Vv]ol(?:ume)?\s*[=:]\s*([\d.+eE-]+)', log)

# ── 분석 ─────────────────────────────────────────────────
def analyze(values, name, target, threshold, unit=""):
    if not values:
        print(f"  ⚠ {name} 파싱 실패 — 로그 포맷 확인 필요")
        print(f"    [DEBUG] 로그 마지막 10줄:")
        for line in log.splitlines()[-10:]:
            print(f"      | {line}")
        return None, None
    vals  = [float(v) for v in values]
    skip  = max(1, len(vals) // 5)     # 전반부 20% equilibration 제외
    eq    = vals[skip:]
    avg   = sum(eq) / len(eq)
    err   = abs(avg - target) / abs(target) * 100 if target != 0 else float('inf')
    return avg, err

avg_temp, err_temp = analyze(temp_values, "평균 온도", TARGET_TEMP, 3.0, "K")
avg_pres, err_pres = analyze(pres_values, "평균 압력", TARGET_PRES, 5.0, "bar")

# 부피 발산 체크
vol_ok = True
if vol_values and len(vol_values) >= 5:
    vols   = [float(v) for v in vol_values]
    v_init = vols[0]
    v_final= vols[-1]
    vol_change = abs(v_final - v_init) / abs(v_init) * 100
    if vol_change > 50:    # 초기 부피 대비 50% 이상 변화는 발산으로 판단
        vol_ok = False

W = 60
print(f"\n{'─'*W}")
print(f"  {'항목':<25} {'측정값':>12} {'기준':>8} {'오차':>7}  판정")
print(f"{'─'*W}")

temp_ok = False
if avg_temp is not None:
    temp_ok = err_temp <= 3.0
    print(f"  {'평균 온도 (K)':<25} {avg_temp:>12.2f} {TARGET_TEMP:>8.1f} {err_temp:>6.2f}%  {'✅' if temp_ok else '❌'}")
else:
    print(f"  {'평균 온도 (K)':<25} {'파싱 실패':>12} {TARGET_TEMP:>8.1f} {'N/A':>7}  ⚠")

pres_ok = False
if avg_pres is not None:
    pres_ok = err_pres <= 5.0
    print(f"  {'평균 압력 (bar)':<25} {avg_pres:>12.4f} {TARGET_PRES:>8.2f} {err_pres:>6.2f}%  {'✅' if pres_ok else '❌'}")
else:
    print(f"  {'평균 압력 (bar)':<25} {'파싱 실패':>12} {TARGET_PRES:>8.2f} {'N/A':>7}  ⚠")

print(f"  {'박스 부피 안정성':<25} {'OK' if vol_ok else '발산':>12} {'발산없음':>8} {'—':>7}  {'✅' if vol_ok else '❌'}")
print(f"{'─'*W}")

all_ok = temp_ok and pres_ok and vol_ok
if all_ok:
    print("  → ✅ NPT PASS")
    sys.exit(0)
else:
    if not temp_ok:
        print(f"  → ❌ 온도 오차 {err_temp:.2f}% > 3.0%")
    if not pres_ok:
        print(f"  → ❌ 압력 오차 {err_pres:.2f}% > 5.0%")
    if not vol_ok:
        print(f"  → ❌ 박스 부피 발산")
    sys.exit(1)
PYEOF
```

```bash
chmod +x scripts/validate_npt.sh
bash scripts/validate_npt.sh
```

#### 5-2-C. NPT 실패 시 자동 디버깅 루프

검증 실패 시 아래 케이스를 **승인 없이 즉시** 순서대로 시도한다.

```
NPT 검증 실패
  │
  ├─ [case A] NaN / Inf 발생 (압력/부피 계산 단계에서 발생 가능성 높음)
  │     조치:
  │       1. barostat_damping을 10배 증가 (1000 → 10000 fs)
  │       2. 소스 코드에서 virial 계산 직후 DEBUG 로그 추가:
  │            fprintf(stderr, "[DEBUG NPT step=%d] P=%.6f V=%.6f\n",
  │                    step, pressure, volume);
  │       3. Parrinello-Rahman이 구현 안 됐으면 먼저 Berendsen barostat로 교체
  │          (Berendsen은 단순하지만 correct ensemble은 아님 — 기능 확인용)
  │       4. 재빌드 → 재실행
  │
  ├─ [case B] 온도 수렴 안 됨 (NPT에서 thermostat만 문제)
  │     조치:
  │       1. NVT에서 통과한 thermostat_damping 값을 그대로 사용 확인
  │       2. NPT 통합 루프에서 thermostat 업데이트 순서 확인
  │          (barostat 적용 전후 thermostat 호출 순서가 중요)
  │       3. DEBUG 로그로 thermostat 체인 변수 추적:
  │            fprintf(stderr, "[DEBUG NPT] T=%.4f xi=%.6f eta=%.6f\n",
  │                    temp, xi, eta);
  │
  ├─ [case C] 압력 수렴 안 됨 (오차 > 5%)
  │     조치:
  │       1. barostat_damping을 2배씩 증가하며 재시도 (1000→2000→4000→8000)
  │       2. steps를 2배 늘려 더 긴 equilibration 후 평균 재계산
  │       3. virial 계산 수식 확인 (LJ pair virial 포함 여부)
  │       4. 단위계 확인: 압력 단위가 bar인지 atm인지 Pa인지
  │       5. DEBUG 로그로 매 step 압력 추적:
  │            fprintf(stderr, "[DEBUG NPT] step=%d P_inst=%.4f V=%.4f\n",
  │                    step, pressure, volume);
  │
  ├─ [case D] 박스 부피 발산
  │     조치:
  │       1. barostat_damping을 10배 증가
  │       2. Parrinello-Rahman → Berendsen으로 임시 교체 (안정성 우선)
  │       3. 초기 압력 스파이크 확인:
  │            head -50 /tmp/npt_out.txt 에서 초기 압력값 확인
  │          → 스파이크가 있으면 NVT restart 파일 사용 또는 pre-equilibration 추가
  │       4. 부피 변화 상한선(max_volume_change)을 소스 코드에 추가
  │
  └─ [case E] 파싱 실패 (압력/부피 출력 없음)
        조치:
          1. input_npt.yaml에서 pressure_output: true 확인
          2. 소스 코드에서 NPT 실행 시 압력/부피 출력 코드가 있는지 확인
          3. 없으면 출력 코드 추가:
               printf("Step=%d Temp=%.4f Press=%.6f Volume=%.4f TotEng=%.6f\n",
                      step, temperature, pressure, volume, total_energy);
          4. validate_npt.sh의 grep 패턴을 실제 포맷에 맞게 수정
```

모든 조치는 `.bak` 백업 후 적용하며, 빌드 실패 시 즉시 복원 후 다음 조치 시도.

---

### 5-3. 자동 반복 루프 전체 흐름

```
START
  │
  ▼
[NVT 실행]
  │  ./md_simulation_cpu examples/lj_32000/input_nvt.yaml | tee /tmp/nvt_out.txt
  ▼
[NVT 검증] bash scripts/validate_nvt.sh
  │  PASS? ──NO──▶ 디버깅 조치 (5-1-D) → 소스 수정 → make → 재실행
  │  YES
  ▼
[NPT 실행]  ← NVT 통과 확인 후에만 진입
  │  ./md_simulation_cpu examples/lj_32000/input_npt.yaml | tee /tmp/npt_out.txt
  ▼
[NPT 검증] bash scripts/validate_npt.sh
  │  PASS? ──NO──▶ 디버깅 조치 (5-2-C) → 소스 수정 → make → 재실행
  │  YES
  ▼
STAGE 5 DONE ✅
→ 즉시 STAGE 6 시작
```

---

### 5-4. 결과 기록 형식

```
[STAGE 5 — Thermostat/Barostat 검증 결과]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NVT (Nosé-Hoover)
  평균 온도    : _____ K  (목표: _____ K, 오차 _____%)  ✅/❌
  에너지 드리프트: 안정 / 드리프트                       ✅/❌
  반복 횟수    : _____회 만에 통과
  적용된 수정  : _______________
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NPT (NVT 통과 후 시작)
  평균 온도    : _____ K  (오차 _____%)                 ✅/❌
  평균 압력    : _____ bar  (오차 _____%)               ✅/❌
  박스 부피    : 안정 / 발산                             ✅/❌
  반복 횟수    : _____회 만에 통과
  적용된 수정  : _______________
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 완료 조건

```
✅ NVT: 평균 온도 오차 ≤ 3.0%, 에너지 드리프트 없음
✅ NPT: 평균 온도 오차 ≤ 3.0%, 평균 압력 오차 ≤ 5.0%, 부피 안정
→ 위 조건 달성 시 승인 없이 즉시 STAGE 6 시작
```

---



### ⚠️ STAGE 6 핵심 원칙

이 MD 코드는 MACE·NequIP 모델 파일을 **C++에서 직접 로드하여 에너지·힘 계산**을
수행하는 구조다. Python/ASE 레이어는 전혀 사용하지 않는다.
모든 연동은 **LibTorch (C++ PyTorch API)** 를 통해 이루어진다.

작업 순서는 반드시 아래를 따른다:
1. 소스코드를 먼저 **전부 읽는다** — 구조를 모르는 채로 코드를 작성하지 않는다.
2. 현재 코드에서 MLP 연동 지점(potential 선택, force 계산 진입점)을 정확히 파악한다.
3. 파악한 구조에 맞게 MACE/NequIP 인터페이스를 구현하거나 수정한다.

### 전체 흐름

```
STAGE 6
  │
  ├─ 6-A  소스코드 전체 분석
  │         ├─ 파일 구조 파악
  │         ├─ MLP 연동 지점 식별 (potential 추상화 구조)
  │         ├─ 현재 LibTorch 연동 여부 확인
  │         └─ input.yaml 파싱 방식 확인
  │
  ├─ 6-B  모델 파일 준비
  │         ├─ MACE: .model → TorchScript .pt 변환
  │         └─ NequIP: nequip-compile → TorchScript .pt2 변환
  │
  ├─ 6-C  MACE 연동 구현
  │         ├─ 소스 구조에 맞는 MACEPotential 클래스 작성
  │         ├─ input.yaml "potential_type: mace" 분기 연결
  │         ├─ Makefile LibTorch 링크 추가
  │         └─ 빌드 + 에너지·힘 검증
  │
  ├─ 6-D  NequIP 연동 구현
  │         ├─ 소스 구조에 맞는 NequIPPotential 클래스 작성
  │         ├─ input.yaml "potential_type: nequip" 분기 연결
  │         └─ 빌드 + 에너지·힘 검증
  │
  ├─ 6-E  벤치마크 (LJ vs MACE vs NequIP, CPU/CUDA)
  │
  └─ 6-F  추론 최적화 (자동 반복 루프)
            ├─ MD 코드 GPU 최적화 (neighbor list, H2D, pipelining)
            └─ MLP 추론 GPU 최적화 (LibTorch, CUDA Graph, stream)
```

---

### 6-A: 소스코드 전체 분석

**이 단계를 건너뛰고 코드를 작성하지 않는다.**

```bash
# 소스 파일 목록
find . \( -name "*.cpp" -o -name "*.cu" -o -name "*.h" -o -name "*.hpp" \) \
  | grep -v "build\|\.git" | sort

# MLP/Potential 관련 키워드 탐색
grep -rn "potential\|mlp\|mace\|nequip\|torch\|libtorch\|model_path\|compute_forces" \
  --include="*.cpp" --include="*.cu" --include="*.h" . | grep -v "build"

# input.yaml 파싱 코드 위치
grep -rn "potential_type\|model_path\|yaml\|config" \
  --include="*.cpp" --include="*.h" . | grep -v "build"

# Makefile 전체 출력
cat Makefile
```

아래 항목을 파악하여 기록한다:

```
[6-A 소스코드 분석 결과]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Potential 추상화 파일  : (예: src/potential/base_potential.h)
MLP 인터페이스 파일    : 없음 / (파일명)
input.yaml 파싱 위치   : (파일명 + 라인)
force 계산 진입점      : (함수명 + 파일명)
LibTorch 링크 여부     : Makefile에 있음 / 없음
현재 모델 로드 코드    : 없음 / torch::jit::load() 있음
GPU neighbor list 구조 : (설명)
데이터 레이아웃        : AoS / SoA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

### 6-B: 모델 파일 준비

#### 6-B-1. 환경 확인

```bash
# LibTorch 경로 (C++ 빌드에 필요)
python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
python3 -c "import torch; print('LibTorch lib:', torch.__file__.replace('__init__.py','lib'))"

# CUDA 지원 확인
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.version.cuda)"

# ABI 확인 (Makefile에 맞춰야 함)
python3 -c "import torch; print('CXX11 ABI:', torch._C._GLIBCXX_USE_CXX11_ABI)"
```

#### 6-B-2. MACE 모델 → TorchScript .pt 변환

C++의 `torch::jit::load()`는 TorchScript `.pt` 파일을 직접 로드한다.

```bash
mkdir -p models/mace models/nequip

pip install mace-torch e3nn

# MACE-MP-0 medium 다운로드 (GitHub: ACEsuit/mace-mp)
python3 - <<'PYEOF'
import urllib.request, os
url = ("https://github.com/ACEsuit/mace-mp/releases/download/"
       "mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model")
out = "models/mace/mace_mp_medium.model"
if not os.path.exists(out):
    urllib.request.urlretrieve(url, out)
    print(f"다운로드 완료: {out}")
else:
    print(f"이미 존재: {out}")
PYEOF

# C++에서 직접 로드 가능한 TorchScript .pt로 변환
python3 -m mace.cli.create_lammps_model \
    models/mace/mace_mp_medium.model \
    models/mace/mace_mp_medium.pt

# 모델 입력/출력 키 확인 (C++ 코드 작성에 반드시 필요)
python3 - <<'PYEOF'
import torch
m = torch.jit.load("models/mace/mace_mp_medium.pt")
m.eval()
print("=== MACE forward 시그니처 ===")
print(m.code[:3000])
print("\n=== 속성 목록 ===")
for name, _ in m.named_parameters():
    print(f"  param: {name}")
for k in m._modules:
    print(f"  module: {k}")
PYEOF
```

#### 6-B-3. NequIP 모델 → TorchScript .pt2 변환

```bash
pip install nequip

# GitHub: mir-group/nequip
# TorchScript 모드로 컴파일 (Python 의존성 없이 C++에서 로드 가능)
nequip-compile \
    nequip.net:mir-group/NequIP-OAM-L:0.1 \
    models/nequip/NequIP-OAM-L_cpu.pt2 \
    --mode torchscript --device cpu

nequip-compile \
    nequip.net:mir-group/NequIP-OAM-L:0.1 \
    models/nequip/NequIP-OAM-L_cuda.pt2 \
    --mode torchscript --device cuda

# 모델 키 확인
python3 - <<'PYEOF'
import torch
m = torch.jit.load("models/nequip/NequIP-OAM-L_cpu.pt2")
m.eval()
print("=== NequIP forward 시그니처 ===")
print(m.code[:3000])
# cutoff, type_names 확인
try: print("r_max:", m.r_max)
except: pass
try: print("type_names:", m.type_names)
except: pass
PYEOF
```

nequip.net 접근 불가 시:
```bash
nequip-compile nequip.net:mir-group/NequIP-OAM-L:0.1 \
    models/nequip/NequIP-OAM-L.pkg --mode package
nequip-compile models/nequip/NequIP-OAM-L.pkg \
    models/nequip/NequIP-OAM-L_cpu.pt2 --mode torchscript --device cpu
```

---

### 6-C: Makefile LibTorch 링크 추가

6-A에서 파악한 Makefile 구조에 맞게 LibTorch를 링크한다.

```makefile
# ── LibTorch 경로 (python으로 자동 감지) ─────────────────────────
TORCH_BASE := $(shell python3 -c \
    "import torch, os; print(os.path.dirname(torch.__file__))")
TORCH_INC  := $(TORCH_BASE)/include
TORCH_LIB  := $(TORCH_BASE)/lib
TORCH_CMAK := $(TORCH_BASE)/share/cmake/Torch

# ── 컴파일 플래그 ────────────────────────────────────────────────
CXXFLAGS_TORCH := \
    -I$(TORCH_INC) \
    -I$(TORCH_INC)/torch/csrc/api/include \
    -D_GLIBCXX_USE_CXX11_ABI=$(shell python3 -c \
        "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")

# ── 링크 플래그 (CPU) ────────────────────────────────────────────
LDFLAGS_TORCH := \
    -L$(TORCH_LIB) -ltorch -ltorch_cpu -lc10 \
    -Wl,-rpath,$(TORCH_LIB)

# ── 링크 플래그 (CUDA) ───────────────────────────────────────────
LDFLAGS_TORCH_CUDA := \
    $(LDFLAGS_TORCH) -ltorch_cuda -lc10_cuda
```

빌드 확인:
```bash
make 2>&1 | tee /tmp/build_torch.txt
[ $? -ne 0 ] && echo "❌ 빌드 실패" && tail -40 /tmp/build_torch.txt
make cuda 2>&1 | tee /tmp/build_torch_cuda.txt
[ $? -ne 0 ] && echo "❌ CUDA 빌드 실패" && tail -40 /tmp/build_torch_cuda.txt
```

---

### 6-D: MACE C++ 연동 구현

6-A에서 파악한 소스 구조에 맞는 위치에 MACE 인터페이스를 구현한다.

#### 핵심 구현 패턴

```cpp
// 위치: 6-A에서 파악한 potential 파일 (없으면 새로 생성)
// ASE 없음. torch::jit::load() 로 .pt 파일 직접 사용.

#include <torch/script.h>
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>   // CUDA 빌드 시

class MACEPotential {
public:
    torch::jit::script::Module model_;
    torch::Device device_;
    bool use_cuda_;

    // 모델 로드 — 최초 1회
    MACEPotential(const std::string& model_path, bool use_cuda)
        : device_(use_cuda ? torch::kCUDA : torch::kCPU)
        , use_cuda_(use_cuda)
    {
        model_ = torch::jit::load(model_path, device_);
        model_.eval();
        model_.to(torch::kFloat32);                         // MD는 float32 충분
        torch::jit::optimize_for_inference(model_);        // inference 최적화
        if (use_cuda_) {
            // GPU 메모리 warm-up: 첫 step에서 cudaMalloc 지연 제거
            torch::cuda::synchronize();
        }
    }

    // 에너지·힘 계산
    // positions, atom_types, edge_src/dst, edge_shifts: MD 코드가 관리하는 데이터
    void compute(
        const float*  positions,   // [N,3] GPU/CPU 포인터
        const int64_t* z,          // [N]   atomic numbers (GPU/CPU)
        const int64_t* edge_src,   // [M]
        const int64_t* edge_dst,   // [M]
        const float*  shifts,      // [M,3] PBC shift vectors
        int N, int M,
        double& energy,
        float*  forces             // [N,3] 출력 (in-place)
    ) {
        torch::NoGradGuard no_grad;   // autograd 비활성화 — 추론 속도 ↑

        auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
        auto opts_i = torch::TensorOptions().dtype(torch::kInt64).device(device_);

        // ── Tensor 생성 (zero-copy: GPU 포인터 그대로 사용) ────────────
        torch::Tensor pos_t, z_t, ei_t, sh_t, batch_t;

        if (use_cuda_) {
            // GPU 메모리 → Tensor (복사 없음)
            pos_t = torch::from_blob((void*)positions, {N,3}, opts_f);
            z_t   = torch::from_blob((void*)z,         {N},   opts_i);
            auto src_t = torch::from_blob((void*)edge_src, {M}, opts_i);
            auto dst_t = torch::from_blob((void*)edge_dst, {M}, opts_i);
            ei_t  = torch::stack({src_t, dst_t}, 0);
            sh_t  = torch::from_blob((void*)shifts, {M,3}, opts_f);
        } else {
            // CPU 메모리 → Tensor
            pos_t = torch::from_blob((void*)positions, {N,3},
                        torch::TensorOptions().dtype(torch::kFloat32));
            z_t   = torch::from_blob((void*)z, {N},
                        torch::TensorOptions().dtype(torch::kInt64));
            auto src_t = torch::from_blob((void*)edge_src, {M},
                        torch::TensorOptions().dtype(torch::kInt64));
            auto dst_t = torch::from_blob((void*)edge_dst, {M},
                        torch::TensorOptions().dtype(torch::kInt64));
            ei_t  = torch::stack({src_t, dst_t}, 0);
            sh_t  = torch::from_blob((void*)shifts, {M,3},
                        torch::TensorOptions().dtype(torch::kFloat32));
        }
        batch_t = torch::zeros({N}, opts_i);

        // ── MACE forward 호출 ──────────────────────────────────────────
        // 실제 키는 6-B에서 확인한 forward 시그니처로 맞출 것
        c10::Dict<std::string, torch::IValue> input;
        input.insert("positions",   pos_t);
        input.insert("node_attrs",  torch::nn::functional::one_hot(z_t, 89)
                                        .to(torch::kFloat32));
        input.insert("batch",       batch_t);
        input.insert("edge_index",  ei_t);
        input.insert("shifts",      sh_t);
        input.insert("unit_shifts", sh_t);

        auto out = model_.forward({input}).toGenericDict();

        // ── 에너지 추출 ────────────────────────────────────────────────
        energy = out.at("energy").toTensor().item<float>();

        // ── 힘 추출 (모델 직접 출력 사용) ────────────────────────────
        if (out.contains("forces")) {
            auto f = out.at("forces").toTensor();  // [N,3] float32, same device
            if (use_cuda_) {
                // GPU 포인터에 직접 복사 (D2D, 빠름)
                cudaMemcpy(forces, f.data_ptr<float>(),
                           N * 3 * sizeof(float), cudaMemcpyDeviceToDevice);
            } else {
                memcpy(forces, f.data_ptr<float>(), N * 3 * sizeof(float));
            }
        }
    }
};
```

#### input.yaml 연결

6-A에서 파악한 파싱 코드에 아래 분기를 추가한다:

```yaml
# examples/mlp_mace/input.yaml
potential_type: "mace"
model_path:     "models/mace/mace_mp_medium.pt"
device:         "cuda"    # 또는 "cpu"
dtype:          "float32"
cutoff:         6.0
```

```cpp
// input.yaml 파싱 후 potential 선택 (기존 코드에 추가)
if (config.potential_type == "mace") {
    potential = std::make_unique<MACEPotential>(
        config.model_path,
        config.device == "cuda"
    );
}
```

---

### 6-D: NequIP C++ 연동 구현

```cpp
class NequIPPotential {
public:
    torch::jit::script::Module model_;
    torch::Device device_;

    NequIPPotential(const std::string& model_path, bool use_cuda)
        : device_(use_cuda ? torch::kCUDA : torch::kCPU)
    {
        model_ = torch::jit::load(model_path, device_);
        model_.eval();
        torch::jit::optimize_for_inference(model_);
    }

    void compute(
        const double* positions,  // [N,3] — NequIP은 float64 권장
        const int64_t* types,     // [N]  type index (0-based, NOT atomic number)
        const int64_t* edge_src, const int64_t* edge_dst,
        const double* edge_vec,   // [M,3] 실제 벡터 (shift 포함)
        int N, int M,
        double& energy,
        double* forces            // [N,3]
    ) {
        torch::NoGradGuard no_grad;
        auto opts_d = torch::TensorOptions().dtype(torch::kFloat64).device(device_);
        auto opts_i = torch::TensorOptions().dtype(torch::kInt64).device(device_);

        c10::Dict<std::string, torch::Tensor> input;
        input.insert("pos",
            torch::from_blob((void*)positions, {N,3}, opts_d).clone());
        input.insert("atom_types",
            torch::from_blob((void*)types, {N}, opts_i));
        auto src_t = torch::from_blob((void*)edge_src, {M}, opts_i);
        auto dst_t = torch::from_blob((void*)edge_dst, {M}, opts_i);
        input.insert("edge_index", torch::stack({src_t, dst_t}, 0));
        input.insert("edge_cell_shift",
            torch::from_blob((void*)edge_vec, {M,3}, opts_d).clone());
        input.insert("batch", torch::zeros({N}, opts_i));

        // 실제 키는 6-B에서 확인한 forward 코드로 맞출 것
        auto out = model_.forward({input}).toGenericDict();

        energy = out.at("total_energy").toTensor().item<double>();
        auto f = out.at("forces").toTensor().to(torch::kCPU);
        memcpy(forces, f.data_ptr<double>(), N * 3 * sizeof(double));
    }
};
```

```yaml
# examples/mlp_nequip/input.yaml
potential_type: "nequip"
model_path:     "models/nequip/NequIP-OAM-L_cuda.pt2"  # cuda 빌드 시
device:         "cuda"
cutoff:         6.0   # 6-B에서 확인한 실제 r_max로 교체
type_map:
  0: "Cu"             # 6-B에서 확인한 type_names로 교체
```

---

### 6-E: 정확성 검증

```bash
#!/bin/bash
# scripts/validate_mlp.sh  사용법: ./scripts/validate_mlp.sh mace|nequip cpu|cuda
MODEL=$1; DEVICE=$2
BIN="./md_simulation_$([ "$DEVICE" = "cuda" ] && echo cuda || echo cpu)"
LOG="/tmp/mlp_${MODEL}_${DEVICE}.txt"

$BIN examples/mlp_${MODEL}/input.yaml 2>&1 | tee $LOG
python3 - <<PYEOF
import sys, re
log = open("$LOG").read()
if re.search(r'\bnan\b|\binf\b', log, re.IGNORECASE):
    print("❌ NaN/Inf 감지"); sys.exit(1)
e_vals = [float(x) for x in re.findall(
    r'[Tt]ot(?:al)?\s*[Ee]nergy\s*[=:]\s*([-\d.eE+]+)', log)]
if not e_vals:
    print("⚠ 에너지 파싱 실패 — 출력 포맷 확인"); sys.exit(2)
skip = max(1, len(e_vals)//5)
eq = e_vals[skip:]
drift = abs(max(eq)-min(eq)) / abs(sum(eq)/len(eq)) * 100
t_m = re.search(r'Total execution time:\s*([\d.]+)', log)
t = float(t_m.group(1)) if t_m else -1
print(f"  에너지 드리프트: {drift:.2f}%  {'✅' if drift<5 else '❌'}")
print(f"  실행시간: {t:.3f} s")
sys.exit(0 if drift < 5 else 3)
PYEOF
```

```bash
chmod +x scripts/validate_mlp.sh
bash scripts/validate_mlp.sh mace  cpu
bash scripts/validate_mlp.sh mace  cuda
bash scripts/validate_mlp.sh nequip cpu
bash scripts/validate_mlp.sh nequip cuda
```

---

### 6-F: 추론 최적화 — GPU 최적화 중심 자동 반복 루프

검증을 통과한 뒤 아래 전략을 우선순위 순서대로 적용한다.
각 전략 적용 후 반드시 `validate_mlp.sh`와 `benchmark_mlp.sh`를 실행하여
정확성과 성능을 동시에 확인한다. **에러나 성능 저하 시 즉시 `.bak` 복원 후 다음 전략으로.**

#### ━━ 범주 1: MD 코드 GPU 최적화 ━━━━━━━━━━━━━━━━━━━━━━━━━━━

이 범주는 MLP와 무관하게 MD 코드 자체의 GPU 병렬성을 최대화한다.

| # | 전략 | 대상 | 예상 효과 |
|---|------|------|-----------|
| G1 | **GPU Neighbor List 구축**: CPU에서 하던 neighbor list를 CUDA 커널로 이전. Cell-linked list를 GPU에서 직접 구축하여 매 step CPU↔GPU 전송 제거 | MD 코드 | 30~60% |
| G2 | **원자 데이터 GPU 상주 (Persistent GPU Buffers)**: `positions`, `velocities`, `forces` 배열을 매 step CPU로 가져오지 않고 GPU 메모리에 상주. step 중 H2D/D2H 전송 제거 | MD 코드 | 10~30% |
| G3 | **CUDA Stream 파이프라이닝**: neighbor list 구축(stream 0)과 MLP 추론(stream 1)을 겹쳐 실행. neighbor list 완성 즉시 MLP 추론 시작 | MD 코드 | 20~40% |
| G4 | **SoA 레이아웃 강제 (GPU 메모리 coalescing)**: `x[i], y[i], z[i]` 분리 배열 → GPU warp 단위 coalesced 접근. `float3`/`double3` struct 대신 SoA 사용 | MD 코드 | 10~25% |
| G5 | **Verlet skin + step skip**: skin = 0.3 Å 설정 시 neighbor list를 평균 10~20 step에 1번만 재구축. rebuild 판단도 GPU에서 수행 | MD 코드 | 15~40% |
| G6 | **Force accumulation atomic 최소화**: Newton 3rd law GPU 구현 시 `atomicAdd` 충돌 줄이기 — atom reordering으로 같은 warp 내 충돌 감소 | MD 코드 | 5~15% |

#### ━━ 범주 2: MLP 추론 GPU 최적화 ━━━━━━━━━━━━━━━━━━━━━━━━━━

| # | 전략 | 대상 | 예상 효과 |
|---|------|------|-----------|
| M1 | **`torch::NoGradGuard` + `model.eval()`**: autograd graph 생성 비활성화. 추론 전용 경로 사용 | 공통 | 5~20% |
| M2 | **`torch::jit::optimize_for_inference(model)`**: op fusion, constant folding 적용. 모델 로드 직후 1회 호출 | 공통 | 10~30% |
| M3 | **float32 강제 (MACE)**: `model.to(torch::kFloat32)`. float64 대비 2x GPU 처리량 향상. 에너지 드리프트 확인 필수 | MACE | 1.5~2x |
| M4 | **GPU 메모리 재사용 (Tensor 재생성 제거)**: step마다 `torch::from_blob()` 재생성 대신 고정 크기 Tensor를 한 번 allocate하고 data만 업데이트 (`copy_()` 사용) | 공통 | 10~25% |
| M5 | **H2D 전송 제거**: MD 코드가 이미 GPU에 positions를 보유하고 있으면 `torch::from_blob(cuda_ptr, ...)` 로 zero-copy Tensor 생성 (복사 없음). 6-D 구현 패턴 참고 | 공통 | 15~35% |
| M6 | **CUDA Graph 캡처**: step N 이후 neighbor list와 atom 수가 고정되면 `cudaGraph`로 MLP 추론 전체를 캡처. 이후 `cudaGraphLaunch` 1회로 실행 — kernel launch overhead 제거 | 공통 | 5~20% |
| M7 | **CUDA Stream으로 비동기 추론**: MLP forward를 별도 CUDA stream에서 실행. integration 커널(velocity update 등)과 MLP 추론을 overlap | 공통 | 20~40% |
| M8 | **cuEquivariance (MACE)**: `enable_cueq=True` 설정 시 Tensor Product 연산이 fused CUDA kernel로 대체 — MACE GPU에서 가장 큰 단일 최적화 | MACE | 2~4x |
| M9 | **OpenEquivariance (NequIP)**: `enable_OpenEquivariance` modifier로 equivariant op fusion 활성화 | NequIP | 1.5~3x |

#### ━━ 범주 3: LibTorch 컴파일러 최적화 ━━━━━━━━━━━━━━━━━━━━━

| # | 전략 | 대상 | 예상 효과 |
|---|------|------|-----------|
| C1 | **`torch::jit::setFusionStrategy`**: NVFuser / TensorExprFuser 설정으로 elementwise op kernel fusion 활성화 | CUDA | 10~20% |
| C2 | **`torch.compile()` 동등 효과 (C++)**: TorchScript 모델에 `torch::jit::optimize_for_inference` + `--optimize` 플래그 | 공통 | 10~25% |
| C3 | **CUDA Unified Memory 비활성화**: `PYTORCH_NO_CUDA_MEMORY_CACHING=0` 환경변수로 memory pool 관리 최적화 | CUDA | 5~10% |

#### ━━ 구현 예시: CUDA Graph 캡처 (M6) ━━━━━━━━━━━━━━━━━━━━━

```cpp
// CUDA Graph로 MLP 추론 캡처 — step 수가 충분히 진행된 후 capture
class MLPWithCUDAGraph {
    cudaGraph_t graph_;
    cudaGraphExec_t graph_exec_;
    bool graph_captured_ = false;

    // 고정 크기 입력/출력 Tensor (step마다 data만 업데이트)
    torch::Tensor pos_t_, ei_t_, sh_t_, z_t_, batch_t_;
    torch::Tensor energy_t_, forces_t_;

public:
    void capture(MACEPotential& pot, int N, int M) {
        // warm-up 3회 (graph capture 전 필수)
        for (int i = 0; i < 3; i++) {
            pot.model_.forward({/* 고정 입력 */});
        }
        // graph capture
        cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
        pot.model_.forward({/* 고정 입력 텐서 */});
        cudaStreamEndCapture(stream_, &graph_);
        cudaGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0);
        graph_captured_ = true;
    }

    void execute() {
        // data 업데이트 후 graph replay — kernel launch 최소화
        pos_t_.copy_(new_pos);   // GPU→GPU, 빠름
        cudaGraphLaunch(graph_exec_, stream_);
        cudaStreamSynchronize(stream_);
    }
};
```

#### ━━ 구현 예시: CUDA Stream 파이프라이닝 (G3 + M7) ━━━━━━━━━

```cpp
// Stream 0: neighbor list 구축 (CUDA 커널)
// Stream 1: MLP forward (LibTorch)
// → neighbor list 완성 즉시 MLP 시작, integration step과 overlap

cudaStream_t stream_nl, stream_mlp;
cudaStreamCreate(&stream_nl);
cudaStreamCreate(&stream_mlp);

// neighbor list 이벤트
cudaEvent_t nl_done;
cudaEventCreate(&nl_done);

// 각 MD step:
build_neighbor_list_kernel<<<grid, block, 0, stream_nl>>>(...);
cudaEventRecord(nl_done, stream_nl);

// MLP stream이 nl_done을 기다린 후 추론 시작
cudaStreamWaitEvent(stream_mlp, nl_done, 0);

// LibTorch stream guard로 MLP를 stream_mlp에서 실행
{
    c10::cuda::CUDAStreamGuard guard(
        c10::cuda::getStreamFromExternal(stream_mlp, 0));
    pot.compute(...);
}

// velocity update는 forces가 준비된 후 시작
cudaStreamSynchronize(stream_mlp);
update_velocities_kernel<<<grid, block, 0, stream_nl>>>(...);
```

#### ━━ 자동 반복 루프 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
최적화 우선순위 순서:
  GPU neighbor list (G1) → Persistent GPU Buffers (G2)
  → float32 강제 (M3) → H2D 제거 zero-copy (M5)
  → optimize_for_inference (M2) → CUDA Stream pipeline (G3+M7)
  → Tensor 재생성 제거 (M4) → cuEquivariance/OpenEquivariance (M8/M9)
  → CUDA Graph (M6) → SoA layout (G4) → Verlet skin skip (G5)

LOOP (MACE GPU 먼저 → NequIP GPU):
  1. benchmark_mlp.sh 실행으로 현재 성능 측정
  2. 아직 미적용 전략 중 우선순위 가장 높은 것 선택
  3. cp target.cpp target.cpp.bak
  4. 전략 구현
  5. make cuda 2>&1 | tee /tmp/build.txt
     → 실패: cp target.cpp.bak target.cpp; 다음 전략으로
  6. bash scripts/validate_mlp.sh [model] cuda
     → 실패: cp target.cpp.bak target.cpp; 다음 전략으로
  7. benchmark_mlp.sh 실행, 개선 폭 측정
  8. 이력 출력 (아래 형식)
  9. 목표 미달 → LOOP
```

#### ━━ 이력 출력 형식 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```
════════════════════════════════════════════════════════
[STAGE 6-F ITERATION N | 모델: MACE/NequIP | DEVICE: CUDA]
전략      : (전략 번호 + 내용)
빌드      : SUCCESS / FAIL
검증      : PASS / FAIL  (NaN: 없음, 드리프트: ___%)
CUDA 시간 : _____ s  (이전 _____ s 대비 _____% 개선)
LJ 대비   : _____x
상태      : DONE ✅ / CONTINUE 🔄
다음 전략 : _______________
════════════════════════════════════════════════════════
→ 승인 없이 즉시 다음 반복
```

#### ━━ 벤치마크 스크립트 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

```bash
#!/bin/bash
# scripts/benchmark_mlp.sh
RUNS=3
bench() {
    local label=$1 bin=$2 input=$3 total=0
    for i in $(seq 1 $RUNS); do
        t=$($bin $input 2>&1 | grep "Total execution time" \
            | awk '{print $(NF-1)}')
        total=$(python3 -c "print($total + $t)")
    done
    printf "  %-38s %8.3f s\n" "$label" "$(python3 -c "print($total/$RUNS)")"
}

echo "── CPU ──────────────────────────────────────────"
bench "LJ baseline"         ./md_simulation_cpu  examples/lj_32000/input.yaml
bench "MACE"                ./md_simulation_cpu  examples/mlp_mace/input.yaml
bench "NequIP"              ./md_simulation_cpu  examples/mlp_nequip/input.yaml

echo "── CUDA ─────────────────────────────────────────"
[ -f ./md_simulation_cuda ] && {
bench "LJ baseline"         ./md_simulation_cuda examples/lj_32000/input.yaml
bench "MACE (최적화前)"     ./md_simulation_cuda examples/mlp_mace/input.yaml
bench "NequIP (최적화前)"   ./md_simulation_cuda examples/mlp_nequip/input.yaml
}
```

---

### 완료 조건

```
[MACE]
✅ models/mace/mace_mp_medium.pt 준비 완료
✅ C++ 연동 빌드 성공 (CPU + CUDA)
✅ validate_mlp.sh mace cpu  PASS
✅ validate_mlp.sh mace cuda PASS
✅ GPU 최적화 후 CUDA 성능 최적화前 대비 ≥ 2x 개선

[NequIP]
✅ models/nequip/NequIP-OAM-L_cuda.pt2 준비 완료
✅ C++ 연동 빌드 성공 (CPU + CUDA)
✅ validate_mlp.sh nequip cpu  PASS
✅ validate_mlp.sh nequip cuda PASS
✅ GPU 최적화 후 CUDA 성능 최적화前 대비 ≥ 2x 개선

[벤치마크]
✅ benchmark_mlp.sh 완료
✅ LJ 대비 MLP CUDA 오버헤드 배수 기록 완료
✅ MACE vs NequIP 성능 비교 결과 기록 완료
```

### STAGE 6 최종 결과 기록 형식

```
[STAGE 6 — MLP 연동 및 GPU 최적화 최종 결과]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                          MACE-MP-0 medium     NequIP-OAM-L
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
C++ 연동 (빌드)           ✅/❌               ✅/❌
에너지 드리프트           _____%  ✅/❌       _____%  ✅/❌
CPU 시간                  _____ s             _____ s
CUDA 최적화前             _____ s             _____ s
CUDA 최적화後             _____ s (_____x↑)   _____ s (_____x↑)
적용된 GPU 최적화 전략    G__,G__,M__,M__     G__,G__,M__,M__
LJ 대비 오버헤드 (CUDA)   _____x              _____x
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
추천 모델/설정: _______________
추가 개선 방향: _______________
```

---

## STAGE 7: 종합 벤치마크 및 최종 리포트

### 목적
모든 STAGE 결과를 통합하여 최종 성능 및 정확성 보고서를 작성한다.

### 최종 측정 항목

```bash
# 모든 모드를 순서대로 실행하여 최종 수치 기록

# 1.  LJ         / verlet_cell_list  / CPU + CUDA
# 2.  LJ         / morton_code       / CPU + CUDA
# 3.  NVT        / verlet_cell_list  / CPU + CUDA
# 4.  NPT        / verlet_cell_list  / CPU + CUDA
# 5.  MACE-MP-0  / verlet_cell_list  / CPU + CUDA
# 6.  MACE-MP-0  / morton_code       / CPU + CUDA
# 7.  NequIP-OAM / verlet_cell_list  / CPU + CUDA
# 8.  NequIP-OAM / morton_code       / CPU + CUDA
# ※ MPI 버전은 NP=1,2,4,8로 각각 측정
```

### 최종 리포트 형식

```
╔══════════════════════════════════════════════════════════════╗
║          MD Simulation 최적화 최종 결과 보고서               ║
╠══════════════════════════════════════════════════════════════╣
║ STAGE 1: 에너지 정확성   ✅/❌                               ║
║   TotEng 오차: _____%  PotEng: _____%  KinEng: _____%        ║
╠══════════════════════════════════════════════════════════════╣
║ STAGE 2: Verlet-Cell 성능                                    ║
║   CPU : _____ s  (LAMMPS _____x)  ✅/🔶/🔴                  ║
║   CUDA: _____ s  (LAMMPS _____x)  ✅/🔶/🔴                  ║
╠══════════════════════════════════════════════════════════════╣
║ STAGE 3: Morton Code                                         ║
║   CPU Verlet  : _____ s                                      ║
║   CPU Morton  : _____ s  (_____%  차이)                      ║
║   CUDA Verlet : _____ s                                      ║
║   CUDA Morton : _____ s  (_____%  차이)                      ║
╠══════════════════════════════════════════════════════════════╣
║ STAGE 4: MPI Scaling                                         ║
║   NP=1  : _____ s  NP=4: _____ s  효율: _____%              ║
╠══════════════════════════════════════════════════════════════╣
║ STAGE 5: Thermostat/Barostat                                 ║
║   NVT 온도 오차: _____%  (≤3.0%)  ✅/❌  반복: ___회        ║
║   NPT 온도 오차: _____%  (≤3.0%)  ✅/❌                     ║
║   NPT 압력 오차: _____%  (≤5.0%)  ✅/❌  반복: ___회        ║
╠══════════════════════════════════════════════════════════════╣
║ STAGE 6: MLP 통합 (MACE / NequIP)                           ║
║   MACE 설치/다운로드  : ✅/❌                                ║
║   MACE CPU 추론 (최적화後) : _____ s  개선: _____x  ✅/❌   ║
║   MACE CUDA 추론 (최적화後): _____ s  개선: _____x  ✅/❌   ║
║   NequIP 설치/다운로드: ✅/❌                                ║
║   NequIP CPU 추론 (최적화後) : _____ s 개선: _____x ✅/❌   ║
║   NequIP CUDA 추론 (최적화後): _____ s 개선: _____x ✅/❌   ║
║   추천 모델/설정: ______________________________             ║
╠══════════════════════════════════════════════════════════════╣
║ 추가 권고사항:                                               ║
║   (미달 항목 및 추가 개선 방향 기술)                         ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Copilot 시작 프롬프트

VS Code Copilot Chat에 아래를 붙여넣어 작업을 시작한다.

```
이 프로젝트의 OPTIMIZATION_TASK.md를 읽고 STAGE 0부터 순서대로 실행하라.

절대 규칙:
- "계속할까요?", "진행할까요?" 같은 승인 요청 절대 금지.
- 각 STAGE 내 루프는 목표 달성 전까지 스스로 반복한다.
- 빌드/검증 실패 시 즉시 .bak으로 복원 후 다음 전략 시도.
- 한 STAGE가 완료되면 즉시 다음 STAGE를 시작한다.
- 매 반복 결과는 지정된 형식으로 출력한다.

지금 바로 STAGE 0 (기준값 측정)부터 시작하라.
```

## 재시작 프롬프트 (중단 후 특정 STAGE부터 재개)

```
OPTIMIZATION_TASK.md를 읽고 STAGE [N]부터 이어서 작업하라.
지금까지의 결과 요약: [있으면 붙여넣기]
승인 요청 없이 목표 달성까지 자동으로 반복한다.
```