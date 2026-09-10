# Input 스크립트 작성 가이드

## 기본 구조

DeepScaleMD는 YAML 형식의 입력 파일을 사용한다. 파일은 여러 섹션(section)과 키-값 쌍으로 구성된다.

```yaml
# 섹션 헤더: 들여쓰기 없는 줄, 콜론으로 끝남
section_name:
  key: value    # 섹션 내 파라미터 (들여쓰기 필수)

# 최상위 키: 들여쓰기 없음
top_level_key: value
```

**실행 방법:**
```bash
bin/md_simulation_cuda input.yaml
mpirun -n 4 bin/md_simulation_cuda input.yaml
```

---

## 1. 필수 파라미터

이 파라미터들이 없으면 시뮬레이션이 시작되지 않는다.

```yaml
# 입력 파일
coordinate_file:   water.xyz        # 초기 좌표 (XYZ 형식)
topology_file:     water.top        # 토폴로지 (원자 타입, 질량, 전하, FF 파라미터)

# 시뮬레이션 조건
target_temperature: 300.0           # 목표 온도 (K) — 초기 속도 분포에 사용
timestep:           0.5             # 시간 스텝 (fs)
total_steps:        10000           # 총 MD 스텝 수
```

Canonical classical force-field package를 사용할 때는 좌표·topology·parameter
세 경로 대신 manifest 하나를 지정할 수 있다.

```yaml
input_system: examples/dsMD_input_lj/manifest.yaml
target_temperature: 300.0
timestep: 1.0
total_steps: 1000
```

manifest 내부 경로는 manifest 위치를 기준으로 해석된다. 포맷 정의는
[`06_dsMD_input_Format.md`](06_dsMD_input_Format.md)를 참고한다.

---

## 2. 시뮬레이션 박스

```yaml
# 박스 크기 (Angstrom)
box_x: 30.0
box_y: 30.0
box_z: 30.0

# 주기 경계 조건
periodic_x: true
periodic_y: true
periodic_z: true
```

---

## 3. 이웃 목록 (Neighbor List)

```yaml
cutoff_distance: 12.0              # 힘 계산 차단 거리 (Å)

# 이웃 목록 유형
neighbor_list_type: verlet_cell_list   # verlet_cell_list (권장) | morton_code
verlet_skin:        2.0                # Verlet skin 거리 (Å) — cutoff에 추가
neighbor_list_update_freq: 10          # 최대 재구축 간격 (스텝 수)
                                        # 실제로는 displacement 기준으로 자동 재구축
```

**권장 값:**
- LJ 시스템: `cutoff_distance: 12.0`, `verlet_skin: 2.0`
- MLIP 시스템: `cutoff_distance: 6.0` (모델에 따라 다름), `verlet_skin: 1.0`
- UMA-S: `cutoff_distance: 6.0`

---

## 4. 적분기 (Integrator)

```yaml
integrator: velocity_verlet    # velocity_verlet (권장) | verlet | leap_frog
```

| 값 | 설명 |
|----|------|
| `velocity_verlet` | Velocity-Verlet (2단계, 에너지 보존 우수) |
| `verlet` | 기본 Verlet |
| `leap_frog` | Leap-frog (속도 반스텝 오프셋) |

---

## 5. 온도 제어 (Thermostat)

```yaml
thermostat: none           # none | langevin | nose_hoover | berendsen

# Langevin 설정
thermostat: langevin
tau_t: 0.1                 # 완화 시간 (ps) — thermostat_damping과 동의어

# Nosé-Hoover 설정
thermostat: nose_hoover
tau_t: 0.5
```

| 값 | 설명 |
|----|------|
| `none` | NVE (에너지 보존 앙상블) |
| `langevin` | 랑주뱅 (NVT, stochastic) |
| `nose_hoover` | Nosé-Hoover chain (NVT, deterministic) |
| `berendsen` | Berendsen (NVT, velocity rescaling) |

---

## 6. 압력 제어 (Barostat)

```yaml
barostat: none             # none | berendsen | parrinello_rahman

# Berendsen NPT
barostat: berendsen
target_pressure: 1.0       # 목표 압력 (bar)
tau_p: 1.0                 # 완화 시간 (ps) — barostat_damping과 동의어
```

---

## 7. 출력 설정

```yaml
output_prefix:            my_sim      # 출력 파일 기본 이름
output_freq:              100         # 스크린 출력 주기 (스텝)
output_info:              STEP, T, KE, PE, ETOTAL, P   # 출력할 물리량

# Trajectory
trajectory_output_freq:   100                 # trajectory 저장 주기 (스텝)
trajectory_format:        xyz, dcd            # 콤마로 여러 포맷 동시 지정 가능
                                               # xyz | dump | dcd | bp (레거시: both = xyz,dump)
trajectory_include_velocities: false          # 속도 포함 여부 (xyz/bp만 지원)
trajectory_include_forces:     false          # 힘 포함 여부 (xyz/bp만 지원)
```

**주의:** trajectory 파일은 `output_freq`와 `trajectory_output_freq`를 모두
만족하는 스텝에서만 기록된다 (`step % output_freq == 0 AND step %
trajectory_output_freq == 0`). 둘을 다르게 설정했다면 실제 저장 주기는 둘의
최소공배수에 가까워질 수 있으니, 보통은 `output_freq`를
`trajectory_output_freq`의 배수(또는 동일한 값)로 맞추는 것이 안전하다.

**`output_info` 가능 값:**

| 키워드 | 의미 |
|--------|------|
| `STEP` | 현재 스텝 번호 |
| `T` | 온도 (K) |
| `KE` | 운동 에너지 (kcal/mol) |
| `PE` | 위치 에너지 (kcal/mol) |
| `ETOTAL` | 총 에너지 (kcal/mol) |
| `P` 또는 `PRESS` | 압력 (bar) |
| `FMAX` | 최대 힘 크기 |

**`trajectory_format` 옵션별 특징:**

| 값 | 파일 | 특징 |
|----|------|------|
| `xyz` | `<prefix>_<step>.xyz` (스텝마다 별도 파일) | 텍스트, 사람이 읽기 쉬움 |
| `dump` | `<prefix>_dump.<step>` (스텝마다 별도 파일) | LAMMPS dump 텍스트 포맷 |
| `dcd` | `<prefix>.dcd` (단일 파일, 프레임 append) | CHARMM/NAMD 바이너리, VMD/MDAnalysis에서 바로 열림. 위치만 기록(속도/힘 미지원) |
| `bp` | `<prefix>.bp/` (ADIOS2 BP 디렉터리) | 진짜 병렬 쓰기 — 대규모 노드용, 아래 참고 |

`xyz`/`dump`/`dcd`는 모두 매 출력 스텝마다 전체 랭크의 입자를 rank 0으로
`MPI_Gatherv`한 뒤 `global_id` 순서로 재정렬해서 기록한다 (도메인 분해로 각
랭크가 갖는 원자 집합은 공간 위치에 따라 정해지고 마이그레이션으로 계속
바뀌므로, gather 순서 그대로 저장하면 프레임마다 같은 행이 다른 원자를
가리키게 된다 — 이를 막기 위한 재정렬이다). 즉 이 세 포맷은 **랭크 수가
커질수록 rank 0 쪽 통신·쓰기 비용이 커지는 직렬 병목**을 그대로 갖고 있다.

### `bp` (ADIOS2 BP) — 대규모 노드 병렬 쓰기

`bp`는 gather를 전혀 하지 않는다. 각 랭크가 **자신이 소유한 로컬 입자만**을
독립적인 로컬 블록으로 곧장 쓰고, 그 블록에 `global_id` 배열을 함께 붙여
후처리 단계에서 전역 순서를 재구성할 수 있게 한다. ADIOS2의 BP5 엔진은
랭크별 블록을 aggregator 기반으로 병합하므로, HDF5류 parallel I/O에서
문제가 되는 "매 스텝 collective 메타데이터 연산" 병목이 구조적으로 없다 —
랭크 수가 수백~수천으로 늘어나도 이 방식이 훨씬 잘 스케일링된다.

```yaml
trajectory_format: bp
```

이 포맷을 쓰려면 빌드 시 `-DENABLE_ADIOS2=ON`과 MPI 지원으로
빌드된 ADIOS2가 필요하다 (`docs/BUILD.md` 참고). 이 옵션 없이 빌드된
바이너리에서 `trajectory_format: bp`를 지정하면 설정 파싱 단계에서 바로
에러가 난다.

`<prefix>.bp/` 디렉터리에는 매 스텝 `x, y, z, global_id`(그리고 설정에 따라
`vx,vy,vz`/`fx,fy,fz`), `step`, `time_fs`, `box`가 랭크별 로컬 배열로
저장되고, `natoms_global` attribute로 전체 원자 수가 기록된다. VMD가 직접
읽지는 못하므로, 분석 시에는 ADIOS2 Python 바인딩으로 각 스텝의 랭크별
블록을 읽어 `global_id`로 정렬하는 후처리가 필요하다.

---

## 8. 힘 계산 모드

### 8-1. 내장 힘장 (builtin)

```yaml
force_calculation:
  mode: builtin
```

토폴로지 파일에서 읽은 LJ 파라미터와 전하를 사용한다.

**정전기 계산 (PME):**
```yaml
force_calculation:
  mode: builtin
  coulomb_method: pme          # direct | pme | ewald
  pme_grid_size: [64, 64, 64]
  pme_alpha: 0.25
  pme_order: 4
```

### 8-2. Python MLIP Interface

```yaml
force_calculation:
  mode: python

mlip_config:
  model_type:  uma             # uma | mace | mace_mp | nequip | chgnet | sevennet | orb
  model_path:  path/to/model.pt
  device:      cuda            # cuda | cpu
  dtype:       float32         # float32 | float64
```

**지원 model_type:**

| model_type | 설명 | 모델 파일 필요 |
|-----------|------|---------------|
| `uma`, `uma_s` | fairchem UMA-S/M | ✅ `.pt` |
| `mace`, `mace_mp` | MACE-MP-0 | ❌ (자동 다운로드) or ✅ |
| `mace_off` | MACE-OFF | ❌ (자동 다운로드) |
| `nequip` | NequIP | ✅ deployed `.pth` |
| `chgnet` | CHGNet | ❌ (자동 다운로드) |
| `sevennet`, `7net` | SevenNet | ❌ (자동 다운로드) |
| `orb` | ORB | ❌ (자동 다운로드) |

**MACE 사전 학습 모델 크기:**
```yaml
mlip_config:
  model_type:  mace_mp
  model_size:  small           # small | medium | large
  device:      cuda
```

### 8-3. C++ MLIP Runtime (TorchScript 전용)

```yaml
force_calculation:
  mode: libtorch
  spec_file: examples/my_model/model_spec.yaml
```

`model_spec.yaml` 작성에 대한 상세 내용은 `docs/03_MLIP_Integration.md` 참고.

---

## 9. 좌표 파일 형식 (XYZ)

```
96
Water 32 molecules L=9.8627
O  1.232837  1.232837  1.232837
H  0.748241  0.651338  1.818719
H  1.717432  1.814335  1.818719
...
```

- 첫 줄: 원자 수
- 둘째 줄: 주석
- 이후: `원소기호  x  y  z` (Angstrom)

---

## 10. 토폴로지 파일 형식 (.top)

```
# Water topology
[atomtypes]
# name  mass(g/mol)  charge(e)  epsilon(kcal/mol)  sigma(Å)
OW   15.9994  -0.8476  0.1553  3.1536
HW    1.0080   0.4238  0.0000  0.0000

[bonds]
# type1  type2  k(kcal/mol/Å²)  r0(Å)
OW  HW  450.0  0.9572

[angles]
# type1  type2  type3  k(kcal/mol/rad²)  theta0(deg)
HW  OW  HW  55.0  104.52
```

---

## 11. 완전한 예시 파일

### NVE (에너지 보존)

```yaml
coordinate_file: water_32mol.xyz
topology_file:   water_32mol.top
target_temperature: 300.0
timestep:    1.0
total_steps: 10000

box_x: 9.8627
box_y: 9.8627
box_z: 9.8627
periodic_x: true
periodic_y: true
periodic_z: true

output_prefix: nve_water
output_freq:   100
output_info:   STEP, T, KE, PE, ETOTAL

trajectory_output_freq: 100
trajectory_format: xyz

cutoff_distance: 9.0
neighbor_list_type: verlet_cell_list
verlet_skin: 2.0
neighbor_list_update_freq: 10

integrator: velocity_verlet
thermostat: none
barostat:   none

force_calculation:
  mode: builtin
```

### NVT with MACE

```yaml
coordinate_file: water_32mol.xyz
topology_file:   water_32mol.top
target_temperature: 300.0
timestep:    0.5
total_steps: 1000

box_x: 9.8627
box_y: 9.8627
box_z: 9.8627
periodic_x: true
periodic_y: true
periodic_z: true

output_prefix: nvt_mace
output_freq:   10
output_info:   STEP, T, KE, PE

trajectory_output_freq: 50
trajectory_format: xyz

cutoff_distance: 6.0
neighbor_list_type: verlet_cell_list
verlet_skin: 1.0
neighbor_list_update_freq: 5

integrator: velocity_verlet
thermostat: langevin
tau_t: 0.1
barostat: none

force_calculation:
  mode: python

mlip_config:
  model_type: mace_mp
  model_size: small
  device:     cuda
  dtype:      float32
```

### NPT with UMA

```yaml
coordinate_file: water_32mol.xyz
topology_file:   water_32mol.top
target_temperature: 300.0
timestep:    0.5
total_steps: 1000

box_x: 9.8627
box_y: 9.8627
box_z: 9.8627
periodic_x: true
periodic_y: true
periodic_z: true

output_prefix: npt_uma
output_freq:   10
output_info:   STEP, T, KE, PE, P

trajectory_output_freq: 50
trajectory_format: xyz

cutoff_distance: 6.0
neighbor_list_type: verlet_cell_list
verlet_skin: 1.0
neighbor_list_update_freq: 5

integrator: velocity_verlet
thermostat: nose_hoover
tau_t: 0.1
barostat: berendsen
target_pressure: 1.0
tau_p: 1.0

force_calculation:
  mode: python

mlip_config:
  model_type: uma
  model_path: examples/mlp_uma/uma-s-1p1.pt
  device:     cuda
  dtype:      float32
```

---

## 12. 에너지 최소화 (Energy Minimization)

PDB/PSF 등 실험 구조에서 만든 초기 좌표는 원자 간 겹침(steric clash)이나
비물리적으로 늘어난 결합을 포함하기 쉽다. 이 상태로 바로 MD를 시작하면 힘이
발산해 시뮬레이션이 즉시 깨진다. `minimize` 섹션을 추가하면 MD 루프(1 스텝)를
시작하기 전에 [FIRE](https://doi.org/10.1103/PhysRevLett.97.170201)
(Fast Inertial Relaxation Engine) 알고리즘으로 potential energy가 지역 최솟값에
가까워질 때까지 좌표를 이완시킨 뒤 MD를 시작한다.

```yaml
minimize:
  method: fire            # none (기본값) | fire
  max_steps: 500           # 최소화 반복 횟수 상한
  force_tolerance: 5.0     # kcal/mol/Å — 전역 최대 |F_i|가 이 값 아래로 내려오면 수렴
  # 아래는 고급 옵션 (생략 시 timestep 기반으로 자동 설정됨)
  # dt_start: 0.5          # fs, FIRE 의사-동역학의 초기 시간 간격
  # dt_max:   2.0           # fs, 위 값의 상한
  output_freq: 10           # 진행 상황을 출력할 반복 간격
```

**동작 방식:**
- FIRE는 속도(velocity) 배열을 의사-동역학 상태로 재사용한다. 최소화 시작 시
  Maxwell-Boltzmann로 초기화된 열적 속도는 0으로 리셋되고, 최소화가 끝나면
  다시 0으로 리셋된 뒤 MD가 실제 `target_temperature`로 시작한다 — 즉
  `minimize`는 초기 속도 분포에 영향을 주지 않는다.
- 매 반복마다 힘을 재계산하고(이웃 목록/halo도 매번 갱신), 전역 최대 힘
  성분(|F_i|, 모든 랭크에 걸쳐)이 `force_tolerance` 아래로 떨어지면 즉시
  종료한다. `max_steps`에 도달하면 수렴 여부와 무관하게 중단하고 MD를
  이어서 시작한다(경고와 함께 마지막 PE/Fmax를 출력).
- CPU와 CUDA 빌드 양쪽에서 동작하며, CUDA 빌드에서는 좌표·힘·속도가 최소화
  전체 구간 동안 GPU에 상주한 채로 계산된다(스칼라 몇 개만 매 반복 호스트로
  전송됨).
- MPI 다중 랭크에서도 동작한다 — 힘/속도 합과 최대값은 모든 랭크에 걸쳐
  전역 리듀스되므로, 각 랭크는 항상 동일한 수렴/스텝 크기 결정을 내린다.

**단점:** 매 반복마다 이웃 목록을 무조건 재구축하므로(초기 겹침 때문에
displacement 기반 재사용을 안전하게 건너뛸 수 없음) 큰 시스템에서는 최소화
구간이 상대적으로 느릴 수 있다. `max_steps`를 시스템 크기에 맞게 제한하는 것을
권장한다.

---

## 13. 자주 발생하는 오류

| 오류 메시지 | 원인 | 해결책 |
|------------|------|--------|
| `Missing required configuration parameters` | 필수 파라미터 누락 | `coordinate_file`, `target_temperature`, `timestep`, `total_steps` 확인 |
| `Cannot open config file` | 파일 경로 오류 | 프로젝트 루트에서 실행, 상대경로 확인 |
| `No edges found in input system` | MLIP graph 미생성 | UMA의 경우 `external_graph_gen=True` 필요 (자동 처리) |
| `Failed to load model` | TorchScript 비호환 | `libtorch` 모드에서 fairchem 모델 사용 불가 → `python` 모드 사용 |
| `Invalid timestep` | `timestep ≤ 0` | 양수 값 설정 |
