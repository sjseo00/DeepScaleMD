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
trajectory_output_freq:   100         # trajectory 저장 주기 (스텝)
trajectory_format:        xyz         # xyz | lammpsdump
trajectory_include_velocities: false  # 속도 포함 여부
trajectory_include_forces:     false  # 힘 포함 여부
```

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
  mode: mlip_runtime
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

## 12. 자주 발생하는 오류

| 오류 메시지 | 원인 | 해결책 |
|------------|------|--------|
| `Missing required configuration parameters` | 필수 파라미터 누락 | `coordinate_file`, `target_temperature`, `timestep`, `total_steps` 확인 |
| `Cannot open config file` | 파일 경로 오류 | 프로젝트 루트에서 실행, 상대경로 확인 |
| `No edges found in input system` | MLIP graph 미생성 | UMA의 경우 `external_graph_gen=True` 필요 (자동 처리) |
| `Failed to load model` | TorchScript 비호환 | `mlip_runtime` 모드에서 fairchem 모델 사용 불가 → `python` 모드 사용 |
| `Invalid timestep` | `timestep ≤ 0` | 양수 값 설정 |
