# DeepScaleMD MLIP 예제 모음

모든 예제는 프로젝트 루트 `/home1/kisti/DeepScaleMD` 에서 실행합니다.

## 공통 좌표 파일

| 파일 | 내용 | 사용 예제 |
|------|------|---------|
| `bench_32mol.xyz` | 32개 수분자 (96 atoms, 9.86 Å box) | 모든 수계 MLIP 예제 |
| `bench_32mol.top` | 수분자 토폴로지 | 동일 |

---

## 지원 경로 및 MLIP 목록

| 예제 디렉터리 | MLIP | 경로 | 웨이트 준비 | 자동 다운로드 |
|-------------|------|------|------------|------------|
| `mlip_mace/` | MACE-MP-0 | Python | 자동 | ✅ |
| `mlip_mace/` (input_off.yaml) | MACE-OFF23 | Python | 자동 | ✅ |
| `mlip_nequip/` | NequIP | Python | 학습 필요 | ❌ |
| `mlip_sevennet/` | SevenNet-0 | Python | 자동 | ✅ |
| `mlip_chgnet/` | CHGNet v0.3 | Python | 내장 | ✅ |
| `mlip_mattersim/` | MatterSim-1M | Python | HuggingFace | ✅ (인터넷) |
| `mlip_matgl/` | M3GNet | Python | HuggingFace | ✅ (인터넷) |
| `mlip_matgl/` (input_tensornet.yaml) | TensorNet | Python | HuggingFace | ✅ (인터넷) |
| `mlip_deepmd/` | DeePMD-kit | Python | 학습/다운로드 | ⚠️ |
| `mlip_orb/` | ORB-v2 | Python | 자동 | ✅ |
| `mlip_gap/` | GAP (QUIP) | Python | 수동 다운로드 | ⚠️ |
| `mlip_snap/` | SNAP | LAMMPS only | 수동 다운로드 | ⚠️ |
| `cpp_mace/` | MACE TorchScript | **C++ Runtime** | export 스크립트 | ✅ (MACE 필요) |
| `cpp_nequip/` | NequIP TorchScript | **C++ Runtime** | 학습+컴파일 | ❌ |
| `cpp_deepmd/` | DeePMD frozen | **C++ Runtime** | 학습/다운로드 | ⚠️ |

---

## 빠른 시작 (자동 다운로드 가능한 모델)

```bash
cd /home1/kisti/DeepScaleMD

# MACE-MP (가장 범용, 추천)
bash examples/mlip_mace/run.sh

# CHGNet (내장 웨이트, 즉시 실행)
bash examples/mlip_chgnet/run.sh

# SevenNet-0
bash examples/mlip_sevennet/run.sh

# ORB-v2
bash examples/mlip_orb/run.sh

# C++ Runtime: MACE (TorchScript export 포함)
bash examples/cpp_mace/run.sh
```

---

## 웨이트 준비가 필요한 모델

### NequIP (Python + C++ 경로)
```bash
pip install nequip
nequip-train configs/water_training.yaml
nequip-compile results/water/deployed.pth --output examples/mlip_nequip/compiled.nequip.pth
cp examples/mlip_nequip/compiled.nequip.pth examples/cpp_nequip/
```

### DeePMD (Python + C++ 경로)
```bash
pip install deepmd-kit[torch]
# 학습 후:
dp freeze -o examples/mlip_deepmd/frozen_model.pth
cp examples/mlip_deepmd/frozen_model.pth examples/cpp_deepmd/
```

### GAP (Python 경로)
```bash
pip install quippy-ase
# https://libatoms.github.io/GAP/data.html 에서 XML 다운로드
# examples/mlip_gap/gap_water.xml 로 저장
```

### SNAP (LAMMPS 전용)
```bash
git clone https://github.com/FitSNAP/FitSNAP.git /tmp/fitsnap
cp /tmp/fitsnap/examples/Ta_Linear_JCP2014/Ta06A.snap* examples/mlip_snap/
bash examples/mlip_snap/run.sh
```

---

## Python Interface vs C++ MLIP Runtime 비교

| 항목 | Python Interface | C++ MLIP Runtime |
|------|-----------------|-----------------|
| 설정 | `force_calculation.mode: python` | `force_calculation.mode: mlip_runtime` |
| 모델 로딩 | Python 어댑터 (mlip_interface.py) | C++ TorchScript 직접 로드 |
| Python 오버헤드 | 스텝마다 Python 호출 | 없음 (순수 C++/CUDA) |
| 지원 모델 | 모든 Python 패키지 | TorchScript 형식만 |
| 디버깅 | 쉬움 | 어려움 |
| 대규모 시뮬레이션 | 오버헤드 있음 | GPU-resident 최적화 |

---

## 필요 패키지 일괄 설치

```bash
pip install mace-torch sevenn chgnet mattersim matgl orb-models
pip install deepmd-kit[torch]      # DeePMD
pip install quippy-ase             # GAP
pip install nequip                 # NequIP
```
