# Parrinello-Rahman Barostat 빠른 참조 가이드

## 🚀 빠른 시작

### 테스트 실행
```bash
cd /home1/kisti/DeepScaleMD
./run_all_debug.sh
```

### 결과 확인
- ✅ PASS: barostat 정상 동작
- ❌ FAIL: 로그 파일 확인 필요

---

## 🔍 디버그 메시지 해석

### [DEBUG PR-1] Volume 비정상
```
[DEBUG PR-1] FATAL: volume 비정상 = 0.0
```
**원인**: 박스 행렬이 잘못 초기화됨  
**해결**: input.yaml의 box_length 확인

### [DEBUG PR-2] W(barostat mass) 비정상
```
[DEBUG PR-2] FATAL: W(barostat mass) 비정상 = -100
```
**원인**: barostat 초기화 오류  
**해결**: 코드에서 W 초기화 확인 (기본값: 10000)

### [DEBUG PR-3] 순간 압력 NaN
```
[DEBUG PR-3] FATAL: 순간 압력 NaN 검출
  virial_pressure = nan
```
**원인**: Virial 계산에서 NaN 발생  
**해결**: DEBUG VIRIAL 메시지 확인, cutoff 증가

### [DEBUG PR-4] h 행렬 행렬식 거의 0
```
[DEBUG PR-4] FATAL: h 행렬 행렬식 = 1.23e-50
```
**원인**: 박스가 붕괴됨  
**해결**: 이전 스텝의 h_dot 발산 확인, W 증가

### [DEBUG PR-5] h_dot_dot 발산
```
[DEBUG PR-5] WARN: h_dot_dot 발산 감지 = 1.234e12
  → dt=0.002 가 너무 크거나 W=100 가 너무 작음
  → Clamping h_dot_dot to prevent divergence
```
**원인**: dt가 너무 크거나 W가 너무 작음  
**해결**: 
- dt 감소: 0.002 → 0.001
- W 증가: 10000 → 50000
- tau_p 증가: 1.0 → 2.0

### [DEBUG VIRIAL] NaN/Inf 검출
```
[DEBUG VIRIAL] NaN/Inf: pair(123,456) r2=1.23e-20 force_mag=inf
```
**원인**: 입자가 너무 가까움 (r ≈ 0)  
**해결**:
- cutoff 증가: 2.5 → 3.0
- 초기 구조 검증 (입자 겹침 제거)

---

## ⚙️ 파라미터 튜닝 빠른 가이드

### 문제: 발산 (NaN/Inf 발생)
**해결**:
```yaml
timestep: 0.001      # 현재의 절반
tau_p:    2.0        # 현재의 2배
```

### 문제: 압력 수렴 느림
**해결**:
```yaml
tau_p: 0.5           # 감소 (단, ≥ 10*dt 유지)
```

### 문제: Volume 급격히 변함
**해결**:
```yaml
barostat:
  mass: 50000.0      # W 증가 (기본 10000)
```

---

## 📊 정상 동작 확인

### 로그에 다음이 포함되어야 함:
```
[DEBUG PARAM] 파라미터 검증 통과
[DEBUG PR-LOG] step=100 V=32780.2 P_inst=1.198 bar P_tgt=1.0 bar
[DEBUG VIRIAL] virial_local=1256.78 kcal/mol
```

### FATAL/WARN 메시지가 없어야 함

---

## 🛠️ 일반적인 수정

### 1. dt가 너무 크다는 경고
**input.yaml**:
```yaml
timestep: 0.001  # 기존 0.002에서 감소
```

### 2. tau_p가 너무 작다는 경고
**input.yaml**:
```yaml
barostat:
  tau_p: 0.02    # 최소 10*dt
```

### 3. Virial NaN 발생
**input.yaml**:
```yaml
cutoff: 3.0      # 기존 2.5에서 증가
```

---

## 📁 로그 파일 위치

### 자동 테스트 후:
```
debug_logs_YYYYMMDD_HHMMSS/
├── 01_single_cpu.log
├── 02_mpi_4proc.log
├── 03_cuda.log
└── 04_mpi_cuda.log
```

### 버그 재현 후:
```
debug_pr_barostat.log
```

---

## 🔧 긴급 수정 (빠른 NVT 폴백)

NPT가 계속 실패하면 일시적으로 NVT로 전환:

**input.yaml**:
```yaml
ensemble: nvt   # npt → nvt 변경
```

---

## 📞 추가 지원

### 상세 문서
- `DEBUG_BAROSTAT.md` — 전체 디버깅 가이드
- `PR_BAROSTAT_DEBUG_FIXES.md` — 수정 사항 요약
- `PARRINELLO_RAHMAN_DEBUG_COMPLETE.md` — 완료 보고서

### 스크립트
- `./reproduce_bug.sh` — 버그 재현 및 진단
- `./run_all_debug.sh` — 전체 환경 자동 테스트

---

**마지막 업데이트**: 2026-03-04
