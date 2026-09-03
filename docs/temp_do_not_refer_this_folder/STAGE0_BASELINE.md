# STAGE 0 — BASELINE RECORD

Date: 2026-02-27

## LAMMPS 기준값 (Step 10000)
```
REF_TIME          = 48.073 s
REF_TOTAL_ENERGY  = 28465.953  (kcal/mol)
REF_POT_ENERGY    = -1762.2899 (kcal/mol)
REF_KIN_ENERGY    = 30228.243  (kcal/mol)
REF_TEMPERATURE   = 316.91456  K
```

## CPU 현재 상태
```
CPU_TIME          = 22.479 s  (LAMMPS 대비 0.47x - 2배 빠름!)
CPU_TOTAL_ENERGY  = 28465.9363 (오차 0.00%)  ✅
CPU_POT_ENERGY    = -1738.1579 (오차 1.37%)  ❌
CPU_KIN_ENERGY    = 30204.0942 (오차 0.08%)  ✅
```

## CUDA 현재 상태 (문제 발견)
```
CUDA_TIME         = 120.232 s  (LAMMPS 대비 2.50x - CPU보다 5배 느림!)
CUDA_TOTAL_ENERGY = 28465.8052 (오차 0.01%)  ✅
CUDA_POT_ENERGY   = -1759.2776 (오차 0.17%)  ✅
CUDA_KIN_ENERGY   = 30225.0827 (오차 0.01%)  ✅
```

**문제 진단:**
1. **CPU**: Potential Energy 오차 1.37% → STAGE 1에서 수정 필요
2. **CUDA**: "Using CPU backend" 메시지가 출력됨 → CUDA가 실제로 사용되지 않음
   - 120초 소요 (CPU 22초보다 느림)
   - Neighbor list 구축 시간이 49.6초로 비정상적으로 김
   - CUDA 코드 경로가 활성화되지 않음

## 다음 단계
- **STAGE 1**: LJ 에너지 정확성 검증 및 수정 (CPU 우선)
- **CUDA 수정**: 빌드 시스템 확인 및 CUDA 경로 활성화
