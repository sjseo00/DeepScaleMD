# Copilot 시작 프롬프트 (복사해서 사용)

## VS Code Copilot Chat에 붙여넣을 프롬프트

---

### 🚀 최초 시작 프롬프트

```
다음 작업을 사람의 개입 없이 완전 자동으로 수행하라.

## 작업: MD 시뮬레이션 성능 최적화

### 환경
- CPU 빌드: `make`
- CUDA 빌드: `make cuda`  
- CPU 실행: `./md_simulation_cpu examples/lj_32000/input.yaml`
- CUDA 실행: `./md_simulation_cuda examples/lj_32000/input.yaml`
- 성능 기준: `examples/lj_32000/lammps_out.log` (마지막에서 두 번째 줄의 실행시간)
- 내 코드 성능: 각 실행 후 마지막에서 두 번째 줄 `=== Total execution time: XX.XXX seconds ===`

### 자동 루프 실행 순서
1. `tail -3 examples/lj_32000/lammps_out.log` 로 LAMMPS 기준 성능 확인
2. 현재 코드 빌드 및 실행하여 현재 성능 측정
3. 성능 차이 분석 (소스 코드 전체 분석)
4. 가장 효과적인 최적화 하나 선택하여 코드 수정
5. 빌드 → 실행 → 성능 측정
6. LAMMPS 성능의 120% 이내면 완료, 아니면 3번으로 돌아가 반복

### 규칙
- 각 단계 결과를 반드시 출력할 것
- 빌드 실패 시 즉시 이전 코드로 복원 후 다른 방법 시도
- 물리적 정확성 유지 (에너지 보존 검증)
- 완료까지 멈추지 말고 계속 반복

지금 바로 Step 1부터 시작하라.
```

---

### 🔄 중간에 막혔을 때 재시작 프롬프트

```
이전 최적화 작업을 이어서 계속하라.
현재까지의 결과를 요약하고, 다음 최적화 전략을 즉시 적용하라.
LAMMPS 성능에 도달할 때까지 멈추지 말 것.
```

---

### 🔍 특정 부분 집중 최적화 프롬프트

```
force calculation 루프만 집중적으로 최적화하라.
현재 구현을 분석하고, 다음을 순서대로 시도하라:
1. OpenMP 병렬화
2. SIMD 벡터화  
3. Newton 3rd law 최적화
4. Neighbor list 개선
각 시도 후 `make && ./md_simulation_cpu examples/lj_32000/input.yaml` 로 성능 측정.
```

