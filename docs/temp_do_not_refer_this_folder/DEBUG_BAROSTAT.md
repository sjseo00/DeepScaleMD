# Parrinello-Rahman Barostat 디버깅 가이드

> **목표**: NPT 앙상블에서 Parrinello-Rahman barostat가 동작하지 않는 문제를 단계별로 해결한다.
> 코드가 멈추지 않고 **디버깅 메시지를 출력하며 계속 수정**되도록 방어적으로 작성한다.

---

## 빌드 및 실행 참조

| 항목 | 명령 |
|------|------|
| CPU 빌드 | `make` |
| CUDA 빌드 | `make cuda` |
| CPU 실행 | `./md_simulation_cpu examples/lj_32000/input.yaml` |
| CUDA 실행 | `./md_simulation_cuda examples/lj_32000/input.yaml` |
| 성능 로그 | `examples/lj_32000/lammps_out.log` |
| 성능 측정 | 실행 로그 마지막에서 두 번째 줄: `=== Total execution time: XX.XXX seconds ===` |

---

## 테스트 매트릭스

아래 4가지 실행 환경 모두에서 Parrinello-Rahman barostat가 정상 동작해야 한다.

| # | 환경 | 실행 명령 예시 |
|---|------|---------------|
| 1 | 단일 CPU | `./md_simulation_cpu examples/lj_32000/input.yaml` |
| 2 | MPI | `mpirun -np 4 ./md_simulation_cpu examples/lj_32000/input.yaml` |
| 3 | CUDA | `./md_simulation_cuda examples/lj_32000/input.yaml` |
| 4 | MPI + CUDA | `mpirun -np 4 ./md_simulation_cuda examples/lj_32000/input.yaml` |

---

## 1단계 — 문제 재현 및 초기 진단

### 1-1. 재현 스크립트

```bash
#!/bin/bash
# reproduce_bug.sh
# Parrinello-Rahman barostat 버그 재현 및 초기 진단

set -euo pipefail
LOG="debug_pr_barostat.log"
echo "===== [STEP 1] 재현 시작: $(date) =====" | tee -a "$LOG"

# --- CPU 단독 실행 ---
echo "[DEBUG] 환경 1: 단일 CPU 실행" | tee -a "$LOG"
if ./md_simulation_cpu examples/lj_32000/input.yaml 2>&1 | tee -a "$LOG"; then
    echo "[INFO] 단일 CPU: 정상 완료" | tee -a "$LOG"
else
    echo "[WARN] 단일 CPU: 비정상 종료 (exit=$?)" | tee -a "$LOG"
fi

# --- MPI 실행 ---
echo "[DEBUG] 환경 2: MPI 실행 (np=4)" | tee -a "$LOG"
if mpirun -np 4 ./md_simulation_cpu examples/lj_32000/input.yaml 2>&1 | tee -a "$LOG"; then
    echo "[INFO] MPI: 정상 완료" | tee -a "$LOG"
else
    echo "[WARN] MPI: 비정상 종료 (exit=$?)" | tee -a "$LOG"
fi

# --- CUDA 실행 ---
echo "[DEBUG] 환경 3: CUDA 실행" | tee -a "$LOG"
if ./md_simulation_cuda examples/lj_32000/input.yaml 2>&1 | tee -a "$LOG"; then
    echo "[INFO] CUDA: 정상 완료" | tee -a "$LOG"
else
    echo "[WARN] CUDA: 비정상 종료 (exit=$?)" | tee -a "$LOG"
fi

# --- MPI + CUDA 실행 ---
echo "[DEBUG] 환경 4: MPI+CUDA 실행 (np=4)" | tee -a "$LOG"
if mpirun -np 4 ./md_simulation_cuda examples/lj_32000/input.yaml 2>&1 | tee -a "$LOG"; then
    echo "[INFO] MPI+CUDA: 정상 완료" | tee -a "$LOG"
else
    echo "[WARN] MPI+CUDA: 비정상 종료 (exit=$?)" | tee -a "$LOG"
fi

echo "===== [STEP 1] 재현 완료 =====" | tee -a "$LOG"
```

### 1-2. 진단 체크리스트

실행 후 로그에서 아래 항목을 확인한다.

```
[진단 1] "barostat" 또는 "pressure" 관련 경고/에러 라인 추출
  grep -iE "barostat|pressure|parrinello|rahman|npt|volume" debug_pr_barostat.log

[진단 2] NaN / Inf 발생 여부 확인
  grep -iE "nan|inf|overflow|diverge" debug_pr_barostat.log

[진단 3] 비정상 종료 코드 확인
  grep "비정상 종료" debug_pr_barostat.log
```

---

## 2단계 — Parrinello-Rahman 알고리즘 구현 검증

### 2-1. 핵심 수식 체크

Parrinello-Rahman barostat의 핵심 방정식은 다음과 같다.

```
박스 행렬 h의 운동 방정식:
  d²h/dt² = V · W⁻¹ · (P_int - P_ext) · (hᵀ)⁻¹

입자 운동 방정식 (바로스탯 보정항):
  m·r̈ = F - m·Ġ·G⁻¹·ṙ

여기서
  G   = hᵀ · h          (메트릭 텐서)
  W   = τ_p² · P_ext    (바로스탯 질량 파라미터)
  τ_p = pressure coupling time constant
```

### 2-2. 구현 검증 포인트별 디버그 출력 코드

아래 C++ 스니펫을 Parrinello-Rahman 업데이트 함수 내부에 삽입한다.

```cpp
// === [DEBUG PR-1] 입력 파라미터 검증 ===
void ParrinelloRahmanBarostat::update(
    double dt, double volume,
    const Matrix3& h, const Matrix3& h_dot,
    const Matrix3& pressure_tensor, double target_pressure,
    double tau_p)
{
    // --- 입력값 유효성 검사 ---
    if (std::isnan(volume) || volume <= 0.0) {
        fprintf(stderr, "[DEBUG PR-1] FATAL: volume 비정상 = %e\n", volume);
        fprintf(stderr, "  → volume이 0 이하이면 barostat 계산 불가. NVT로 폴백.\n");
        is_active_ = false;   // 비정상 종료 대신 barostat 비활성화 후 계속 진행
        return;
    }

    // --- W (바로스탯 질량) 계산 검증 ---
    double W = tau_p * tau_p * target_pressure;
    if (W <= 0.0 || std::isnan(W)) {
        fprintf(stderr, "[DEBUG PR-2] FATAL: W(barostat mass) 비정상 = %e\n", W);
        fprintf(stderr, "  tau_p=%e, P_target=%e\n", tau_p, target_pressure);
        fprintf(stderr, "  → tau_p > 0, P_target > 0 인지 input.yaml 확인 필요\n");
        is_active_ = false;
        return;
    }

    // --- 압력 텐서 검증 ---
    double P_inst = (pressure_tensor[0][0]
                   + pressure_tensor[1][1]
                   + pressure_tensor[2][2]) / 3.0;
    if (std::isnan(P_inst)) {
        fprintf(stderr, "[DEBUG PR-3] FATAL: 순간 압력 NaN 검출\n");
        fprintf(stderr, "  pressure_tensor diagonal = (%e, %e, %e)\n",
            pressure_tensor[0][0],
            pressure_tensor[1][1],
            pressure_tensor[2][2]);
        fprintf(stderr, "  → 이전 스텝의 virial 계산 오류일 가능성 있음\n");
        is_active_ = false;
        return;
    }

    // --- h 행렬 역행렬 계산 ---
    Matrix3 h_inv;
    double det = h.determinant();
    if (std::abs(det) < 1e-30) {
        fprintf(stderr, "[DEBUG PR-4] FATAL: h 행렬 행렬식 = %e (거의 0)\n", det);
        fprintf(stderr, "  → 박스가 붕괴되었을 가능성. 이전 스텝에서 h_dot 발산 여부 확인\n");
        is_active_ = false;
        return;
    }
    h_inv = h.inverse();

    // --- h_dot (박스 속도) 업데이트 ---
    // d(h_dot)/dt = (V/W) * (P_int - P_ext*I) * h_inv_T
    Matrix3 delta_P = pressure_tensor;
    for (int i = 0; i < 3; ++i)
        delta_P[i][i] -= target_pressure;

    Matrix3 h_dot_dot = (volume / W) * delta_P * h_inv.transpose();

    // --- 발산 감지 ---
    double h_dot_dot_norm = h_dot_dot.frobenius_norm();
    if (std::isnan(h_dot_dot_norm) || h_dot_dot_norm > 1e10) {
        fprintf(stderr, "[DEBUG PR-5] WARN: h_dot_dot 발산 감지 = %e\n", h_dot_dot_norm);
        fprintf(stderr, "  → dt=%e 가 너무 크거나 tau_p=%e 가 너무 작음\n", dt, tau_p);
        fprintf(stderr, "  → tau_p를 현재의 2배로 임시 증가하여 재시도\n");
        tau_p *= 2.0;
        W = tau_p * tau_p * target_pressure;
        h_dot_dot = (volume / W) * delta_P * h_inv.transpose();
        fprintf(stderr, "[DEBUG PR-5] 조정 후 tau_p=%e, h_dot_dot_norm=%e\n",
                tau_p, h_dot_dot.frobenius_norm());
    }

    // --- Velocity Verlet half-step: h_dot 업데이트 ---
    h_dot_ = h_dot_ + 0.5 * dt * h_dot_dot;

    // --- h 업데이트 ---
    h_ = h_ + dt * h_dot_ + 0.5 * dt * dt * h_dot_dot;

    // --- 주기적 로깅 (100스텝마다) ---
    if (step_ % 100 == 0) {
        fprintf(stdout, "[DEBUG PR-LOG] step=%d V=%.4e P_inst=%.4e P_tgt=%.4e "
                        "|h_dot|=%.4e\n",
                step_, volume, P_inst, target_pressure,
                h_dot_.frobenius_norm());
        fflush(stdout);
    }
    ++step_;
}
```

---

## 3단계 — Virial (바이리얼) 계산 검증

압력 계산의 근원인 virial이 잘못되면 barostat 전체가 오동작한다.

```cpp
// === [DEBUG VIRIAL] LJ 쌍 virial 계산 검증 ===
void compute_virial_debug(
    const std::vector<Vec3>& positions,
    const std::vector<Vec3>& forces,
    double box_length, double cutoff,
    Matrix3& virial_out)
{
    virial_out.zero();
    double virial_trace = 0.0;
    int pair_count = 0;
    int nan_pair_count = 0;

    for (size_t i = 0; i < positions.size(); ++i) {
        for (size_t j = i + 1; j < positions.size(); ++j) {
            Vec3 dr = positions[j] - positions[i];

            // 최소 이미지 규약 적용
            dr[0] -= box_length * std::round(dr[0] / box_length);
            dr[1] -= box_length * std::round(dr[1] / box_length);
            dr[2] -= box_length * std::round(dr[2] / box_length);

            double r2 = dr.dot(dr);
            if (r2 > cutoff * cutoff || r2 < 1e-20) continue;

            double r2_inv = 1.0 / r2;
            double r6_inv = r2_inv * r2_inv * r2_inv;

            // LJ virial: w_ij = r · dU/dr = 48*eps*(r6_inv^2 - 0.5*r6_inv)
            double w = 48.0 * r6_inv * (r6_inv - 0.5);

            if (std::isnan(w) || std::isinf(w)) {
                ++nan_pair_count;
                if (nan_pair_count <= 5) {
                    fprintf(stderr,
                        "[DEBUG VIRIAL] NaN/Inf: pair(%zu,%zu) r2=%e r6_inv=%e\n",
                        i, j, r2, r6_inv);
                }
                continue;   // 해당 쌍만 건너뛰고 계속 진행
            }

            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    virial_out[a][b] += w * dr[a] * dr[b] * r2_inv;

            virial_trace += w;
            ++pair_count;
        }
    }

    if (nan_pair_count > 0) {
        fprintf(stderr,
            "[DEBUG VIRIAL] WARN: %d 쌍에서 NaN/Inf 발생 (총 %d 쌍 계산됨)\n",
            nan_pair_count, pair_count);
        fprintf(stderr, "  → 원인 후보: r < sigma (입자 겹침), cutoff 너무 작음\n");
    }

    fprintf(stdout, "[DEBUG VIRIAL] virial_trace=%.6e pair_count=%d\n",
            virial_trace, pair_count);
}
```

---

## 4단계 — input.yaml 파라미터 검증

Parrinello-Rahman barostat가 올바르게 동작하려면 input.yaml의 파라미터가 물리적으로 합리적이어야 한다.

```yaml
# examples/lj_32000/input.yaml — Parrinello-Rahman NPT 설정 예시

ensemble: npt

# Lennard-Jones 파라미터 (무차원 단위)
epsilon: 1.0
sigma:   1.0
mass:    1.0
cutoff:  2.5       # 최소 2.0*sigma 권장

# 시간 적분
timestep: 0.002    # LJ 단위. 0.005 이상이면 barostat 불안정 위험

# 온도 제어 (Nosé-Hoover)
thermostat:
  type:   nose-hoover
  target: 1.0      # 목표 온도 (LJ 단위)
  tau_t:  0.1      # 열조절기 시간 상수

# 압력 제어 (Parrinello-Rahman)
barostat:
  type:        parrinello-rahman
  target:      1.0      # 목표 압력 (LJ 단위)
  tau_p:       1.0      # 압력 커플링 시간 상수 — 너무 작으면 발산!
  compressibility: 4.5e-5  # 등온 압축률 (필요한 경우)

# 시뮬레이션 규모
n_atoms:    32000
box_length: 32.0   # sigma 단위
n_steps:    50000
output_freq: 500
```

### 파라미터 유효성 검사 코드

```cpp
// === [DEBUG PARAM] input.yaml 로드 후 즉시 실행 ===
void validate_npt_params(const SimParams& p) {
    bool ok = true;

    fprintf(stdout, "[DEBUG PARAM] === NPT 파라미터 검증 시작 ===\n");

    if (p.timestep <= 0 || p.timestep > 0.005) {
        fprintf(stderr,
            "[DEBUG PARAM] WARN: timestep=%e — NPT에서는 0.001~0.002 권장\n",
            p.timestep);
    }

    if (p.barostat.tau_p <= 0) {
        fprintf(stderr,
            "[DEBUG PARAM] FATAL: tau_p=%e — 양수여야 함\n", p.barostat.tau_p);
        ok = false;
    } else if (p.barostat.tau_p < 10.0 * p.timestep) {
        fprintf(stderr,
            "[DEBUG PARAM] WARN: tau_p=%e 가 너무 작음 (timestep의 %g배)\n"
            "  → 최소 tau_p >= 10*dt = %e 권장\n",
            p.barostat.tau_p,
            p.barostat.tau_p / p.timestep,
            10.0 * p.timestep);
    }

    if (p.barostat.target_pressure <= 0) {
        fprintf(stderr,
            "[DEBUG PARAM] FATAL: target_pressure=%e — 양수여야 함\n",
            p.barostat.target_pressure);
        ok = false;
    }

    if (p.cutoff < 2.0 * p.sigma) {
        fprintf(stderr,
            "[DEBUG PARAM] WARN: cutoff=%e < 2*sigma=%e\n"
            "  → virial 계산 부정확으로 압력 오차 증가\n",
            p.cutoff, 2.0 * p.sigma);
    }

    if (!ok) {
        fprintf(stderr,
            "[DEBUG PARAM] 치명적 파라미터 오류. barostat를 비활성화하고 NVT로 진행합니다.\n");
        // 프로그램을 중단하지 않고 NVT 폴백
        p.ensemble = "nvt";
    } else {
        fprintf(stdout, "[DEBUG PARAM] 파라미터 검증 통과\n");
    }
}
```

---

## 5단계 — MPI 환경 동기화 검증

MPI 환경에서는 박스 행렬 `h`와 `h_dot`이 모든 랭크에서 동일해야 한다.

```cpp
// === [DEBUG MPI] MPI 랭크 간 barostat 상태 동기화 확인 ===
#ifdef USE_MPI
#include <mpi.h>
void sync_and_verify_barostat_mpi(Matrix3& h, Matrix3& h_dot, int step) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // rank 0의 h, h_dot을 모든 랭크로 브로드캐스트
    MPI_Bcast(h.data(), 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_dot.data(), 9, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 동기화 후 검증 (100스텝마다)
    if (step % 100 == 0 && rank == 0) {
        fprintf(stdout,
            "[DEBUG MPI] step=%d: h 및 h_dot 브로드캐스트 완료 (ranks=%d)\n",
            step, size);

        double vol = h.determinant();
        if (vol <= 0) {
            fprintf(stderr,
                "[DEBUG MPI] FATAL: rank 0에서 volume=%e (음수 또는 0)\n", vol);
        } else {
            fprintf(stdout, "[DEBUG MPI] volume=%.6e\n", vol);
        }
        fflush(stdout);
    }
}
#else
// MPI 없는 환경에서는 no-op
void sync_and_verify_barostat_mpi(Matrix3&, Matrix3&, int) {}
#endif
```

---

## 6단계 — CUDA 커널 내 barostat 경계 조건 검증

GPU에서는 박스 스케일링이 커널 내에서 잘못 적용될 수 있다.

```cpp
// === [DEBUG CUDA] GPU 박스 스케일링 커널 ===
__global__ void scale_positions_kernel(
    double* __restrict__ pos,   // [N*3]
    const double* __restrict__ h_new,
    const double* __restrict__ h_old_inv,
    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double rx = pos[3*i+0];
    double ry = pos[3*i+1];
    double rz = pos[3*i+2];

    // 분수 좌표로 변환 후 새 박스 행렬 적용
    // s = h_old_inv * r
    double sx = h_old_inv[0]*rx + h_old_inv[1]*ry + h_old_inv[2]*rz;
    double sy = h_old_inv[3]*rx + h_old_inv[4]*ry + h_old_inv[5]*rz;
    double sz = h_old_inv[6]*rx + h_old_inv[7]*ry + h_old_inv[8]*rz;

    // NaN 감지 (GPU 내)
    if (isnan(sx) || isnan(sy) || isnan(sz)) {
        // GPU에서는 printf 제한적으로 사용 가능
        printf("[DEBUG CUDA kernel] NaN in scaled pos: atom %d "
               "(rx=%e,ry=%e,rz=%e)\n", i, rx, ry, rz);
        // 해당 원자를 원점 근처로 클램핑 (시뮬레이션 계속 진행)
        pos[3*i+0] = 0.0;
        pos[3*i+1] = 0.0;
        pos[3*i+2] = 0.0;
        return;
    }

    // r_new = h_new * s
    pos[3*i+0] = h_new[0]*sx + h_new[1]*sy + h_new[2]*sz;
    pos[3*i+1] = h_new[3]*sx + h_new[4]*sy + h_new[5]*sz;
    pos[3*i+2] = h_new[6]*sx + h_new[7]*sy + h_new[8]*sz;
}

// 커널 호출 래퍼 — CUDA 에러 확인 포함
void scale_positions_gpu(double* d_pos, const double* d_h_new,
                          const double* d_h_old_inv, int N, int step) {
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;
    scale_positions_kernel<<<blocks, threads>>>(d_pos, d_h_new, d_h_old_inv, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,
            "[DEBUG CUDA] step=%d: scale_positions_kernel 에러: %s\n",
            step, cudaGetErrorString(err));
        // 크래시 대신 경고만 출력하고 계속 진행
    }

    cudaDeviceSynchronize();
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr,
            "[DEBUG CUDA] step=%d: DeviceSynchronize 에러: %s\n",
            step, cudaGetErrorString(err));
    }
}
```

---

## 7단계 — 통합 디버그 메인 루프

모든 환경에서 공통으로 사용하는 NPT 시뮬레이션 메인 루프 (디버그 모드).

```cpp
// === md_npt_debug.cpp — NPT 메인 루프 (디버깅 전용) ===

void run_npt_debug(SimState& state, const SimParams& params) {

    // --- 파라미터 사전 검증 (절대 크래시하지 않음) ---
    validate_npt_params(params);

    ParrinelloRahmanBarostat barostat(params.barostat.tau_p,
                                      params.barostat.target_pressure);

    fprintf(stdout, "[DEBUG MAIN] NPT 디버그 모드 시작\n");
    fprintf(stdout, "[DEBUG MAIN] n_steps=%d, dt=%e, tau_p=%e, P_tgt=%e\n",
            params.n_steps, params.timestep,
            params.barostat.tau_p, params.barostat.target_pressure);
    fflush(stdout);

    for (int step = 0; step < params.n_steps; ++step) {

        // 1. 속도 베를레 1/2 스텝
        integrate_velocity_half(state, params.timestep);

        // 2. 위치 업데이트
        integrate_position(state, params.timestep);

        // 3. 주기 경계 조건 적용
        apply_pbc(state);

        // 4. 힘 계산
        compute_forces(state, params);

        // 5. Virial 계산 (디버그 출력 포함)
        Matrix3 virial;
        compute_virial_debug(state.positions, state.forces,
                             state.box_length, params.cutoff, virial);

        // 6. 압력 텐서 계산
        Matrix3 pressure_tensor = compute_pressure_tensor(
            state.velocities, state.masses, virial, state.volume());

        // 7. Parrinello-Rahman barostat 업데이트
        if (barostat.is_active()) {
            barostat.update(params.timestep, state.volume(),
                            state.h, state.h_dot,
                            pressure_tensor,
                            params.barostat.target_pressure,
                            params.barostat.tau_p);

            // MPI 동기화
            sync_and_verify_barostat_mpi(state.h, state.h_dot, step);

            // GPU 위치 스케일링
            scale_positions_gpu(state.d_pos, state.h.data(),
                                 state.h_old_inv.data(),
                                 state.n_atoms, step);
        } else {
            // barostat 비활성화됨 — 경고 출력 후 NVT로 계속
            if (step % 1000 == 0) {
                fprintf(stderr,
                    "[DEBUG MAIN] step=%d: barostat 비활성 (NVT 모드로 진행 중)\n",
                    step);
            }
        }

        // 8. Nosé-Hoover 서모스탯 업데이트
        thermostat_nose_hoover(state, params);

        // 9. 속도 베를레 2/2 스텝
        integrate_velocity_half(state, params.timestep);

        // 10. 주기적 상태 출력
        if (step % params.output_freq == 0) {
            double P_inst = (pressure_tensor[0][0]
                           + pressure_tensor[1][1]
                           + pressure_tensor[2][2]) / 3.0;
            double T_inst = compute_temperature(state);
            fprintf(stdout,
                "[THERMO] step=%7d T=%.4f P=%.4f V=%.4f "
                "barostat_active=%d\n",
                step, T_inst, P_inst, state.volume(),
                (int)barostat.is_active());
            fflush(stdout);
        }
    }

    fprintf(stdout, "[DEBUG MAIN] NPT 루프 완료\n");
}
```

---

## 8단계 — 자동 디버깅 실행 스크립트

4가지 환경 모두 테스트하고 결과를 요약한다.

```bash
#!/bin/bash
# run_all_debug.sh
# Parrinello-Rahman barostat 전체 환경 디버깅 자동화

LOG_DIR="debug_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

PASS=0; FAIL=0

run_test() {
    local NAME="$1"
    local CMD="$2"
    local LOG="$LOG_DIR/${NAME}.log"

    echo "========================================"
    echo "[TEST] $NAME"
    echo "  CMD: $CMD"
    echo "========================================"

    eval "$CMD" > "$LOG" 2>&1
    local EXIT_CODE=$?

    # barostat 활성 여부 확인
    local BAROSTAT_ACTIVE
    BAROSTAT_ACTIVE=$(grep -c "barostat_active=1" "$LOG" || true)

    # 발산 여부 확인
    local DIVERGED
    DIVERGED=$(grep -cE "FATAL|NaN|Inf|발산" "$LOG" || true)

    echo "  종료 코드  : $EXIT_CODE"
    echo "  barostat 활성 스텝 수 : $BAROSTAT_ACTIVE"
    echo "  경고/오류 수            : $DIVERGED"

    if [ "$EXIT_CODE" -eq 0 ] && [ "$BAROSTAT_ACTIVE" -gt 0 ]; then
        echo "  결과: ✅ PASS (barostat 정상 동작)"
        PASS=$((PASS + 1))
    else
        echo "  결과: ❌ FAIL"
        echo "  --- 마지막 20줄 로그 ---"
        tail -20 "$LOG"
        FAIL=$((FAIL + 1))
    fi
    echo ""
}

# ---- 빌드 ----
echo "[BUILD] CPU 빌드..."
make > "$LOG_DIR/build_cpu.log" 2>&1 || echo "[WARN] CPU 빌드 실패"

echo "[BUILD] CUDA 빌드..."
make cuda > "$LOG_DIR/build_cuda.log" 2>&1 || echo "[WARN] CUDA 빌드 실패"

# ---- 테스트 실행 ----
run_test "01_single_cpu" \
    "./md_simulation_cpu examples/lj_32000/input.yaml"

run_test "02_mpi_4proc" \
    "mpirun -np 4 ./md_simulation_cpu examples/lj_32000/input.yaml"

run_test "03_cuda" \
    "./md_simulation_cuda examples/lj_32000/input.yaml"

run_test "04_mpi_cuda" \
    "mpirun -np 4 ./md_simulation_cuda examples/lj_32000/input.yaml"

# ---- 요약 ----
echo "========================================"
echo "[SUMMARY] 테스트 결과"
echo "  PASS: $PASS / $((PASS + FAIL))"
echo "  FAIL: $FAIL / $((PASS + FAIL))"
echo "  로그 디렉터리: $LOG_DIR"
echo "========================================"

# 성능 측정 (모든 테스트 통과 시)
if [ "$FAIL" -eq 0 ]; then
    echo "[PERF] 성능 측정..."
    for LOG in "$LOG_DIR"/*.log; do
        PERF=$(grep "Total execution time" "$LOG" || true)
        [ -n "$PERF" ] && echo "  $(basename "$LOG"): $PERF"
    done
fi
```

---

## 9단계 — 흔한 실패 원인 및 빠른 수정

| 증상 | 원인 | 수정 방법 |
|------|------|-----------|
| `volume = 0 또는 음수` | 박스 행렬 det 계산 오류 | `h` 초기화 코드 확인; `h = diag(L)` 로 명시적 초기화 |
| `P_inst = NaN` | virial 계산 시 r=0 division | pair loop에 `if (r2 < 1e-20) continue` 가드 추가 |
| `tau_p 너무 작음` | barostat 발산 | `tau_p >= 10*dt` 보장 (예: dt=0.002이면 tau_p>=0.02) |
| MPI에서만 실패 | h/h_dot 비동기화 | 매 스텝 `MPI_Bcast(h, ...)` 추가 |
| CUDA에서만 실패 | GPU 박스 스케일링 미적용 | `scale_positions_kernel` 호출 여부 확인 |
| MPI+CUDA에서만 실패 | GPU 데이터와 CPU 상태 불일치 | `cudaMemcpy` 후 `MPI_Bcast` 순서 확인 |
| barostat 무시됨 | `ensemble: nvt` 로 설정됨 | `input.yaml`에서 `ensemble: npt` 확인 |

---

## 참조: 관련 파일 위치

```
프로젝트 루트/
├── src/
│   ├── barostat/
│   │   └── parrinello_rahman.cpp   ← 2단계 디버그 코드 삽입 위치
│   ├── force/
│   │   └── lj_force.cpp            ← 3단계 virial 디버그 삽입 위치
│   ├── integrator/
│   │   └── velocity_verlet.cpp
│   └── md_npt_debug.cpp            ← 7단계 메인 루프
├── cuda/
│   └── scale_positions.cu          ← 6단계 CUDA 커널 삽입 위치
├── examples/lj_32000/
│   ├── input.yaml                  ← 4단계 파라미터 수정
│   └── lammps_out.log
├── reproduce_bug.sh                ← 1단계
└── run_all_debug.sh                ← 8단계
```
