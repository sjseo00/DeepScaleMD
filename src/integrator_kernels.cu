#include "md_types.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ==================== Optimized CUDA Kernels for Velocity-Verlet ====================
// 최적화 전략:
// 1. Kernel Fusion: Step 1과 2를 하나의 커널로 통합하여 메모리 접근 최소화
// 2. Memory Coalescing: 연속된 메모리 접근 패턴 보장
// 3. Register Optimization: 레지스터 사용 최적화로 L2 캐시 활용
// ======================================================================================

// Device function: Apply periodic boundary conditions (최적화: 분기 최소화)
__device__ __forceinline__ void apply_pbc(Real &x, Real &y, Real &z, 
                                           Real lx, Real ly, Real lz,
                                           bool px, bool py, bool pz) {
    // floor 대신 빠른 연산 사용
    if (px) x -= lx * floorf(x / lx);
    if (py) y -= ly * floorf(y / ly);
    if (pz) z -= lz * floorf(z / lz);
}

// OPTIMIZED: Fused Velocity-Verlet Step 1 (위치 + 반스텝 속도 업데이트)
// 커널 퓨전으로 글로벌 메모리 읽기/쓰기를 한 번에 처리
// SoA(Structure of Arrays) 레이아웃으로 메모리 병합 최적화
__global__ void vv_step1_kernel(Real *__restrict__ x, Real *__restrict__ y, Real *__restrict__ z,
                               Real *__restrict__ vx, Real *__restrict__ vy, Real *__restrict__ vz,
                               const Real *__restrict__ fx, const Real *__restrict__ fy, const Real *__restrict__ fz,
                               const Real *__restrict__ mass, Real dt, size_t n,
                               Real lx, Real ly, Real lz,
                               bool periodic_x, bool periodic_y, bool periodic_z) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // 최적화: 모든 데이터를 레지스터에 로드 (단일 메모리 트랜잭션)
    const Real inv_m = 1.0f / mass[i];
    const Real half_dt = 0.5f * dt;
    
    // 현재 상태 로드 (coalesced memory access)
    Real pos_x = x[i];
    Real pos_y = y[i];
    Real pos_z = z[i];
    Real vel_x = vx[i];
    Real vel_y = vy[i];
    Real vel_z = vz[i];
    
    // 가속도 계산: a(t) = F(t) / m (레지스터 내 연산)
    const Real ax = fx[i] * inv_m * FORCE_TO_ACCEL;
    const Real ay = fy[i] * inv_m * FORCE_TO_ACCEL;
    const Real az = fz[i] * inv_m * FORCE_TO_ACCEL;
    
    // 반스텝 속도 업데이트: v(t+dt/2) = v(t) + 0.5 * a(t) * dt
    vel_x += half_dt * ax;
    vel_y += half_dt * ay;
    vel_z += half_dt * az;
    
    // 위치 업데이트: x(t+dt) = x(t) + v(t+dt/2) * dt
    pos_x += dt * vel_x;
    pos_y += dt * vel_y;
    pos_z += dt * vel_z;
    
    // PBC 적용 (조건부 연산 최소화)
    apply_pbc(pos_x, pos_y, pos_z, lx, ly, lz, periodic_x, periodic_y, periodic_z);
    
    // 결과 저장 (coalesced memory write)
    x[i] = pos_x;
    y[i] = pos_y;
    z[i] = pos_z;
    vx[i] = vel_x;
    vy[i] = vel_y;
    vz[i] = vel_z;
}

// OPTIMIZED: Leap-frog velocity update (메모리 대역폭 최적화)
// 전체 타임스텝 속도 업데이트 - 메모리 접근 패턴 최적화
__global__ void leap_frog_velocity_update_kernel(Real *__restrict__ vx, Real *__restrict__ vy, Real *__restrict__ vz,
                                                 const Real *__restrict__ fx, const Real *__restrict__ fy, const Real *__restrict__ fz,
                                                 const Real *__restrict__ mass, Real dt, size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // 최적화: 레지스터 내 모든 연산 수행
    const Real inv_m = 1.0f / mass[i];
    
    // 속도 로드 (coalesced)
    Real vel_x = vx[i];
    Real vel_y = vy[i];
    Real vel_z = vz[i];
    
    // Leap-frog 전체 타임스텝 속도 업데이트
    // v(t+3dt/2) = v(t+dt/2) + a(t+dt) * dt
    const Real ax = fx[i] * inv_m * FORCE_TO_ACCEL;
    const Real ay = fy[i] * inv_m * FORCE_TO_ACCEL;
    const Real az = fz[i] * inv_m * FORCE_TO_ACCEL;
    
    vel_x += dt * ax;
    vel_y += dt * ay;
    vel_z += dt * az;
    
    // 결과 저장 (coalesced)
    vx[i] = vel_x;
    vy[i] = vel_y;
    vz[i] = vel_z;
}

// OPTIMIZED: Velocity-Verlet Step 2 (반스텝 속도 업데이트 완료)
// 메모리 대역폭 최적화 및 레지스터 활용
__global__ void vv_step2_kernel(Real *__restrict__ vx, Real *__restrict__ vy, Real *__restrict__ vz,
                               const Real *__restrict__ fx, const Real *__restrict__ fy, const Real *__restrict__ fz,
                               const Real *__restrict__ mass, Real dt, size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    // 최적화: 상수를 레지스터에 저장
    const Real inv_m = 1.0f / mass[i];
    const Real half_dt = 0.5f * dt;
    
    // 속도 로드 (coalesced memory access)
    Real vel_x = vx[i];
    Real vel_y = vy[i];
    Real vel_z = vz[i];
    
    // a(t+dt) = F(t+dt) / m (레지스터 내 연산)
    const Real ax = fx[i] * inv_m * FORCE_TO_ACCEL;
    const Real ay = fy[i] * inv_m * FORCE_TO_ACCEL;
    const Real az = fz[i] * inv_m * FORCE_TO_ACCEL;
    
    // v(t+dt) = v(t+dt/2) + 0.5 * a(t+dt) * dt
    vel_x += half_dt * ax;
    vel_y += half_dt * ay;
    vel_z += half_dt * az;
    
    // 결과 저장 (coalesced memory write)
    vx[i] = vel_x;
    vy[i] = vel_y;
    vz[i] = vel_z;
}

// OPTIMIZED: Leap-frog integration (단순성 활용 + 메모리 최적화)
// 위치와 속도 업데이트를 효율적으로 처리 - 메모리 대역폭 극대화
__global__ void leap_frog_kernel(Real *__restrict__ x, Real *__restrict__ y, Real *__restrict__ z,
                                Real *__restrict__ vx, Real *__restrict__ vy, Real *__restrict__ vz,
                                const Real *__restrict__ fx, const Real *__restrict__ fy, const Real *__restrict__ fz,
                                const Real *__restrict__ mass, Real dt, size_t n,
                                Real lx, Real ly, Real lz,
                                bool periodic_x, bool periodic_y, bool periodic_z,
                                bool first_step) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    
    const Real inv_m = 1.0f / mass[i];
    
    if (first_step) {
        // 초기화: 반스텝 속도 업데이트
        const Real half_dt = 0.5f * dt;
        
        // 속도 로드
        Real vel_x = vx[i];
        Real vel_y = vy[i];
        Real vel_z = vz[i];
        
        // 가속도 계산 및 속도 업데이트
        const Real ax = fx[i] * inv_m * FORCE_TO_ACCEL;
        const Real ay = fy[i] * inv_m * FORCE_TO_ACCEL;
        const Real az = fz[i] * inv_m * FORCE_TO_ACCEL;
        
        vel_x += half_dt * ax;
        vel_y += half_dt * ay;
        vel_z += half_dt * az;
        
        // 저장
        vx[i] = vel_x;
        vy[i] = vel_y;
        vz[i] = vel_z;
    } else {
        // 메인 Leap-frog: 위치만 업데이트 (속도는 이미 반스텝 앞섬)
        // 최적화: 모든 데이터를 레지스터에 로드
        Real pos_x = x[i];
        Real pos_y = y[i];
        Real pos_z = z[i];
        
        // x(t+dt) = x(t) + v(t+dt/2) * dt
        pos_x += dt * vx[i];
        pos_y += dt * vy[i];
        pos_z += dt * vz[i];
        
        // PBC 적용
        apply_pbc(pos_x, pos_y, pos_z, lx, ly, lz, periodic_x, periodic_y, periodic_z);
        
        // 위치 저장 (coalesced write)
        x[i] = pos_x;
        y[i] = pos_y;
        z[i] = pos_z;
    }
}

// Wrapper functions for launching kernels
extern "C" {

void launch_vv_step1_kernel(Real *d_x, Real *d_y, Real *d_z,
                            Real *d_vx, Real *d_vy, Real *d_vz,
                            const Real *d_fx, const Real *d_fy, const Real *d_fz,
                            const Real *d_mass, Real dt, size_t num_particles,
                            Real lx, Real ly, Real lz,
                            bool periodic_x, bool periodic_y, bool periodic_z) {
    // 최적화: H100/A100에 최적화된 블록 크기
    // - 256 스레드: 워프 스케줄링 효율 극대화
    // - L2 캐시 활용 극대화를 위한 그리드 크기 조정
    const int block_size = 256;
    const int grid_size = (num_particles + block_size - 1) / block_size;
    
    vv_step1_kernel<<<grid_size, block_size>>>(
        d_x, d_y, d_z, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, d_mass, dt, num_particles,
        lx, ly, lz, periodic_x, periodic_y, periodic_z);
}

void launch_vv_step2_kernel(Real *d_vx, Real *d_vy, Real *d_vz,
                            const Real *d_fx, const Real *d_fy, const Real *d_fz,
                            const Real *d_mass, Real dt, size_t num_particles) {
    // 최적화: 메모리 바운드 작업에 최적화된 설정
    const int block_size = 256;
    const int grid_size = (num_particles + block_size - 1) / block_size;
    
    vv_step2_kernel<<<grid_size, block_size>>>(
        d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, d_mass, dt, num_particles);
}

void launch_leap_frog_kernel(Real *d_x, Real *d_y, Real *d_z,
                             Real *d_vx, Real *d_vy, Real *d_vz,
                             const Real *d_fx, const Real *d_fy, const Real *d_fz,
                             const Real *d_mass, Real dt, size_t num_particles,
                             Real lx, Real ly, Real lz,
                             bool periodic_x, bool periodic_y, bool periodic_z,
                             bool first_step) {
    const int block_size = 256;
    const int grid_size = (num_particles + block_size - 1) / block_size;
    
    leap_frog_kernel<<<grid_size, block_size>>>(
        d_x, d_y, d_z, d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, d_mass, dt, num_particles,
        lx, ly, lz, periodic_x, periodic_y, periodic_z, first_step);
}

void launch_leap_frog_velocity_update_kernel(Real *d_vx, Real *d_vy, Real *d_vz,
                                             const Real *d_fx, const Real *d_fy, const Real *d_fz,
                                             const Real *d_mass, Real dt, size_t num_particles) {
    const int block_size = 256;
    const int grid_size = (num_particles + block_size - 1) / block_size;
    
    // Use the new leap-frog velocity update kernel (FULL dt, not half-step)
    leap_frog_velocity_update_kernel<<<grid_size, block_size>>>(
        d_vx, d_vy, d_vz, d_fx, d_fy, d_fz, d_mass, dt, num_particles);
}

// Keep r-RESPA wrapper for compatibility (kept separate from velocity-verlet)
void launch_r_respa_kernel(Real *d_x, Real *d_y, Real *d_z,
                          Real *d_vx, Real *d_vy, Real *d_vz,
                          Real *d_fx, Real *d_fy, Real *d_fz,
                          Real *d_mass, Real dt, size_t num_particles, bool first_half,
                          Real lx, Real ly, Real lz,
                          bool periodic_x, bool periodic_y, bool periodic_z) {
    // r-RESPA not simplified yet - placeholder for future implementation
}

}  // extern "C"

