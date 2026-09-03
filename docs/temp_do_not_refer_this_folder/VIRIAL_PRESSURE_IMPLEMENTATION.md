# Virial 및 Pressure 계산 구현 가이드

## 문제 요약

현재 DeepScaleMD는 **virial을 제대로 계산하지 않아서** barostat이 정확한 압력을 사용하지 못합니다.

## 현재 문제점

1. **잘못된 virial 계산** (`src/barostat.cpp:201`):
```cpp
// ❌ 틀린 방법
local_virial += particles.x[i] * particles.fx[i] + 
               particles.y[i] * particles.fy[i] + 
               particles.z[i] * particles.fz[i];
```

2. **Force calculator가 virial을 반환하지 않음**
   - `ForceCalculatorCUDA::compute_forces()`가 virial을 계산하거나 저장하지 않음

3. **부정확한 pressure**
   - Kinetic pressure만 사용 → virial term 누락

## 올바른 구현 방법

### 1. Virial의 정의

Virial은 **force calculation 중에 pair별로** 계산해야 합니다:

```cpp
// ✅ 올바른 방법 (force kernel 내부에서)
for each pair (i, j):
    r_ij = r_i - r_j          // distance vector
    f_ij = force on i due to j  // force vector
    virial += r_ij · f_ij     // dot product: dx*fx + dy*fy + dz*fz
```

### 2. Pressure 공식

```
P = (N * k_B * T + Virial) / (3 * V)
```

여기서:
- `N * k_B * T`: kinetic contribution (ideal gas term)
- `Virial`: interaction contribution
- `V`: volume

### 3. 필요한 수정 사항

#### A. `ForceCalculatorCUDA` 수정

**include/force_calculator_cuda.h**:
```cpp
class ForceCalculatorCUDA : public ForceCalculator {
    // ...
    Real* d_virial;  // Device memory for virial
    
    // Getter
    Real get_virial() const { return last_virial; }
};
```

#### B. LJ Force 커널 수정

**src/force_calculator_cuda.cu**:
```cpp
__global__ void compute_lj_forces_kernel(
    Real* x, Real* y, Real* z, 
    Real* fx, Real* fy, Real* fz,
    // ... other params ...
    Real* potential_energy,
    Real* virial,  // NEW: virial output
    // ...
) {
    Real virial_acc = 0.0;  // Per-thread virial accumulator
    
    // Inside pair loop:
    for (int n = 0; n < n_count; ++n) {
        // ... distance calculation ...
        Real dx = xi - x[j];
        Real dy = yi - y[j];
        Real dz = zi - z[j];
        
        // ... force calculation ...
        Real force_mag = 24.0 * eps * r2inv * (2.0 * sig12_r12 - sig6_r6);
        
        // Force accumulation
        Real fx_pair = force_mag * dx;
        Real fy_pair = force_mag * dy;
        Real fz_pair = force_mag * dz;
        
        fx_acc += fx_pair;
        fy_acc += fy_pair;
        fz_acc += fz_pair;
        
        // ✅ Virial accumulation (r · F)
        virial_acc += dx * fx_pair + dy * fy_pair + dz * fz_pair;
        // Note: 0.5 factor already included in force_mag if needed
    }
    
    // Warp reduction for virial (like PE)
    for (int offset = 16; offset > 0; offset /= 2) {
        virial_acc += __shfl_down_sync(0xffffffff, virial_acc, offset);
    }
    
    if ((threadIdx.x & 31) == 0) {
        if (virial_acc != 0.0) {
            atomicAdd(virial, virial_acc);
        }
    }
}
```

#### C. Barostat에서 virial 사용

**src/barostat.cpp**:
```cpp
void AndersenBarostat::apply(ParticleData& particles, SimulationBox& box, 
                             Real dt, size_t local_particles) {
    // Get virial from force calculator (need to pass it as parameter or via system)
    Real global_virial = ...; // From ForceCalculator::get_virial()
    
    // Kinetic energy
    Real global_kinetic_energy = ...;
    Real volume = box.lx * box.ly * box.lz;
    
    // ✅ Correct pressure calculation
    Real kinetic_pressure = 2.0 * global_kinetic_energy / (3.0 * volume);
    Real virial_pressure = global_virial / (3.0 * volume);
    Real pressure = kinetic_pressure + virial_pressure;
    
    // Apply volume scaling
    Real pressure_error = pressure - target_pressure;
    // ...
}
```

### 4. apply_device()에 ForceCalculator 전달

Barostat의 `apply_device()`가 virial에 접근하려면:

**Option 1**: Virial을 인자로 전달
```cpp
virtual void apply_device(Real* d_x, Real* d_y, Real* d_z,
                         Real* d_vx, Real* d_vy, Real* d_vz,
                         Real* d_mass, Real* d_virial,  // NEW
                         SimulationBox& box, Real dt,
                         size_t local_particles,
                         cudaStream_t stream = 0);
```

**Option 2**: ForceCalculator를 참조로 전달
```cpp
void apply(ParticleData& particles, SimulationBox& box, Real dt,
          size_t local_particles, ForceCalculator* force_calc);
```

### 5. MPI 환경 고려

Virial도 MPI reduction 필요:
```cpp
#ifdef USE_MPI
Real local_virial = ...; // from device or host
Real global_virial = 0.0;
MPI_Allreduce(&local_virial, &global_virial, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
```

## 구현 우선순위

1. ✅ **`ForceCalculator`에 virial 저장 공간 추가** (완료)
2. ⏳ **LJ force 커널에서 virial 계산 추가**
3. ⏳ **Electrostatic force 커널에도 virial 추가**
4. ⏳ **`Barostat::apply()`가 virial 사용하도록 수정**
5. ⏳ **`apply_device()`에도 virial 전달 경로 추가**

## 테스트 방법

1. **Known system 테스트**: 
   - Lennard-Jones fluid의 이론적 pressure와 비교
   - 예: NIST reference data

2. **Energy conservation**:
   - NVE ensemble에서 total energy 변화 확인

3. **NPT ensemble**:
   - 목표 압력에 수렴하는지 확인
   - Volume 진동이 합리적인지 확인

## 참고 자료

- LAMMPS virial 계산: [lammps.sandia.gov/doc/compute_pressure.html](https://docs.lammps.org/compute_pressure.html)
- Virial theorem: Allen & Tildesley, "Computer Simulation of Liquids", Chapter 2
- Pressure calculation: Frenkel & Smit, "Understanding Molecular Simulation", Chapter 4

## 다음 단계

현재 이 가이드는 **설계 문서**입니다. 실제 코드 수정을 원하시면:

1. 위의 수정 사항을 단계별로 구현
2. 각 단계마다 컴파일 및 단위 테스트
3. 최종적으로 NPT 시뮬레이션으로 검증
