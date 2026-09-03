# GPU-Resident Architecture Implementation

## Overview
코드가 GPU-resident 구조로 변경되었습니다. 이제 시뮬레이션 데이터가 GPU에 상주하며, 매 스텝마다 Host↔Device 복사가 제거되었습니다.

## 주요 변경사항

### 1. 초기 1회 업로드 (Initial Upload)
**파일**: `src/force_calculator_cuda.cu`

새로운 함수 `initial_upload_to_device()` 추가:
- 시뮬레이션 시작 시 모든 입자 데이터를 GPU로 한 번만 업로드
- 위치 (x, y, z), 속도 (vx, vy, vz)
- 질량 (mass), 타입 (type), 전하 (charge), global_id
- Force field 파라미터 (LJ epsilon/sigma mixing rules)

```cpp
void ForceCalculatorCUDA::initial_upload_to_device(MDSystem& system);
```

### 2. GPU 상주 시뮬레이션 루프
**파일**: `src/md_simulation.cpp`

#### 매 스텝 수행 작업:
1. **Neighbor list 업데이트** (필요시)
   - Host에서 neighbor list 빌드
   - `update_neighbor_list_on_device()` 호출하여 GPU로 업로드

2. **Integration on GPU**
   - `integrate_on_device()`: 위치 업데이트 및 반보 속도 업데이트
   - Velocity Verlet 또는 Leap-frog 지원
   - **Integrator 타입은 input 파일에서 설정** (하드코딩 안됨)

3. **Force calculation on GPU**
   - `compute_forces()`: GPU에서 힘 계산
   - 결과가 GPU에 남음 (Host로 복사 안함)

4. **Complete velocity update on GPU**
   - `complete_velocity_update_on_device()`: 속도 업데이트 완료
   - **Integrator 타입에 따라 자동 선택**

5. **Kinetic energy on GPU**
   - `calculate_kinetic_energy()`: GPU에서 KE 계산
   - 결과만 Host로 다운로드

### 3. 드문 이벤트: Host로 다운로드
**파일**: `src/force_calculator_cuda.cu`

새로운 함수 `download_from_device()` 추가:
- 출력/trajectory 저장 시에만 호출
- 필요한 필드만 다운로드 (x, v, f)

```cpp
void ForceCalculatorCUDA::download_from_device(MDSystem& system);
```

### 4. 제거된 함수
- ~~`copy_to_device()`~~ → `initial_upload_to_device()` + `update_neighbor_list_on_device()`로 대체
- ~~`copy_from_device()`~~ → `download_from_device()`로 대체

## 새로운 API

### ForceCalculatorCUDA 클래스
```cpp
// 초기화 (1회)
void initial_upload_to_device(MDSystem& system);

// Neighbor list 업데이트 (필요시)
void update_neighbor_list_on_device(MDSystem& system);

// GPU 상주 Integration
void integrate_on_device(MDSystem& system, Real dt, const std::string& integrator_type);
void complete_velocity_update_on_device(MDSystem& system, Real dt, const std::string& integrator_type);

// 출력용 다운로드 (드물게)
void download_from_device(MDSystem& system);
```

## Integrator 지원

코드는 **input 파일의 integrator 설정을 존중**합니다:
- `integrator_type == "verlet"` 또는 `"velocity_verlet"`: Velocity Verlet
- `integrator_type == "leap_frog"`: Leap-frog
- `integrator_type == "r_respa"`: r-RESPA (CPU only)

하드코딩 없이 동적으로 선택됩니다.

## 성능 개선

### Before (기존 구조)
```
매 스텝마다:
  Host → Device: x, y, z, vx, vy, vz, fx, fy, fz, neighbor_list
  GPU: Force calculation
  Device → Host: fx, fy, fz, energies
```

### After (GPU-resident 구조)
```
초기 1회:
  Host → Device: 모든 데이터

매 스텝:
  GPU: Integration (position update)
  GPU: Force calculation
  GPU: Integration (velocity update)
  GPU: Kinetic energy

출력시에만:
  Device → Host: x, y, z, vx, vy, vz, fx, fy, fz
```

## 데이터 흐름

```
시작:
  initialize() → initial_upload_to_device()
  
메인 루프:
  1. (필요시) update_neighbor_list_on_device()
  2. integrate_on_device()           [GPU]
  3. compute_forces()                 [GPU]
  4. complete_velocity_update_on_device() [GPU]
  5. calculate_kinetic_energy()      [GPU → energies만 Host]
  
출력시:
  download_from_device() → write_trajectory()
```

## 주의사항

1. **Neighbor list는 여전히 Host에서 빌드**
   - GPU 빌드 구현은 복잡하므로 나중에 추가 가능
   - 빌드 후 GPU로 업로드만 함

2. **Electrostatic forces 제외**
   - 요청대로 전기력은 고려하지 않음
   - 필요시 나중에 추가 가능

3. **MPI 호환성 유지**
   - Halo exchange는 여전히 Host에서 수행
   - GPU 데이터와 동기화 필요시 download/upload

4. **Thermostat/Barostat은 CPU**
   - 이들은 자주 호출되지 않으므로 CPU에서 수행
   - 필요시 GPU 버전 추가 가능

## 컴파일 및 실행

변경사항 없음. 기존대로:
```bash
make clean
make
./md_simulation_cpu input.yaml
```

CUDA 버전이 자동으로 GPU-resident 모드로 실행됩니다.

## 향후 개선 가능 항목

1. GPU에서 neighbor list 빌드
2. GPU에서 thermostat/barostat
3. CUDA-aware MPI for 직접 GPU-GPU 통신
4. Multiple GPU streams for overlap
5. Electrostatic forces on GPU (if needed)
