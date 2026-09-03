# PME Calculator Build Fix Summary

## 문제 분석 (Problem Analysis)

### 발견된 문제 (Issues Found)
1. **빌드 에러**: `make cuda` 실행 시 PMECalculator 관련 함수들이 undefined reference 에러 발생
2. **근본 원인**: Makefile의 오브젝트 파일 생성 규칙 순서 문제
   - CUDA 빌드 시 `pme_calculator.cu`가 아닌 `pme_calculator.cpp`가 먼저 컴파일됨
   - `pme_calculator.cpp`는 `#ifndef USE_CUDA`로 감싸져 있어 CUDA 빌드 시 빈 오브젝트 파일 생성
   - 링커가 PMECalculator 심볼을 찾을 수 없어 에러 발생

### 기술적 세부사항 (Technical Details)
```
컴파일 순서 문제:
obj/pme_calculator.o 생성 시:
- src/pme_calculator.cpp (있음) → .cpp 규칙 매칭 (먼저 정의됨)
- src/pme_calculator.cu (있음)  → .cu 규칙 매칭 (나중에 정의됨)

Make는 첫 번째 매칭되는 규칙을 선택 → .cpp 파일 컴파일
결과: USE_CUDA가 정의되어 있어도 .cpp 파일이 컴파일되어 빈 심볼 생성
```

## 해결 방법 (Solution)

### 1. Makefile 수정 (Makefile Fix)

**변경 내용**: CUDA 빌드 시 `.cu` 파일 규칙을 `.cpp` 규칙보다 먼저 정의

**변경 전 (Before)**:
```makefile
ifeq ($(BUILD_TYPE),cpu)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(PYTHON_FLAGS) -c $< -o $@
else ifeq ($(BUILD_TYPE),cuda)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@
else ifeq ($(BUILD_TYPE),sycl)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(SYCL_CXX) $(CXXFLAGS) $(PYTHON_FLAGS) -c $< -o $@
endif

ifeq ($(BUILD_TYPE),cuda)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@
endif
```

**변경 후 (After)**:
```makefile
# For CUDA builds, prioritize .cu files over .cpp files by placing the .cu rule first
ifeq ($(BUILD_TYPE),cuda)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@
else ifeq ($(BUILD_TYPE),cpu)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(PYTHON_FLAGS) -c $< -o $@
else ifeq ($(BUILD_TYPE),sycl)
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(SYCL_CXX) $(CXXFLAGS) $(PYTHON_FLAGS) -c $< -o $@
endif
```

**핵심 변경사항**:
- CUDA 빌드 시 `.cu` 규칙을 최우선으로 배치
- Make가 `obj/pme_calculator.o`를 생성할 때 `src/pme_calculator.cu`를 먼저 매칭하도록 수정

### 2. FFTW3 라이브러리 설치 (FFTW3 Library Installation)

**문제**: CPU 빌드 시 `fftw3.h` 헤더 파일을 찾을 수 없음

**해결**: FFTW3 개발 패키지 설치

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y libfftw3-dev
```

#### CentOS/RHEL/Rocky Linux:
```bash
sudo yum install -y fftw-devel
```

#### macOS:
```bash
brew install fftw
```

### 3. README.md 업데이트 (README.md Update)

README 파일에 다음 내용 추가:
- FFTW3를 필수 의존성으로 명시
- 각 운영체제별 의존성 설치 방법 추가
- PME 계산에서 CPU는 FFTW3, GPU는 cuFFT를 사용한다는 설명 추가

## 검증 결과 (Verification Results)

### CPU 빌드 성공 (CPU Build Success)
```bash
$ make clean && make cpu
# ... 컴파일 진행 ...
# pme_calculator.cpp 컴파일 (FFTW3 사용)
# 링크 성공: bin/md_simulation_cpu 생성
```

### CUDA 빌드 성공 (CUDA Build Success)
```bash
$ make clean && make cuda
# ... 컴파일 진행 ...
# pme_calculator.cu 컴파일 (cuFFT 사용)
# 링크 성공: bin/md_simulation_cuda 생성
```

## 파일 구조 (File Structure)

```
DeepScaleMD/
├── src/
│   ├── pme_calculator.cpp    # CPU 버전 (FFTW3 사용)
│   ├── pme_calculator.cu     # CUDA 버전 (cuFFT 사용)
│   └── pme_kernels.cu        # CUDA 커널 구현
├── include/
│   └── pme_calculator.h      # 공통 헤더 (조건부 컴파일)
└── Makefile                   # 빌드 시스템 (수정됨)
```

## 조건부 컴파일 로직 (Conditional Compilation Logic)

### pme_calculator.h
```cpp
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cufft.h>
// CUDA 관련 멤버 변수 및 메서드
#else
#include <fftw3.h>
// CPU 관련 멤버 변수 및 메서드
#endif
```

### pme_calculator.cpp
```cpp
#ifndef USE_CUDA
// CPU 구현만 포함
// ...
#endif
```

### pme_calculator.cu
```cpp
#ifdef USE_CUDA
// CUDA 구현만 포함
// ...
#endif
```

## 요약 (Summary)

✅ **해결 완료**:
1. Makefile 규칙 순서 수정으로 CUDA 빌드 시 올바른 소스 파일 컴파일
2. FFTW3 설치로 CPU 빌드 의존성 해결
3. README 문서화로 향후 빌드 문제 예방

✅ **테스트 완료**:
- `make cpu`: 성공 (pme_calculator.cpp + FFTW3)
- `make cuda`: 성공 (pme_calculator.cu + cuFFT)

🎯 **핵심 교훈**:
- Make의 패턴 규칙은 정의 순서대로 매칭됨
- 동일한 오브젝트 파일을 생성하는 여러 소스 파일이 있을 때는 규칙 순서가 중요함
- 조건부 컴파일 사용 시 빌드 시스템과의 일관성 유지 필요
