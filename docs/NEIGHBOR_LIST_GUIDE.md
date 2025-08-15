# Neighbor List Implementation Guide

이 프로젝트는 MD 시뮬레이션을 위한 두 가지 neighbor list 구현을 제공합니다.

## 1. Traditional Cell List (`cell_list`)
- **특징**: 전통적인 공간 분할 방식
- **적용**: 모든 컴파일 타입에서 사용 가능
- **메모리**: 적은 메모리 사용
- **성능**: CPU에서 효율적

## 2. Morton/Hilbert CSR (`morton_hilbert`)
- **특징**: 공간 충전 곡선과 CSR 포맷을 사용한 최신 방식
- **적용**: CUDA 빌드에서만 사용 가능 (`BUILD_TYPE=cuda`)
- **최적화**: GPU 메모리 coalescing을 위한 타일 스테이징
- **성능**: GPU에서 높은 성능

## 사용법

### YAML 설정에서 선택
```yaml
# 전통적인 Cell List 사용
neighbor_list_type: "cell_list"

# Morton/Hilbert CSR 사용 (CUDA 빌드 필요)
neighbor_list_type: "morton_hilbert"
neighbor_list_tile_size: 256  # GPU 최적화를 위한 타일 크기
```

### 컴파일
```bash
# CPU 빌드 (cell_list만 사용 가능)
make BUILD_TYPE=cpu

# CUDA 빌드 (두 방식 모두 사용 가능)
make BUILD_TYPE=cuda
```

### 실행 예제
```bash
# Morton/Hilbert 테스트
./bin/md_simulation_cuda examples/morton_test.yaml

# 전통적인 방식
./bin/md_simulation_cpu examples/input.yaml
```

## 성능 특징

### Cell List
- 구현 단순성: ⭐⭐⭐⭐⭐
- CPU 성능: ⭐⭐⭐⭐
- GPU 성능: ⭐⭐⭐
- 메모리 효율: ⭐⭐⭐⭐⭐

### Morton/Hilbert CSR  
- 구현 복잡성: ⭐⭐⭐⭐⭐
- CPU 성능: ⭐⭐⭐
- GPU 성능: ⭐⭐⭐⭐⭐
- 메모리 지역성: ⭐⭐⭐⭐⭐

## 주요 기능

### Morton/Hilbert 방식의 고급 기능
1. **공간 충전 곡선**: 3D 좌표를 1D Morton/Hilbert 코드로 변환
2. **파티클 재정렬**: 메모리 지역성 향상을 위한 데이터 재배치
3. **CSR 포맷**: GPU에 최적화된 압축 희소 행 형식
4. **타일 스테이징**: GPU 메모리 coalescing 최적화
5. **자동 복원**: 계산 후 원래 파티클 순서로 복원

### 자동 fallback
- Morton/Hilbert를 요청했지만 CUDA가 없는 경우 자동으로 Cell List로 전환
- 안전한 메모리 제한으로 시스템 안정성 보장
