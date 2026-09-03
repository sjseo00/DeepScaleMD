# CUDA Debug TODO

## 현재 상태 (2026-02-27)

### 빌드 상태
- ✅ CUDA 빌드 성공
- ✅ GPU 인식됨 (NVIDIA GeForce RTX 2080 Ti)
- ✅ ForceCalculatorCUDA 클래스 인스턴스화 성공
- ✅ USE_CUDA 매크로 정의됨

### 문제 증상
1. **힘이 계산되지 않음**
   - `fx = fy = fz ≈ 0.000`
   - Potential Energy ≈ -50 kcal/mol (정상: -1760 kcal/mol)
   - 입자들이 자유 비행 (free flight) 상태

2. **Neighbor list 업데이트 안됨**
   - Neighbor list builds = 1 (정상: ~280)
   - `compute_max_displacement_on_device()` 항상 작은 값 반환

3. **성능**
   - 실행 시간: 9.7초 (CPU 22초보다 빠르지만 힘 계산 안됨)

### 진단 결과

**Neighbor List 데이터 흐름:**
```
CPU에서 빌드 → update_neighbor_list_on_device() 호출 
→ cudaMemcpyAsync로 GPU 복사 → cudaStreamSynchronize() 호출
→ compute_forces() 실행 → compute_lj_forces_kernel() 실행
→ BUT: kernel에서 num_neighbors[i] = 0
```

**디버그 로그 출력:**
```
[DEBUG] update_neighbor_list_on_device: 
  num_particles=32000
  nlist.neighbors.size()=568128  ✅ CPU에서 정상 빌드
  nlist.num_neighbors.size()=32000
  nlist.neighbor_offset.size()=32000
```

### 가능한 원인

1. **메모리 할당 문제**
   - `d_num_neighbors`, `d_neighbor_offset` 포인터가 nullptr
   - 할당은 `allocate_device_memory()`에서 수행됨
   - 하지만 `initial_upload_to_device()` 이전에 호출되지 않았을 가능성

2. **동기화 문제**
   - Stream 동기화가 충분하지 않음
   - 다른 stream에서 kernel이 실행될 가능성

3. **포인터 전달 문제**
   - Kernel에 전달되는 포인터가 올바르지 않음
   - `compute_lj_forces_kernel<<<>>>()` 호출 시 포인터 검증 필요

### 다음 디버깅 단계

1. **Kernel 내부에서 디버그 출력**
   ```cuda
   if (threadIdx.x == 0 && blockIdx.x == 0) {
       printf("[CUDA] i=0, n_count=%d, n_start=%d\n", 
              num_neighbors[0], neighbor_offset[0]);
   }
   ```

2. **포인터 유효성 검증**
   ```cpp
   if (is_rank_zero()) {
       std::cout << "d_neighbors=" << (void*)d_neighbors 
                 << ", d_num_neighbors=" << (void*)d_num_neighbors
                 << ", d_neighbor_offset=" << (void*)d_neighbor_offset << std::endl;
   }
   ```

3. **GPU 메모리 내용 확인**
   ```cpp
   std::vector<int> host_num_neighbors(10);
   cudaMemcpy(host_num_neighbors.data(), d_num_neighbors, 10*sizeof(int), 
              cudaMemcpyDeviceToHost);
   for (int i = 0; i < 10; i++) {
       std::cout << "num_neighbors[" << i << "]=" << host_num_neighbors[i] << std::endl;
   }
   ```

4. **Reference position 초기화 검증**
   - `d_x_ref`, `d_y_ref`, `d_z_ref`가 제대로 초기화되었는지 확인
   - `compute_max_displacement_on_device()` 반환값 검증

### 우회 방법 (빠른 해결)

CPU neighbor list 빌드 후 강제 업데이트:
```cpp
// md_simulation.cpp에서
if (using_cuda && cuda_calc) {
    // 매 20 step마다 강제 neighbor list 업데이트
    if (current_step % 20 == 0) {
        neighbor_builder->build(...);
        cuda_calc->update_neighbor_list_on_device(*system);
    }
}
```

---

## 현재 우선순위

**STAGE 0~5는 CPU 버전으로 완료** (이미 작동하며 성능 우수)

CUDA 최적화는 별도 세션에서 진행:
- 위 디버깅 단계 수행
- GPU profiling (nsight systems/compute)
- Kernel 최적화
- Multi-GPU 지원

---

## 수정 이력

### 2026-02-27
1. ✅ `initial_upload_to_device()`에 reference position 초기화 추가
2. ✅ `update_neighbor_list_on_device()`에 `cudaStreamSynchronize()` 추가
3. ⚠️ 여전히 힘 계산 안됨 → 추가 디버깅 필요
