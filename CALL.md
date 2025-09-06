# DeepScaleMD 함수 호출 순서 요약 (CALL.md)

아래는 대표적인 MD 시뮬레이션 실행 흐름의 함수 호출 순서 예시입니다.

1. **main()**
   - 프로그램 시작, 입력 파일 파싱
2. **MDSystem::initialize()**
   - 시스템 초기화 (입자, 박스, force field 등)
3. **MDSystem::setup_neighbor_list()**
   - 네이버 리스트 생성
4. **MDSystem::setup_integrator()**
   - 적분기(Integrator) 설정
5. **MDSystem::run()**
   - 시뮬레이션 루프 시작
     - for step in total_steps:
       1. **ForceManager::calculate_forces()**
          - 내부적으로 force_calculator_cpu/cuda/sycl 등에서 실제 force 계산 함수 호출
          - 예: force_calculator_cpu::compute(), force_calculator_cuda::compute()
       2. **Integrator::integrate()**
          - 예: integrator::velocity_verlet(), integrator::leapfrog() 등에서 속도/위치 업데이트
       3. **Thermostat/Barostat::apply()** (필요시)
          - 예: thermostat::apply(), barostat::apply() 함수에서 온도/압력 조절
       4. **NeighborList::update()** (필요시)
          - neighbor_list::update() 또는 optimized_neighbor_list::update() 함수 호출
       5. **OutputManager::write()**
          - output_manager::write_step(), output_manager::write_thermo() 등에서 결과 출력
6. **MDSystem::finalize()**
   - 시뮬레이션 종료, 결과 저장

> 실제 함수명/클래스명은 코드에 따라 다를 수 있습니다.
> 상세 호출 구조는 src/ 및 include/ 폴더의 주요 cpp/h 파일을 참고하세요.
