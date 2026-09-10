cmake_minimum_required(VERSION 3.18)

project(DeepScaleMD VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

set(BACKEND "CPU" CACHE STRING
    "Execution backend: CPU, CUDA, or SYCL")
set_property(CACHE BACKEND PROPERTY STRINGS CPU CUDA SYCL)
string(TOUPPER "${BACKEND}" BACKEND)

option(ENABLE_PYTHON "Enable the embedded Python force interface" ON)
option(ENABLE_LIBTORCH
       "Enable the native LibTorch/TorchScript MLIP runtime (CUDA only)" OFF)
option(ENABLE_ADIOS2
       "Enable the ADIOS2 BP parallel trajectory writer (trajectory_format: bp)" OFF)
option(NATIVE_OPTIMIZATION
       "Enable native CPU optimization for Release builds" ON)
set(DSDECOMPX_DIR "" CACHE PATH
    "Path to dsDecompX package config dir or dsDecompX install/build prefix")

set(_valid_backends CPU CUDA SYCL)
if(NOT BACKEND IN_LIST _valid_backends)
    message(FATAL_ERROR
        "Invalid BACKEND='${BACKEND}'. "
        "Choose CPU, CUDA, or SYCL.")
endif()

if(ENABLE_LIBTORCH AND NOT BACKEND STREQUAL "CUDA")
    message(FATAL_ERROR
        "ENABLE_LIBTORCH requires BACKEND=CUDA.")
endif()

set(_deepscalemd_uses_cuda OFF)
if(BACKEND STREQUAL "CUDA")
    set(_deepscalemd_uses_cuda ON)

    if(DEFINED CMAKE_CUDA_HOST_COMPILER)
        message(STATUS "DeepScaleMD: CUDA host compiler ${CMAKE_CUDA_HOST_COMPILER}")
    endif()

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
if(DSDECOMPX_DIR)
    if(EXISTS "${DSDECOMPX_DIR}/dsDecompXConfig.cmake")
        set(dsDecompX_DIR "${DSDECOMPX_DIR}" CACHE PATH
            "Path to dsDecompX package config directory" FORCE)
    elseif(EXISTS "${DSDECOMPX_DIR}/lib/cmake/dsDecompX/dsDecompXConfig.cmake")
        set(dsDecompX_DIR "${DSDECOMPX_DIR}/lib/cmake/dsDecompX" CACHE PATH
            "Path to dsDecompX package config directory" FORCE)
    else()
        message(FATAL_ERROR
            "DSDECOMPX_DIR='${DSDECOMPX_DIR}' does not contain dsDecompXConfig.cmake. "
            "Use a dsDecompX build dir, config dir, or install prefix.")
    endif()
endif()
find_package(dsDecompX 0.3 REQUIRED CONFIG)
find_package(OpenMP REQUIRED COMPONENTS CXX)

if(_deepscalemd_uses_cuda AND NOT dsDecompX_WITH_CUDA)
    message(FATAL_ERROR
        "${BACKEND} requires a CUDA-enabled dsDecompX package. "
        "Build dsDecompX with -DENABLE_CUDA=ON and set DSDECOMPX_DIR accordingly.")
endif()
if(NOT _deepscalemd_uses_cuda AND dsDecompX_WITH_CUDA)
    message(FATAL_ERROR
        "${BACKEND} requires the CPU dsDecompX package, but a "
        "CUDA-enabled package was selected. Point DSDECOMPX_DIR at the "
        "separate CPU dsDecompX package.")
endif()

if(ENABLE_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
endif()

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(FFTW3 QUIET IMPORTED_TARGET fftw3)
endif()
if(TARGET PkgConfig::FFTW3)
    set(_fftw_target PkgConfig::FFTW3)
else()
    find_path(FFTW3_INCLUDE_DIR fftw3.h REQUIRED)
    find_library(FFTW3_LIBRARY NAMES fftw3 REQUIRED)
    add_library(deepscalemd_fftw3 UNKNOWN IMPORTED)
    set_target_properties(deepscalemd_fftw3 PROPERTIES
        IMPORTED_LOCATION "${FFTW3_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIR}")
    set(_fftw_target deepscalemd_fftw3)
endif()

set(DEEPSCALEMD_COMMON_SOURCES
    src/md_system.cpp
    src/topology.cpp
    src/integrator.cpp
    src/constraint_solver.cpp
    src/thermostat.cpp
    src/barostat.cpp
    src/neighbor_list.cpp
    src/minimizer.cpp
    src/simulation_backend_factory.cpp
    src/halo_backend.cpp
    src/comm_backend.cpp
    src/md_simulation.cpp
    src/force_python/python_interface.cpp
    src/force_calculator_cpu.cpp
    src/force_manager.cpp
    src/mpi_utils.cpp
    src/mpi_optimizer.cpp
    src/main.cpp)

set(DEEPSCALEMD_HOST_SOURCES
    src/timing_breakdown.cpp
    src/force_libtorch/mlip_runtime_stub.cpp)

set(DEEPSCALEMD_MLIP_CUDA_SOURCES
    src/force_libtorch/config/yaml_parser.cpp
    src/force_libtorch/runtime/execution_plan.cpp
    src/force_libtorch/runtime/mlip_runtime.cpp
    src/force_libtorch/binding/md_data_sources.cpp
    src/force_libtorch/mlip_force_bridge.cpp
    src/force_libtorch/operations/operation_registry_singletons.cpp
    src/force_libtorch/operations/cuda/preprocess_ops.cu
    src/force_libtorch/operations/cuda/postprocess_ops.cu
    src/force_libtorch/cuda/edge_conversion.cu
    src/force_libtorch/cuda/force_accumulation.cu)

set(DEEPSCALEMD_CUDA_SOURCES
    src/integrator_kernels.cu
    src/barostat_kernels.cu
    src/thermostat_kernels.cu
    src/force_calculator_cuda.cu
    src/neighbor_list_kernels.cu
    src/neighbor_list_kernels_optimized.cu
    src/timing_breakdown.cpp
    ${DEEPSCALEMD_MLIP_CUDA_SOURCES})

add_executable(deepscalemd ${DEEPSCALEMD_COMMON_SOURCES})
add_executable(DeepScaleMD::deepscalemd ALIAS deepscalemd)

target_compile_features(deepscalemd PRIVATE cxx_std_17)
target_include_directories(deepscalemd PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/force_libtorch")
target_compile_definitions(deepscalemd PRIVATE USE_MPI)
target_link_libraries(deepscalemd PRIVATE
    dsdecompx::dsdecompx
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    ${_fftw_target}
    ${CMAKE_DL_LIBS})

if(ENABLE_PYTHON)
    target_compile_definitions(deepscalemd PRIVATE ENABLE_PYTHON_INTERFACE)
    target_link_libraries(deepscalemd PRIVATE Python3::Python Python3::NumPy)
endif()

if(ENABLE_ADIOS2)
    # Needs an MPI-enabled ADIOS2 install (ADIOS2_USE_MPI=ON) for the
    # adios2::cxx11_mpi target -- the BP writer's per-rank parallel local
    # arrays require the MPI-aware engine, not the serial cxx11 bindings.
    find_package(ADIOS2 REQUIRED)
    target_compile_definitions(deepscalemd PRIVATE USE_ADIOS2)
    target_link_libraries(deepscalemd PRIVATE adios2::cxx11_mpi)
endif()

if(BACKEND STREQUAL "CPU")
    target_sources(deepscalemd PRIVATE ${DEEPSCALEMD_HOST_SOURCES})
    set_target_properties(deepscalemd PROPERTIES OUTPUT_NAME dsMD)
elseif(BACKEND STREQUAL "SYCL")
    target_sources(deepscalemd PRIVATE ${DEEPSCALEMD_HOST_SOURCES} src/force_calculator_sycl.cpp)
    target_compile_definitions(deepscalemd PRIVATE USE_SYCL)
    target_compile_options(deepscalemd PRIVATE -fsycl)
    target_link_options(deepscalemd PRIVATE -fsycl)
    set_target_properties(deepscalemd PROPERTIES OUTPUT_NAME dsMD)
else()
    find_package(yaml-cpp REQUIRED CONFIG)
    if(TARGET yaml-cpp::yaml-cpp)
        set(_yaml_target yaml-cpp::yaml-cpp)
    else()
        set(_yaml_target yaml-cpp)
    endif()

    target_sources(deepscalemd PRIVATE ${DEEPSCALEMD_CUDA_SOURCES})
    target_compile_definitions(deepscalemd PRIVATE USE_CUDA)
    target_compile_options(deepscalemd PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    target_link_libraries(deepscalemd PRIVATE
        ${_yaml_target} CUDA::cudart CUDA::cublas CUDA::curand CUDA::cufft)
    set_target_properties(deepscalemd PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        OUTPUT_NAME dsMD)

    if(ENABLE_LIBTORCH)
        find_package(Torch REQUIRED CONFIG)
        target_sources(deepscalemd PRIVATE
            src/force_libtorch/backend/pytorch_backend.cpp
            src/force_libtorch/backend/pytorch_backend_register.cpp)
        target_compile_definitions(deepscalemd PRIVATE USE_LIBTORCH)
        target_link_libraries(deepscalemd PRIVATE ${TORCH_LIBRARIES})
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")
    target_compile_options(deepscalemd PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>)
    if(NATIVE_OPTIMIZATION)
        target_compile_options(deepscalemd PRIVATE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-march=native>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-mtune=native>)
    endif()
endif()

set_target_properties(deepscalemd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
install(TARGETS deepscalemd RUNTIME DESTINATION bin)

if(BUILD_TESTING)
    add_test(
        NAME dsMD_input_mie_12_6_smoke
        COMMAND deepscalemd "${CMAKE_CURRENT_SOURCE_DIR}/examples/dsMD_input_lj/input.yaml")
    set_tests_properties(dsMD_input_mie_12_6_smoke PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        PASS_REGULAR_EXPRESSION "Nonbonded Mie potential: [(]12,6[)]")

    # The 20 A periodic box produces only two bins per dimension at the
    # 10 A neighbor cutoff. This catches both wrapped-stencil duplication and
    # a missing F(r0) evaluation before the first velocity-Verlet half-kick.
    add_test(
        NAME dsMD_input_initial_force_small_cell_regression
        COMMAND deepscalemd "${CMAKE_CURRENT_SOURCE_DIR}/examples/dsMD_input_lj/input.yaml")
    set_tests_properties(dsMD_input_initial_force_small_cell_regression PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        PASS_REGULAR_EXPRESSION "Step: 0, PE: -0[.]2244")

    add_test(
        NAME dsMD_input_mie_9_6_smoke
        COMMAND deepscalemd "${CMAKE_CURRENT_SOURCE_DIR}/examples/dsMD_input_lj/input_mie96.yaml")
    set_tests_properties(dsMD_input_mie_9_6_smoke PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        PASS_REGULAR_EXPRESSION "Nonbonded Mie potential: [(]9,6[)]")

    # A zero-temperature 90-degree angle with a 120-degree target must move
    # downhill. This catches a sign reversal between harmonic angle energy
    # and force on both CPU and CUDA backends.
    add_test(
        NAME dsMD_input_harmonic_angle_force_sign
        COMMAND deepscalemd "${CMAKE_CURRENT_SOURCE_DIR}/examples/dsMD_input_angle/input.yaml")
    set_tests_properties(dsMD_input_harmonic_angle_force_sign PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        PASS_REGULAR_EXPRESSION "Step: 1, KE: [^,]+, PE: 13[.]6")

    # Exercises the shared signed-dihedral gradient with both a periodic
    # proper term and a CHARMM-style harmonic improper term. Starting from
    # rest, their combined potential must move downhill.
    add_test(
        NAME dsMD_input_torsion_force_sign
        COMMAND deepscalemd "${CMAKE_CURRENT_SOURCE_DIR}/examples/dsMD_input_torsion/input.yaml")
    set_tests_properties(dsMD_input_torsion_force_sign PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        PASS_REGULAR_EXPRESSION "Step: 1, KE: [^,]+, PE: 58[.]444")

    add_test(
        NAME dsMD_input_fire_minimizer
        COMMAND deepscalemd "${CMAKE_CURRENT_SOURCE_DIR}/examples/dsMD_input_angle/input_minimize.yaml")
    set_tests_properties(dsMD_input_fire_minimizer PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        PASS_REGULAR_EXPRESSION "Reinitialized post-minimization velocities at T = 300[.]000 K")

    add_test(
        NAME dsMD_input_xyz_dcd_output
        COMMAND "${CMAKE_COMMAND}"
            "-DDSMD_EXECUTABLE=$<TARGET_FILE:deepscalemd>"
            "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            "-DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestTrajectoryOutput.cmake")
endif()

message(STATUS "DeepScaleMD backend: ${BACKEND}")
message(STATUS "DeepScaleMD LibTorch runtime: ${ENABLE_LIBTORCH}")
message(STATUS "dsDecompX CUDA support: ${dsDecompX_WITH_CUDA}")
message(STATUS "dsDecompX NCCL support: ${dsDecompX_WITH_NCCL}")
message(STATUS "dsDecompX NVSHMEM support: ${dsDecompX_WITH_NVSHMEM}")
