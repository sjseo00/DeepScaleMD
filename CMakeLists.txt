cmake_minimum_required(VERSION 3.18)

project(DeepScaleMD VERSION 0.1.0 LANGUAGES CXX)

set(BACKEND "CPU" CACHE STRING
    "Execution backend: CPU, CUDA, or SYCL")
set_property(CACHE BACKEND PROPERTY STRINGS CPU CUDA SYCL)
string(TOUPPER "${BACKEND}" BACKEND)

option(DEEPSCALEMD_ENABLE_PYTHON "Enable the embedded Python force interface" ON)
option(DEEPSCALEMD_ENABLE_LIBTORCH
       "Enable the native LibTorch/TorchScript MLIP runtime (CUDA only)" OFF)
option(DEEPSCALEMD_ENABLE_NATIVE_OPTIMIZATION
       "Enable native CPU optimization for Release builds" ON)

set(_valid_backends CPU CUDA SYCL)
if(NOT BACKEND IN_LIST _valid_backends)
    message(FATAL_ERROR
        "Invalid BACKEND='${BACKEND}'. "
        "Choose CPU, CUDA, or SYCL.")
endif()

if(DEEPSCALEMD_ENABLE_LIBTORCH AND NOT BACKEND STREQUAL "CUDA")
    message(FATAL_ERROR
        "DEEPSCALEMD_ENABLE_LIBTORCH requires BACKEND=CUDA.")
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
find_package(dsDecomp 0.3 REQUIRED CONFIG)
find_package(OpenMP REQUIRED COMPONENTS CXX)

if(_deepscalemd_uses_cuda AND NOT dsDecomp_WITH_CUDA)
    message(FATAL_ERROR
        "${BACKEND} requires a CUDA-enabled dsDecomp package. "
    "Build dsDecomp with -DENABLE_CUDA=ON and select its install "
        "prefix through CMAKE_PREFIX_PATH.")
endif()
if(NOT _deepscalemd_uses_cuda AND dsDecomp_WITH_CUDA)
    message(FATAL_ERROR
        "${BACKEND} requires the CPU dsDecomp package, but a "
        "CUDA-enabled package was selected. Point CMAKE_PREFIX_PATH at the "
        "separate CPU dsDecomp installation.")
endif()

if(DEEPSCALEMD_ENABLE_PYTHON)
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
    src/thermostat.cpp
    src/barostat.cpp
    src/neighbor_list.cpp
    src/simulation_backend_factory.cpp
    src/halo_backend.cpp
    src/comm_backend.cpp
    src/md_simulation.cpp
    src/python_interface.cpp
    src/force_calculator_cpu.cpp
    src/force_calculator_ml.cpp
    src/force_manager.cpp
    src/mpi_utils.cpp
    src/mpi_optimizer.cpp
    src/main.cpp)

set(DEEPSCALEMD_HOST_SOURCES
    src/timing_breakdown.cpp
    src/mlip_runtime_stub.cpp)

set(DEEPSCALEMD_MLIP_CUDA_SOURCES
    src/mlip/config/yaml_parser.cpp
    src/mlip/runtime/execution_plan.cpp
    src/mlip/runtime/mlip_runtime.cpp
    src/mlip/binding/md_data_sources.cpp
    src/mlip_force_bridge.cpp
    src/mlip/operations/operation_registry_singletons.cpp
    src/mlip/operations/cuda/preprocess_ops.cu
    src/mlip/operations/cuda/postprocess_ops.cu
    src/potential/mlip/cuda/edge_conversion.cu
    src/potential/mlip/cuda/force_accumulation.cu)

set(DEEPSCALEMD_CUDA_SOURCES
    src/integrator_kernels.cu
    src/barostat_kernels.cu
    src/thermostat_kernels.cu
    src/force_calculator_cuda.cu
    src/neighbor_list_kernels.cu
    src/neighbor_list_kernels_optimized.cu
    src/cuda_graphs_system.cpp
    src/timing_breakdown.cpp
    ${DEEPSCALEMD_MLIP_CUDA_SOURCES})

add_executable(deepscalemd ${DEEPSCALEMD_COMMON_SOURCES})
add_executable(DeepScaleMD::deepscalemd ALIAS deepscalemd)

target_compile_features(deepscalemd PRIVATE cxx_std_17)
target_include_directories(deepscalemd PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/mlip"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/potential")
target_compile_definitions(deepscalemd PRIVATE USE_MPI)
target_link_libraries(deepscalemd PRIVATE
    dsdecomp::dsdecomp
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    ${_fftw_target}
    ${CMAKE_DL_LIBS})

if(DEEPSCALEMD_ENABLE_PYTHON)
    target_compile_definitions(deepscalemd PRIVATE ENABLE_PYTHON_INTERFACE)
    target_link_libraries(deepscalemd PRIVATE Python3::Python Python3::NumPy)
endif()

if(BACKEND STREQUAL "CPU")
    target_sources(deepscalemd PRIVATE ${DEEPSCALEMD_HOST_SOURCES})
    set_target_properties(deepscalemd PROPERTIES OUTPUT_NAME dsmd)
elseif(BACKEND STREQUAL "SYCL")
    target_sources(deepscalemd PRIVATE ${DEEPSCALEMD_HOST_SOURCES} src/force_calculator_sycl.cpp)
    target_compile_definitions(deepscalemd PRIVATE USE_SYCL)
    target_compile_options(deepscalemd PRIVATE -fsycl)
    target_link_options(deepscalemd PRIVATE -fsycl)
    set_target_properties(deepscalemd PROPERTIES OUTPUT_NAME dsmd)
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
        OUTPUT_NAME dsmd)

    if(DEEPSCALEMD_ENABLE_LIBTORCH)
        find_package(Torch REQUIRED CONFIG)
        target_sources(deepscalemd PRIVATE
            src/mlip/backend/pytorch_backend.cpp
            src/mlip/backend/pytorch_backend_register.cpp)
        target_compile_definitions(deepscalemd PRIVATE USE_LIBTORCH)
        target_link_libraries(deepscalemd PRIVATE ${TORCH_LIBRARIES})
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")
    target_compile_options(deepscalemd PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>)
    if(DEEPSCALEMD_ENABLE_NATIVE_OPTIMIZATION)
        target_compile_options(deepscalemd PRIVATE
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-march=native>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-mtune=native>)
    endif()
endif()

set_target_properties(deepscalemd PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
install(TARGETS deepscalemd RUNTIME DESTINATION bin)

message(STATUS "DeepScaleMD backend: ${BACKEND}")
message(STATUS "DeepScaleMD LibTorch runtime: ${DEEPSCALEMD_ENABLE_LIBTORCH}")
message(STATUS "dsDecomp CUDA support: ${dsDecomp_WITH_CUDA}")
message(STATUS "dsDecomp NCCL support: ${dsDecomp_WITH_NCCL}")
message(STATUS "dsDecomp NVSHMEM support: ${dsDecomp_WITH_NVSHMEM}")
