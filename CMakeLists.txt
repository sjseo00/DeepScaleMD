cmake_minimum_required(VERSION 3.18)

project(DeepScaleMD VERSION 0.1.0 LANGUAGES CXX)


if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel)
    message(STATUS "DeepScaleMD: CMAKE_BUILD_TYPE not set, defaulting to Release "
                    "(pass -DCMAKE_BUILD_TYPE=Debug explicitly for a debug build)")
endif()

include(CTest)

set(BACKEND "CPU" CACHE STRING
    "Execution backend: CPU, CUDA, or SYCL")
set_property(CACHE BACKEND PROPERTY STRINGS CPU CUDA SYCL)
string(TOUPPER "${BACKEND}" BACKEND)

option(ENABLE_PYTHON "Enable the embedded Python MLIP interface (mlip_backend: python)" ON)
option(ENABLE_LIBTORCH
       "Build the LibTorch MLIP backend (mlip_backend: libtorch); BACKEND=CUDA only" OFF)
set(LIBTORCH_PYTHON "" CACHE FILEPATH
    "Python interpreter whose torch package provides LibTorch (default: python3 on PATH); ignored when Torch_DIR is set")
option(ENABLE_ADIOS2
       "Enable the ADIOS2 BP parallel trajectory writer (trajectory_format: bp)" OFF)
option(ENABLE_HDF5
       "Enable the H5MD parallel trajectory writer (trajectory_format: h5md)" OFF)
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
        "ENABLE_LIBTORCH requires -DBACKEND=CUDA: the LibTorch MLIP backend is not available in "
        "${BACKEND} builds. Use mlip_backend: python there.")
endif()

set(_deepscalemd_uses_cuda OFF)
if(BACKEND STREQUAL "CUDA")
    set(_deepscalemd_uses_cuda ON)

    if(DEFINED CMAKE_CUDA_HOST_COMPILER)
        message(STATUS "DeepScaleMD: CUDA host compiler ${CMAKE_CUDA_HOST_COMPILER}")
    endif()

    # Left unset, nvcc falls back to an old default virtual architecture
    # (pre-Pascal on most toolkits). double-precision atomicAdd -- used
    # heavily in the per-pair force kernels and PME grid deposition -- has no
    # native instruction below compute capability 6.0, so nvcc silently emits
    # a CAS-loop emulation instead. That emulation is what actually runs on
    # the GPU regardless of its real architecture, and under the heavy
    # atomicAdd contention in these kernels it is catastrophically slower
    # (10-100x) than the native instruction. Default to a reasonably modern
    # architecture list so a plain `cmake -DBACKEND=CUDA` build isn't silently
    # crippled; override with -DCMAKE_CUDA_ARCHITECTURES=<arch> to match your GPU
    # exactly (e.g. 86 for RTX A6000/30xx, 80 for A100, 90 for H100).
    # 90 (Hopper/H100, GH200) is included alongside the workstation/datacenter
    # Ampere/Ada architectures for forward compatibility with the
    # thread-block-cluster PME spread launch path (force_calculator_cuda.cu,
    # pme_spread_charges_cluster_kernel). That kernel itself has no
    # architecture-specific device code and builds fine for every listed
    # arch; only the *launch* -- a cudaLaunchKernelEx call with a cluster
    # dimension attribute -- actually requires Hopper, and that call is
    # gated at runtime by device_supports_clusters_, never selected on any
    # GPU below compute capability 9.0.
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 80 86 89 90)
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
find_package(dsDecompX 1.0 REQUIRED CONFIG)
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

# Optional: FFTW's OpenMP-threaded backend. Preferred over libfftw3_threads
# (pthread-based) since it shares the same OpenMP runtime already linked via
# OpenMP::OpenMP_CXX rather than mixing two threading models. Purely additive
# -- if not found, PME reciprocal-space FFTs stay single-threaded (which is
# also what happens at runtime with OMP_NUM_THREADS=1 either way) and nothing
# else is affected.
find_library(FFTW3_OMP_LIBRARY NAMES fftw3_omp)
if(FFTW3_OMP_LIBRARY)
    set(_fftw_omp_target ${FFTW3_OMP_LIBRARY})
    set(DEEPSCALEMD_HAVE_FFTW_OMP TRUE)
    message(STATUS "DeepScaleMD: found libfftw3_omp (${FFTW3_OMP_LIBRARY}) -- "
                    "PME reciprocal-space FFTs will run multi-threaded "
                    "(OMP_NUM_THREADS threads, same as everything else)")
else()
    set(_fftw_omp_target "")
    set(DEEPSCALEMD_HAVE_FFTW_OMP FALSE)
    message(STATUS "DeepScaleMD: libfftw3_omp NOT found -- PME reciprocal-space "
                    "FFTs will run SINGLE-THREADED regardless of OMP_NUM_THREADS, "
                    "which becomes a larger fraction of step time at higher core "
                    "counts. Install a FFTW3 build with its OpenMP backend (or set "
                    "FFTW3_OMP_LIBRARY manually) to fix this.")
endif()

# Optional: FFTW's MPI-parallel transform, needed only when
# force_calculation.pme_ranks > 1 (a distributed FFT *within* the dedicated
# PME rank group -- see PmeReciprocalEngine::ensure_initialized_distributed).
# pme_ranks 0 or 1 never touch this; purely additive like FFTW3_OMP above.
find_path(FFTW3_MPI_INCLUDE_DIR fftw3-mpi.h)
find_library(FFTW3_MPI_LIBRARY NAMES fftw3_mpi)
if(FFTW3_MPI_INCLUDE_DIR AND FFTW3_MPI_LIBRARY)
    add_library(deepscalemd_fftw3_mpi UNKNOWN IMPORTED)
    set_target_properties(deepscalemd_fftw3_mpi PROPERTIES
        IMPORTED_LOCATION "${FFTW3_MPI_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_MPI_INCLUDE_DIR}")
    set(_fftw_mpi_target deepscalemd_fftw3_mpi)
    set(DEEPSCALEMD_HAVE_FFTW_MPI TRUE)
    message(STATUS "DeepScaleMD: found libfftw3_mpi (${FFTW3_MPI_LIBRARY}) -- "
                    "force_calculation.pme_ranks > 1 is available")
else()
    set(_fftw_mpi_target "")
    set(DEEPSCALEMD_HAVE_FFTW_MPI FALSE)
    message(STATUS "DeepScaleMD: libfftw3_mpi NOT found -- "
                    "force_calculation.pme_ranks must stay 0 or 1 "
                    "(distributed-FFT-within-the-PME-group unavailable)")
endif()

set(DEEPSCALEMD_COMMON_SOURCES
    src/md_system.cpp
    src/topology.cpp
    src/integrator.cpp
    src/constraint_solver.cpp
    src/thermostat.cpp
    src/barostat.cpp
    src/neighbor_list_cpu.cpp
    src/neighbor_list_factory.cpp
    src/minimizer.cpp
    src/simulation_backend_factory.cpp
    src/halo_backend.cpp
    src/comm_backend.cpp
    src/md_simulation.cpp
    src/PythonBackend.cpp
    src/MLIPRankComm.cpp
    src/PythonModules.cpp
    src/MLIPBackendFactory.cpp
    src/Elements.cpp
    src/MLIPMetadata.cpp
    src/MiniJson.cpp
    src/force_calculator_cpu.cpp
    src/force_manager.cpp
    src/mpi_utils.cpp
    src/pme_rank_split.cpp
    src/pme_reciprocal_engine.cpp
    src/pme_server.cpp
    src/main.cpp)

set(DEEPSCALEMD_HOST_SOURCES
    src/timing_breakdown.cpp)

set(DEEPSCALEMD_CUDA_SOURCES
    src/integrator_kernels.cu
    src/barostat_kernels.cu
    src/thermostat_kernels.cu
    src/force_calculator_cuda.cu
    src/neighbor_list_kernels.cu
    src/neighbor_list_kernels_optimized.cu
    src/timing_breakdown.cpp)

add_executable(deepscalemd ${DEEPSCALEMD_COMMON_SOURCES})
add_executable(DeepScaleMD::deepscalemd ALIAS deepscalemd)

target_compile_features(deepscalemd PRIVATE cxx_std_17)
target_include_directories(deepscalemd PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(deepscalemd PRIVATE USE_MPI)
if(DEEPSCALEMD_HAVE_FFTW_OMP)
    target_compile_definitions(deepscalemd PRIVATE DEEPSCALEMD_HAVE_FFTW_OMP)
endif()
if(DEEPSCALEMD_HAVE_FFTW_MPI)
    target_compile_definitions(deepscalemd PRIVATE DEEPSCALEMD_HAVE_FFTW_MPI)
endif()
# The dsdecompx.hpp that belongs to the library must win over any other copy on the include path
# (CPATH, an old install under ~/.local). GCC searches CPATH before -isystem directories (and ignores
# a -I for a directory that is also -isystem), so the include directories of imported targets are not
# made -isystem here: they stay plain -I and are searched first. HaloExchangeOptions is passed by
# value, so a stale header misreads it.
set_property(TARGET deepscalemd PROPERTY NO_SYSTEM_FROM_IMPORTED ON)
get_target_property(_dsdecompx_include_dirs dsdecompx::dsdecompx INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "DeepScaleMD: dsdecompx include directory: ${_dsdecompx_include_dirs}")
target_link_libraries(deepscalemd PRIVATE
    dsdecompx::dsdecompx
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX
    ${_fftw_omp_target}
    ${_fftw_mpi_target}
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

if(ENABLE_HDF5)
    # The H5MD writer opens the file with H5Pset_fapl_mpio() and issues
    # collective MPI-IO hyperslab writes -- both require an HDF5 build
    # configured with --enable-parallel. A serial HDF5 install is missing
    # H5Pset_fapl_mpio entirely, so fail early with a clear message instead
    # of a confusing undefined-reference/link error later.
    find_package(HDF5 REQUIRED COMPONENTS C)
    if(NOT HDF5_IS_PARALLEL)
        message(FATAL_ERROR
            "ENABLE_HDF5 requires a parallel HDF5 build (configured with "
            "--enable-parallel / HDF5_ENABLE_PARALLEL=ON), but the HDF5 "
            "found at ${HDF5_C_LIBRARIES} is serial-only.")
    endif()
    target_compile_definitions(deepscalemd PRIVATE USE_HDF5)
    target_include_directories(deepscalemd PRIVATE ${HDF5_C_INCLUDE_DIRS})
    target_link_libraries(deepscalemd PRIVATE ${HDF5_C_LIBRARIES})
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
    target_sources(deepscalemd PRIVATE ${DEEPSCALEMD_CUDA_SOURCES})
    target_compile_definitions(deepscalemd PRIVATE USE_CUDA)
    target_compile_options(deepscalemd PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    target_link_libraries(deepscalemd PRIVATE
        CUDA::cudart CUDA::cublas CUDA::curand CUDA::cufft)

    # LibTorch for the MLIP backend: the one shipped in the Python torch package (the same torch the model
    # converter exports with). Pass -DTorch_DIR=<...>/share/cmake/Torch to use another LibTorch instead.
    if(ENABLE_LIBTORCH)
        if(NOT Torch_DIR)
            if(NOT LIBTORCH_PYTHON)
                find_program(LIBTORCH_PYTHON NAMES python3 python)
            endif()
            if(NOT LIBTORCH_PYTHON)
                message(FATAL_ERROR
                    "ENABLE_LIBTORCH: no Python interpreter found. Set -DLIBTORCH_PYTHON=<python with torch> "
                    "or -DTorch_DIR=<LibTorch>/share/cmake/Torch.")
            endif()
            execute_process(
                COMMAND "${LIBTORCH_PYTHON}" -c "import torch; print(torch.utils.cmake_prefix_path)"
                OUTPUT_VARIABLE _libtorch_prefix
                RESULT_VARIABLE _libtorch_python_status
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
            if(NOT _libtorch_python_status EQUAL 0 OR NOT _libtorch_prefix)
                message(FATAL_ERROR
                    "ENABLE_LIBTORCH: '${LIBTORCH_PYTHON}' cannot import torch. Set -DLIBTORCH_PYTHON=<python "
                    "with torch> or -DTorch_DIR=<LibTorch>/share/cmake/Torch.")
            endif()
            list(APPEND CMAKE_PREFIX_PATH "${_libtorch_prefix}")
        endif()
        # Torch's CMake config appends -gencode flags for the GPU it autodetects to CMAKE_CUDA_FLAGS. The MD
        # architectures come from CMAKE_CUDA_ARCHITECTURES (and the MD kernels do not build for sm_61), so put
        # the flags back.
        set(_dsmd_cuda_flags_before "${CMAKE_CUDA_FLAGS}")
        find_package(Torch REQUIRED)
        set(CMAKE_CUDA_FLAGS "${_dsmd_cuda_flags_before}")
        if(TORCH_CXX_FLAGS MATCHES "_GLIBCXX_USE_CXX11_ABI=0")
            message(FATAL_ERROR
                "ENABLE_LIBTORCH: this LibTorch uses the pre-cxx11 libstdc++ ABI, but dsDecompX and the MD code "
                "are built with the cxx11 ABI. Use a LibTorch built with the cxx11 ABI "
                "(torch.compiled_with_cxx11_abi() is True).")
        endif()
        target_compile_definitions(deepscalemd PRIVATE DSMD_WITH_LIBTORCH)
        # Torch's include directories and compile definitions (USE_DISTRIBUTED, USE_C10D_NCCL, ...) must not reach the
        # MD sources, so the LibTorch backend sources are a separate static library that links Torch privately; the
        # MD target links only that library. (Linking the Torch targets to the MD target directly, even through
        # $<LINK_ONLY:>, leaked them in a test project; going through a static library did not.) The library uses
        # load_mlip_metadata() from the MD sources, which the executable provides.
        add_library(dsmd_libtorch STATIC
            src/LibTorchBackend.cpp
            src/HaloExchangeOp.cpp
            src/TorchScriptRunner.cpp
            src/AOTIRunner.cpp
            src/OpLibLoader.cpp)
        target_compile_features(dsmd_libtorch PRIVATE cxx_std_17)
        target_include_directories(dsmd_libtorch PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
        target_compile_definitions(dsmd_libtorch PRIVATE USE_CUDA)
        separate_arguments(_torch_cxx_flags UNIX_COMMAND "${TORCH_CXX_FLAGS}")
        target_compile_options(dsmd_libtorch PRIVATE ${_torch_cxx_flags})
        target_link_libraries(dsmd_libtorch PRIVATE ${TORCH_LIBRARIES} CUDA::cudart ${CMAKE_DL_LIBS})
        target_link_libraries(deepscalemd PRIVATE dsmd_libtorch)
        # dsMD finds the LibTorch (and its bundled libraries) it was linked with without LD_LIBRARY_PATH
        set_property(TARGET deepscalemd APPEND PROPERTY BUILD_RPATH "${TORCH_INSTALL_PREFIX}/lib")
        set_property(TARGET deepscalemd APPEND PROPERTY INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib")
        # MD-side GPU work around the model (periodic images, edge list, force folding); only built with LibTorch.
        target_sources(deepscalemd PRIVATE src/EdgeBuilder.cu src/MLIPStep.cu src/MLIPReplica.cpp)
        message(STATUS "DeepScaleMD: LibTorch MLIP backend enabled (Torch ${Torch_VERSION}, ${Torch_DIR})")
    endif()

    # With an NCCL-enabled dsDecompX the multi-rank PME charge-grid sum runs on the device (NCCL
    # allreduce on the PME stream) instead of through pinned host memory and MPI. Turn it off at
    # configure time with -DDSMD_ENABLE_NCCL=OFF.
    option(DSMD_ENABLE_NCCL "Sum the multi-rank PME grid with NCCL when dsDecompX provides it" ON)
    if(DSMD_ENABLE_NCCL AND dsDecompX_WITH_NCCL AND TARGET NCCL::NCCL)
        target_compile_definitions(deepscalemd PRIVATE DSMD_WITH_NCCL=1)
        target_link_libraries(deepscalemd PRIVATE NCCL::NCCL)
        message(STATUS "DeepScaleMD: NCCL PME grid sum enabled")
    else()
        message(STATUS "DeepScaleMD: NCCL PME grid sum not built (MPI grid sum only)")
    endif()
    set_target_properties(deepscalemd PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        OUTPUT_NAME dsMD)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM")
    set(_deepscalemd_opt_configs "$<CONFIG:Release,RelWithDebInfo>")
    target_compile_options(deepscalemd PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall>
        $<$<AND:${_deepscalemd_opt_configs},$<COMPILE_LANGUAGE:CXX>>:-O3>

        $<$<AND:${_deepscalemd_opt_configs},$<COMPILE_LANGUAGE:CXX>>:-fno-math-errno>
        $<$<AND:${_deepscalemd_opt_configs},$<COMPILE_LANGUAGE:CXX>>:-fno-trapping-math>)
    if(NATIVE_OPTIMIZATION)
        set(DEEPSCALEMD_TARGET_ARCH "native" CACHE STRING
            "-march/-mtune value for NATIVE_OPTIMIZATION (default: native, i.e. the build machine's own CPU -- override on HPC systems where the build node differs from the compute node, e.g. skylake-avx512, znver3, x86-64-v3)")
        target_compile_options(deepscalemd PRIVATE
            $<$<AND:${_deepscalemd_opt_configs},$<COMPILE_LANGUAGE:CXX>>:-march=${DEEPSCALEMD_TARGET_ARCH}>
            $<$<AND:${_deepscalemd_opt_configs},$<COMPILE_LANGUAGE:CXX>>:-mtune=${DEEPSCALEMD_TARGET_ARCH}>)
        message(STATUS "DeepScaleMD: CPU codegen target = ${DEEPSCALEMD_TARGET_ARCH} "
                        "(applies to Release/RelWithDebInfo only; set NATIVE_OPTIMIZATION=OFF to disable, "
                        "or -DDEEPSCALEMD_TARGET_ARCH=<value> to target a different CPU than the build machine)")
    else()
        message(STATUS "DeepScaleMD: NATIVE_OPTIMIZATION=OFF -- building for the compiler's "
                        "generic baseline ISA (no AVX2/AVX-512/FMA). The CPU nonbonded kernels "
                        "in force_calculator_cpu.cpp have hand-written AVX2 paths that fall back "
                        "to scalar code without it; expect substantially slower LJ/PME real-space "
                        "performance than the same run built with NATIVE_OPTIMIZATION=ON.")
    endif()


    option(ENABLE_LTO "Enable interprocedural (link-time) optimization for Release/RelWithDebInfo builds" OFF)
    if(ENABLE_LTO)
        include(CheckIPOSupported)
        check_ipo_supported(RESULT _deepscalemd_ipo_supported OUTPUT _deepscalemd_ipo_error)
        if(_deepscalemd_ipo_supported)
            set_target_properties(deepscalemd PROPERTIES
                INTERPROCEDURAL_OPTIMIZATION_RELEASE ON
                INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
            message(STATUS "DeepScaleMD: ENABLE_LTO=ON -- link-time optimization enabled for Release/RelWithDebInfo")
        else()
            message(WARNING "DeepScaleMD: ENABLE_LTO=ON but this toolchain does not support IPO/LTO "
                             "(${_deepscalemd_ipo_error}); continuing without it.")
        endif()
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
        PASS_REGULAR_EXPRESSION "Nonbonded potential[ ]*: Mie [(]12, 6[)]")

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
        PASS_REGULAR_EXPRESSION "Nonbonded potential[ ]*: Mie [(]9, 6[)]")

    add_test(
        NAME dsMD_input_harmonic_angle_force_sign
        COMMAND deepscalemd "${CMAKE_CURRENT_SOURCE_DIR}/examples/dsMD_input_angle/input.yaml")
    set_tests_properties(dsMD_input_harmonic_angle_force_sign PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        PASS_REGULAR_EXPRESSION "Step: 1, KE: [^,]+, PE: 13[.]6")

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
        PASS_REGULAR_EXPRESSION "Reinitialized post-minimization velocities at T = 300([.]000)? K")

    add_test(
        NAME dsMD_input_xyz_dcd_output
        COMMAND "${CMAKE_COMMAND}"
            "-DDSMD_EXECUTABLE=$<TARGET_FILE:deepscalemd>"
            "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
            "-DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestTrajectoryOutput.cmake")

    if(ENABLE_HDF5)
        add_test(
            NAME dsMD_input_h5md_output
            COMMAND "${CMAKE_COMMAND}"
                "-DDSMD_EXECUTABLE=$<TARGET_FILE:deepscalemd>"
                "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}"
                "-DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestTrajectoryOutputH5MD.cmake")
    endif()
endif()

message(STATUS "DeepScaleMD backend: ${BACKEND}")
message(STATUS "dsDecompX CUDA support: ${dsDecompX_WITH_CUDA}")
message(STATUS "dsDecompX NCCL support: ${dsDecompX_WITH_NCCL}")
message(STATUS "dsDecompX NVSHMEM support: ${dsDecompX_WITH_NVSHMEM}")
