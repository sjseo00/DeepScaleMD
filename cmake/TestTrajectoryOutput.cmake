cmake_policy(SET CMP0053 NEW)

if(NOT DEFINED DSMD_EXECUTABLE OR NOT DEFINED SOURCE_DIR OR NOT DEFINED BINARY_DIR)
    message(FATAL_ERROR "DSMD_EXECUTABLE, SOURCE_DIR, and BINARY_DIR are required")
endif()

set(OUTPUT_PREFIX "${BINARY_DIR}/trajectory_format_test")
set(CONFIG_FILE "${BINARY_DIR}/trajectory_format_test.yaml")
file(READ "${SOURCE_DIR}/examples/dsMD_input_angle/input_trajectory.yaml.in" config_text)
string(REPLACE "@SOURCE_DIR@" "${SOURCE_DIR}" config_text "${config_text}")
string(REPLACE "@OUTPUT_PREFIX@" "${OUTPUT_PREFIX}" config_text "${config_text}")
file(WRITE "${CONFIG_FILE}" "${config_text}")

file(REMOVE
    "${OUTPUT_PREFIX}_0.xyz" "${OUTPUT_PREFIX}_1.xyz" "${OUTPUT_PREFIX}_2.xyz"
    "${OUTPUT_PREFIX}.dcd")

execute_process(
    COMMAND "${DSMD_EXECUTABLE}" "${CONFIG_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE run_result
    OUTPUT_VARIABLE run_stdout
    ERROR_VARIABLE run_stderr)
if(NOT run_result EQUAL 0)
    message(FATAL_ERROR "dsMD trajectory run failed:\n${run_stdout}\n${run_stderr}")
endif()

foreach(step RANGE 0 2)
    if(NOT EXISTS "${OUTPUT_PREFIX}_${step}.xyz")
        message(FATAL_ERROR "missing XYZ frame ${step}")
    endif()
endforeach()

file(READ "${OUTPUT_PREFIX}_1.xyz" xyz_frame LIMIT 512)
if(NOT xyz_frame MATCHES "^3\nStep 1 Time 0[.]100 fs KE=0[.]022927")
    message(FATAL_ERROR "XYZ frame header or trajectory-only KE is invalid:\n${xyz_frame}")
endif()
if(NOT xyz_frame MATCHES "\nH [0-9.-]+ [0-9.-]+ [0-9.-]+ [0-9.-]+ [0-9.-]+ [0-9.-]+ [0-9.-]+ [0-9.-]+ [0-9.-]+")
    message(FATAL_ERROR "XYZ element/position/velocity/force columns are invalid")
endif()

if(NOT EXISTS "${OUTPUT_PREFIX}.dcd")
    message(FATAL_ERROR "missing DCD trajectory")
endif()
file(SIZE "${OUTPUT_PREFIX}.dcd" dcd_size)
if(NOT dcd_size EQUAL 544)
    message(FATAL_ERROR "DCD size is ${dcd_size}, expected 544")
endif()
file(READ "${OUTPUT_PREFIX}.dcd" dcd_hex LIMIT 196 HEX)
string(SUBSTRING "${dcd_hex}" 8 8 dcd_magic)
string(SUBSTRING "${dcd_hex}" 16 8 dcd_nset)
string(SUBSTRING "${dcd_hex}" 376 8 dcd_natom)
if(NOT dcd_magic STREQUAL "434f5244")
    message(FATAL_ERROR "DCD magic is invalid: ${dcd_magic}")
endif()
if(NOT dcd_nset STREQUAL "03000000")
    message(FATAL_ERROR "DCD NSET is invalid: ${dcd_nset}")
endif()
if(NOT dcd_natom STREQUAL "03000000")
    message(FATAL_ERROR "DCD NATOM is invalid: ${dcd_natom}")
endif()

message(STATUS "XYZ and DCD trajectory validation passed")
