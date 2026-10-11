cmake_policy(SET CMP0053 NEW)

if(NOT DEFINED DSMD_EXECUTABLE OR NOT DEFINED SOURCE_DIR OR NOT DEFINED BINARY_DIR)
    message(FATAL_ERROR "DSMD_EXECUTABLE, SOURCE_DIR, and BINARY_DIR are required")
endif()

set(OUTPUT_PREFIX "${BINARY_DIR}/trajectory_h5md_test")
set(CONFIG_FILE "${BINARY_DIR}/trajectory_h5md_test.yaml")
file(READ "${SOURCE_DIR}/examples/dsMD_input_angle/input_trajectory_h5md.yaml.in" config_text)
string(REPLACE "@SOURCE_DIR@" "${SOURCE_DIR}" config_text "${config_text}")
string(REPLACE "@OUTPUT_PREFIX@" "${OUTPUT_PREFIX}" config_text "${config_text}")
file(WRITE "${CONFIG_FILE}" "${config_text}")

file(REMOVE
    "${OUTPUT_PREFIX}_0.xyz" "${OUTPUT_PREFIX}_1.xyz" "${OUTPUT_PREFIX}_2.xyz"
    "${OUTPUT_PREFIX}.h5")

execute_process(
    COMMAND "${DSMD_EXECUTABLE}" "${CONFIG_FILE}"
    WORKING_DIRECTORY "${SOURCE_DIR}"
    RESULT_VARIABLE run_result
    OUTPUT_VARIABLE run_stdout
    ERROR_VARIABLE run_stderr)
if(NOT run_result EQUAL 0)
    message(FATAL_ERROR "dsMD H5MD trajectory run failed:\n${run_stdout}\n${run_stderr}")
endif()

if(NOT EXISTS "${OUTPUT_PREFIX}.h5")
    message(FATAL_ERROR "missing H5MD trajectory file ${OUTPUT_PREFIX}.h5")
endif()

# h5dump is optional tooling -- skip the structural check rather than force
# every contributor to install the HDF5 command-line tools to run tests.
find_program(H5DUMP_EXECUTABLE h5dump)
if(H5DUMP_EXECUTABLE)
    execute_process(
        COMMAND "${H5DUMP_EXECUTABLE}" -H "${OUTPUT_PREFIX}.h5"
        RESULT_VARIABLE h5dump_result
        OUTPUT_VARIABLE h5dump_stdout
        ERROR_VARIABLE h5dump_stderr)
    if(NOT h5dump_result EQUAL 0)
        message(FATAL_ERROR "h5dump failed on ${OUTPUT_PREFIX}.h5:\n${h5dump_stdout}\n${h5dump_stderr}")
    endif()

    if(NOT h5dump_stdout MATCHES "DATASET \"value\"")
        message(FATAL_ERROR "H5MD file is missing expected 'value' datasets:\n${h5dump_stdout}")
    endif()
    # 3 output steps (0, 1, 2) at trajectory_output_freq: 1 -> 3 frames along
    # the unlimited leading dimension of /particles/all/position/value.
    if(NOT h5dump_stdout MATCHES "DATASPACE[ \t]+SIMPLE[ \t]*\\{[ \t]*\\([ \t]*3,")
        message(FATAL_ERROR "H5MD position/value frame count is not 3:\n${h5dump_stdout}")
    endif()

    message(STATUS "H5MD trajectory structural validation passed")
else()
    message(STATUS "h5dump not found; skipping H5MD structural validation (file existence only)")
endif()
