
#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <sstream>

#define MLIP_CUDA_CHECK(expr)                                               \
    do {                                                                    \
        cudaError_t _e = (expr);                                            \
        if (_e != cudaSuccess) {                                            \
            std::ostringstream _ss;                                         \
            _ss << "[MLIP CUDA] error at " << __FILE__ << ":" << __LINE__  \
                << " — " << cudaGetErrorString(_e);                         \
            throw std::runtime_error(_ss.str());                            \
        }                                                                   \
    } while (false)

static constexpr int BLOCK_SIZE = 256;

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
