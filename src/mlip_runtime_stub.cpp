/**
 * mlip_runtime_stub.cpp
 *
 * Minimal stub implementation of mlip::MlipRuntime that satisfies the linker
 * when the full MLIP infrastructure (yaml_parser, execution_plan, backend
 * registry, etc.) is NOT compiled in.
 *
 * This stub is used when building the binary for Python-interface mode
 * (force_calculation.mode: python).  In that mode the mlip_runtime code path in
 * md_simulation.cpp is never entered, so these methods are never called.
 *
 * If a user tries to use mlip_runtime mode with this stub, they will get a clear
 * runtime error message rather than a silent crash.
 */

// Bring in mlip_types first to define ::ScalarType before any torch headers.
#include "mlip/mlip_types.h"
#include "mlip/runtime/mlip_runtime.h"

#include <stdexcept>
#include <string>

namespace mlip {

// Stub destructor for MlipScratchAllocator (normally in execution_plan.cpp).
// This translation unit replaces the full MLIP infrastructure for builds that
// only need the Python-interface mode (force_calculation.mode: python).
MlipScratchAllocator::~MlipScratchAllocator() {}

MlipRuntime::~MlipRuntime() {}

void MlipRuntime::initialize(const std::string& /*spec_file*/,
                              const MlipExecutionContext& /*ctx*/)
{
    throw std::runtime_error(
        "[MlipRuntime] STUB: the full C++ MLIP runtime (mlip_runtime mode) is not "
        "compiled in this build.  Use force_calculation.mode: python instead.");
}

void MlipRuntime::compute(const MlipInputView& /*in*/,
                           MlipOutputView&       /*out*/)
{
    throw std::runtime_error("[MlipRuntime] STUB: not available.");
}

double MlipRuntime::cutoff() const { return 0.0; }

} // namespace mlip
