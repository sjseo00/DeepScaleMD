
#include "force_libtorch/mlip_types.h"
#include "force_libtorch/runtime/mlip_runtime.h"

#include <stdexcept>
#include <string>

namespace mlip {

MlipScratchAllocator::~MlipScratchAllocator() {}

MlipRuntime::~MlipRuntime() {}

void MlipRuntime::initialize(const std::string& /*spec_file*/,
                              const MlipExecutionContext& /*ctx*/)
{
    throw std::runtime_error(
        "[MlipRuntime] STUB: the full C++ MLIP runtime (libtorch mode) is not "
        "compiled in this build.  Use force_calculation.mode: python instead.");
}

void MlipRuntime::compute(const MlipInputView& /*in*/,
                           MlipOutputView&       /*out*/)
{
    throw std::runtime_error("[MlipRuntime] STUB: not available.");
}

double MlipRuntime::cutoff() const { return 0.0; }

} // namespace mlip
