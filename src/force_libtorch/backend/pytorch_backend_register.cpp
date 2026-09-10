/**
 * pytorch_backend_register.cpp
 *
 * Compiled with the DEFAULT ABI (ABI=1) so that BackendRegistry::instance()
 * refers to the same singleton used by execution_plan.cpp / mlip_runtime.cpp.
 *
 * pytorch_backend.cpp is compiled with ABI=0 (LibTorch requirement).
 * We bridge the ABI boundary via a C-linkage factory function.
 */

#include "force_libtorch/mlip_types.h"
#include "force_libtorch/backend/backend_interface.h"
#include <memory>

// C-linkage factory defined in pytorch_backend.cpp (compiled with ABI=0).
// The returned IMlipBackend* is a vtable-based interface; virtual dispatch
// works correctly across the ABI boundary because IMlipBackend uses only
// primitive types and does not contain std::string members.
extern "C" mlip::IMlipBackend* mlip_create_pytorch_backend_c();

namespace {
    // Static initializer — registers "pytorch" into the ABI=1 BackendRegistry.
    static mlip::BackendAutoRegister _reg_pytorch_backend(
        "pytorch",
        []() -> std::unique_ptr<mlip::IMlipBackend> {
            return std::unique_ptr<mlip::IMlipBackend>(
                mlip_create_pytorch_backend_c());
        });
} // anonymous namespace
