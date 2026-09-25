
#include "force_libtorch/mlip_types.h"
#include "force_libtorch/backend/backend_interface.h"
#include <memory>

extern "C" mlip::IMlipBackend* mlip_create_pytorch_backend_c();

namespace {
    static mlip::BackendAutoRegister _reg_pytorch_backend(
        "pytorch",
        []() -> std::unique_ptr<mlip::IMlipBackend> {
            return std::unique_ptr<mlip::IMlipBackend>(
                mlip_create_pytorch_backend_c());
        });
} // anonymous namespace
