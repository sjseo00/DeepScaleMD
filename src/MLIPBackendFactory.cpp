#include "MLIPBackend.h"
#include "PythonBackend.h"
#ifdef DSMD_WITH_LIBTORCH
#include "LibTorchBackend.h"
#endif

#include <stdexcept>
#include <string>

namespace {

std::string effective_backend(const MLIPConfig& cfg) {
    return cfg.backend.empty() ? std::string("python") : cfg.backend;
}

}  // namespace

void validate_mlip_config(const MLIPConfig& cfg) {
    const std::string backend = effective_backend(cfg);
    if (backend == "python") {
        // optional here (Python guesses the elements of the types without it), but an entry that is given has to be an element
        for (int z : cfg.type_to_z) {
            if (z < 1 || z > 118) {
                throw std::runtime_error(
                    "Invalid mlip_type_to_z entry " + std::to_string(z) + ": atomic numbers are 1..118");
            }
        }
        return;
    }
    if (backend != "libtorch") {
        throw std::runtime_error(
            "Invalid mlip_backend: '" + backend + "'. Supported values are: python, libtorch");
    }

#if !defined(USE_CUDA)
    throw std::runtime_error(
        "mlip_backend 'libtorch' is only supported in CUDA builds, not in CPU/SYCL builds. "
        "Use 'python'.");
#elif !defined(DSMD_WITH_LIBTORCH)
    throw std::runtime_error(
        "mlip_backend 'libtorch' needs a build with LibTorch support "
        "(configure with -DBACKEND=CUDA -DENABLE_LIBTORCH=ON). Use 'python' or rebuild.");
#else
    if (cfg.model_path.empty()) {
        throw std::runtime_error(
            "mlip_backend 'libtorch' requires mlip_model: <converted model folder>");
    }
    // mlip_type_to_z may be empty here: the MD reads it from the atom type names of the topology before it creates the backend
    for (int z : cfg.type_to_z) {
        if (z < 1 || z > 118) {
            throw std::runtime_error(
                "Invalid mlip_type_to_z entry " + std::to_string(z) + ": atomic numbers are 1..118");
        }
    }
#endif
}

std::unique_ptr<MLIPBackend> create_mlip_backend(const MLIPConfig& cfg) {
    validate_mlip_config(cfg);
    if (effective_backend(cfg) == "python") {
        return std::make_unique<PythonBackend>(cfg);
    }
    // validate_mlip_config accepts 'libtorch' only in a build with LibTorch support.
#ifdef DSMD_WITH_LIBTORCH
    return std::make_unique<LibTorchBackend>(cfg);
#else
    throw std::runtime_error("mlip_backend 'libtorch' needs a build with LibTorch support");
#endif
}
