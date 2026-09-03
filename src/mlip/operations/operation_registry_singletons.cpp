/**
 * operation_registry_singletons.cpp
 *
 * Defines the out-of-line singleton instance() functions for the MLIP
 * operation registries and backend registry.
 *
 * These MUST be defined in exactly ONE translation unit.  Keeping them
 * inline in the header caused NVCC to emit multiple copies of the
 * function-local static (one per .cu/.cpp TU), which led to duplicate
 * singleton initialisation and heap corruption.
 */

#include "mlip/operations/operation_registry.h"
#include "mlip/backend/backend_interface.h"

namespace mlip {

PreprocessRegistry& PreprocessRegistry::instance() {
    static PreprocessRegistry r; return r;
}

PostprocessRegistry& PostprocessRegistry::instance() {
    static PostprocessRegistry r; return r;
}

SourceRegistry& SourceRegistry::instance() {
    static SourceRegistry r; return r;
}

TargetRegistry& TargetRegistry::instance() {
    static TargetRegistry r; return r;
}

BackendRegistry& BackendRegistry::instance() {
    static BackendRegistry r; return r;
}

} // namespace mlip
