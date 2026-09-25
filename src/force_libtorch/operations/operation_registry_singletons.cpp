
#include "force_libtorch/operations/operation_registry.h"
#include "force_libtorch/backend/backend_interface.h"

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
