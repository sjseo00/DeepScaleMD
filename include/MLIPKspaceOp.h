#ifndef MLIP_KSPACE_OP_H
#define MLIP_KSPACE_OP_H

// The LibTorch operator dsmd::kspace(Tensor pos, Tensor charges, float[] params) -> Tensor (src/MLIPKspaceOp.cpp): smooth PME of the
// per-atom charges a model computes (so3lr --kspace pme). Same schema and formulas as its Python version, which is used while a model
// is built and checked: MLIP_tools/mlip_convert/kspace_op.py (see there for the formulas). The box, grid settings and grid sum come
// from the MLIPKspaceContext of the innermost KspaceScope; without one the operator refuses to run.
// Call ensure_kspace_op_registered() once (from LibTorchBackend) so that a static link keeps the file. No LibTorch header here.

#include "MLIPKspace.h"

void ensure_kspace_op_registered();

class KspaceScope {
public:
    explicit KspaceScope(MLIPKspaceContext* context);
    ~KspaceScope();
    KspaceScope(const KspaceScope&) = delete;
    KspaceScope& operator=(const KspaceScope&) = delete;

private:
    MLIPKspaceContext* previous_;
};

#endif // MLIP_KSPACE_OP_H
