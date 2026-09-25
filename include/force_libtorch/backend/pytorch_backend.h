
#pragma once
#ifndef MLIP_PYTORCH_BACKEND_H
#define MLIP_PYTORCH_BACKEND_H

#include "force_libtorch/backend/backend_interface.h"
#include <memory>
#include <string>

namespace mlip {

class PyTorchBackend : public IMlipBackend {
public:
    PyTorchBackend();
    ~PyTorchBackend() override;

    void initialize(
        const BackendConfiguration& config,
        const MlipExecutionContext& context
    ) override;

    BackendModelMetadata inspect_model() override;

    void execute(
        const std::vector<BackendTensorView>& inputs,
        std::vector<BackendTensorView>&       outputs,
        cudaStream_t                          stream
    ) override;

    void synchronize() override;
    const char* name() const override { return "pytorch"; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    BackendConfiguration  cfg_;
    MlipExecutionContext  ctx_;
};

} // namespace mlip
#endif // MLIP_PYTORCH_BACKEND_H
