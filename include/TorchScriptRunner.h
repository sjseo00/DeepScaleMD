#ifndef TORCH_SCRIPT_RUNNER_H
#define TORCH_SCRIPT_RUNNER_H

#include "TorchModelRunner.h"

#include <torch/script.h>

#include <string>

// Runs a TorchScript wrapper saved by the converter (mlip_convert, export_torchscript.py).
class TorchScriptRunner : public TorchModelRunner {
public:
    // Loads model_file onto `device` and puts it in eval mode. Throws std::runtime_error naming the file when it is
    // missing or cannot be loaded; a missing custom operator library is called out (see OpLibLoader.h).
    TorchScriptRunner(const std::string& model_file, const torch::Device& device);

    std::vector<torch::Tensor> run(const std::vector<torch::Tensor>& inputs) override;

private:
    std::string model_file_;
    torch::jit::Module module_;
};

#endif // TORCH_SCRIPT_RUNNER_H
