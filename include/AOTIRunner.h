#ifndef AOTI_RUNNER_H
#define AOTI_RUNNER_H

#include "TorchModelRunner.h"

#include <memory>
#include <string>
#include <vector>

namespace torch::inductor {
class AOTIModelPackageLoader;
}

// Runs an AOTInductor package (model.pt2) written by the converter (mlip_convert, export_aoti.py): the standard wrapper with
// the force already unfolded into the compiled graph, no Python and no TorchScript. A package is compiled for one device
// (a GPU architecture, metadata `target_gpu`, or the CPU for tests) and its inputs must be on that device.
class AOTIRunner : public TorchModelRunner {
public:
    // Loads model_file. Throws std::runtime_error naming the file when it is missing or cannot be loaded, or when the
    // package was compiled for another kind of device than `device`.
    AOTIRunner(const std::string& model_file, const torch::Device& device);
    ~AOTIRunner() override;

    // The inputs of TorchModelRunner::run, with n_owned moved to the device of the package (it is compiled as a tensor
    // input like the others).
    std::vector<torch::Tensor> run(const std::vector<torch::Tensor>& inputs) override;

private:
    std::string model_file_;
    torch::Device device_;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader_;
};

#endif // AOTI_RUNNER_H
