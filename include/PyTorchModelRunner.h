#ifndef PYTORCH_MODEL_RUNNER_H
#define PYTORCH_MODEL_RUNNER_H

#include "TorchModelRunner.h"

#include <string>

// Runs a model folder converted with --backend pytorch (metadata "backend": "pytorch"): there is no model file. The embedded
// Python builds the converter's wrapper again from metadata.json (MLIP_tools/mlip_convert/pytorch_runner.py, the same code the converter
// uses) and runs it every step, without TorchScript. Everything around the call (inputs, ghost atoms, layer exchange through
// dsmd::halo_exchange, adding the forces) is LibTorchBackend's, as for a TorchScript model.
//
// The tensors go to Python and back as DLPack capsules: the MD's GPU arrays are shared, not copied, and Python runs on the CUDA
// stream LibTorchBackend set (the current stream is shared by both sides of the process).
//
// Needs a build with the embedded Python (-DENABLE_PYTHON=ON) and an interpreter already started with the activated environment
// (LibTorchBackend imports torch through import_python_modules first and checks its version against LibTorch).
class PyTorchModelRunner : public TorchModelRunner {
public:
    // python_path: the folder that holds the mlip_convert package (metadata python_path), put at the front of sys.path.
    PyTorchModelRunner(const std::string& model_folder, const std::string& python_path, const torch::Device& device);
    ~PyTorchModelRunner() override;

    std::vector<torch::Tensor> run(const std::vector<torch::Tensor>& inputs) override;

private:
    std::string model_folder_;
    void* run_method_ = nullptr;   // PyObject*: the bound method PyTorchModel.run (no Python header here)
};

#endif // PYTORCH_MODEL_RUNNER_H
