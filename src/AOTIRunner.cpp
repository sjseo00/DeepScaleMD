#include "AOTIRunner.h"

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <filesystem>
#include <stdexcept>

AOTIRunner::AOTIRunner(const std::string& model_file, const torch::Device& device)
    : model_file_(model_file), device_(device) {
    if (!std::filesystem::is_regular_file(model_file)) {
        throw std::runtime_error("AOTIRunner: model file not found: " + model_file);
    }
    try {
        loader_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(model_file);
    } catch (const std::exception& error) {
        throw std::runtime_error("AOTIRunner: cannot load '" + model_file + "': " + error.what());
    }

    // The package knows what it was compiled for; running it with inputs on another kind of device fails inside the
    // compiled code with a message that does not say why.
    const auto metadata = loader_->get_metadata();
    const auto key = metadata.find("AOTI_DEVICE_KEY");
    if (key != metadata.end()) {
        const std::string wanted = device_.is_cuda() ? "cuda" : "cpu";
        if (key->second != wanted) {
            throw std::runtime_error("AOTIRunner: '" + model_file + "' was compiled for the " + key->second +
                                     " but is being loaded for the " + wanted +
                                     "; convert it again for this device (python -m mlip_convert ... --device " + wanted + ")");
        }
    }
}

AOTIRunner::~AOTIRunner() = default;

std::vector<torch::Tensor> AOTIRunner::run(const std::vector<torch::Tensor>& inputs) {
    if (inputs.size() != 4) {
        throw std::invalid_argument("AOTIRunner::run expects {pos, atomic_numbers, edge_index, n_owned}");
    }
    std::vector<torch::Tensor> arguments(inputs.begin(), inputs.end());
    arguments[3] = arguments[3].to(device_);

    std::vector<torch::Tensor> outputs;
    try {
        outputs = loader_->run(arguments);
    } catch (const std::exception& error) {
        throw std::runtime_error("AOTIRunner: running '" + model_file_ + "' failed: " + error.what());
    }
    if (outputs.size() != 3) {
        throw std::runtime_error("AOTIRunner: '" + model_file_ + "' must return (energy, atom_energy, forces)");
    }
    return outputs;
}
