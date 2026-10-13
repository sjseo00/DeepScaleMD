#include "TorchScriptRunner.h"

#include <filesystem>
#include <stdexcept>

namespace {

// Errors LibTorch gives when a model uses an operator that is not registered (design 6.1).
bool looks_like_missing_op(const std::string& message) {
    return message.find("Unknown builtin op") != std::string::npos ||
           message.find("Could not find schema") != std::string::npos ||
           message.find("Couldn't find an operator") != std::string::npos;
}

}  // namespace

TorchScriptRunner::TorchScriptRunner(const std::string& model_file, const torch::Device& device)
    : model_file_(model_file) {
    if (!std::filesystem::is_regular_file(model_file)) {
        throw std::runtime_error("TorchScriptRunner: model file not found: " + model_file);
    }
    try {
        module_ = torch::jit::load(model_file, device);
        module_.eval();
    } catch (const c10::Error& error) {
        std::string message = "TorchScriptRunner: cannot load '" + model_file + "': " + error.msg();
        if (looks_like_missing_op(error.msg())) {
            message += "\n(the model uses a custom operator that is not registered: list its library in "
                       "required_op_libs of metadata.json so it is loaded first)";
        }
        throw std::runtime_error(message);
    }
}

std::vector<torch::Tensor> TorchScriptRunner::run(const std::vector<torch::Tensor>& inputs) {
    if (inputs.size() != 4) {
        throw std::invalid_argument("TorchScriptRunner::run expects {pos, atomic_numbers, edge_index, n_owned}");
    }
    std::vector<c10::IValue> arguments(inputs.begin(), inputs.end());

    c10::IValue result;
    try {
        result = module_.forward(std::move(arguments));
    } catch (const c10::Error& error) {
        throw std::runtime_error("TorchScriptRunner: forward of '" + model_file_ + "' failed: " + error.msg());
    }

    if (!result.isTuple() || result.toTuple()->elements().size() != 3) {
        throw std::runtime_error("TorchScriptRunner: '" + model_file_ +
                                 "' must return (energy, atom_energy, forces)");
    }
    std::vector<torch::Tensor> outputs;
    for (const c10::IValue& element : result.toTuple()->elements()) {
        if (!element.isTensor()) {
            throw std::runtime_error("TorchScriptRunner: '" + model_file_ + "' returned a non-tensor output");
        }
        outputs.push_back(element.toTensor());
    }
    return outputs;
}
