#include "LibTorchBackend.h"

#include "AOTIRunner.h"
#include "OpLibLoader.h"
#include "PythonModules.h"
#include "TorchModelRunner.h"
#include "TorchScriptRunner.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/version.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// 1 eV in kcal/mol; the same value as EV_TO_KCAL_PER_MOL in mlip_python/mlip_interface.py, so results of the
// python and libtorch backends compare directly (design A.6).
constexpr double kEvToKcalPerMol = 23.0609;

constexpr int kWarmupCalls = 3;   // TorchScript profiles the first calls and optimizes the later ones

std::string join_ints(const std::vector<int>& values) {
    std::string text;
    for (int v : values) text += (text.empty() ? "" : ", ") + std::to_string(v);
    return text;
}

std::string join_strings(const std::vector<std::string>& values) {
    std::string text;
    for (const std::string& v : values) text += (text.empty() ? "" : ", ") + v;
    return text;
}

// An AOTI package holds kernels compiled for one GPU architecture (metadata target_gpu): refuse one built for the CPU or for
// another architecture before loading it, with the reason, instead of failing inside the compiled code.
void check_aoti_target(const MLIPMetadata& meta, const torch::Device& device) {
    if (meta.target_gpu == "cpu") {
        throw std::runtime_error("LibTorchBackend: model '" + meta.model_name + "' is an AOTI package compiled for the CPU "
                                 "(a test package); the MLIP backend runs on the GPU: convert it again with --device cuda");
    }
    const cudaDeviceProp& props = *at::cuda::getDeviceProperties(device.index());
    const std::string here = "sm_" + std::to_string(props.major) + std::to_string(props.minor);
    if (meta.target_gpu != here) {
        throw std::runtime_error("LibTorchBackend: model '" + meta.model_name + "' is an AOTI package compiled for " +
                                 meta.target_gpu + ", but this GPU (" + props.name + ") is " + here +
                                 "; AOTI packages are specific to a GPU architecture: convert the model again on a node "
                                 "with this GPU");
    }
}

}  // namespace

struct LibTorchBackend::Impl {
    MLIPMetadata metadata;
    std::unique_ptr<TorchModelRunner> runner;
    torch::Device device{torch::kCUDA, 0};
    torch::ScalarType dtype = torch::kFloat64;
    double energy_to_md = kEvToKcalPerMol;   // model energy unit -> kcal/mol
    torch::Tensor type_to_z;                 // int64 [number of MD types] on the device

    // Runs the model on tensors already on the device (shared by compute() and the warm-up).
    // edge_shift is used only by a model with edge shifts (metadata edge_shifts).
    std::vector<torch::Tensor> run(const torch::Tensor& pos, const torch::Tensor& atomic_numbers,
                                   const torch::Tensor& edge_index, const torch::Tensor& edge_shift, int64_t n_owned) {
        if (metadata.edge_shifts) {
            return runner->run({pos, atomic_numbers, edge_index, edge_shift, torch::tensor(n_owned, torch::kInt64)});
        }
        return runner->run({pos, atomic_numbers, edge_index, torch::tensor(n_owned, torch::kInt64)});
    }

    void warm_up() {
        const auto real = torch::TensorOptions().device(device).dtype(dtype);
        const auto integer = torch::TensorOptions().device(device).dtype(torch::kInt64);
        torch::Tensor pos = torch::zeros({2, 3}, real);
        pos[1][0] = 0.5 * metadata.cutoff;
        torch::Tensor z = torch::full({2}, static_cast<int64_t>(metadata.elements.front()), integer);
        torch::Tensor edges = torch::tensor(std::vector<int64_t>{0, 1, 1, 0}, torch::kInt64).reshape({2, 2}).to(device);
        torch::Tensor shift = torch::zeros({2, 3}, real);
        torch::NoGradGuard no_grad;
        for (int call = 0; call < kWarmupCalls; ++call) {
            std::vector<torch::Tensor> out = run(pos, z, edges, shift, 1);
            if (out.size() != (metadata.edge_shifts ? 4u : 3u) || out[2].sizes() != torch::IntArrayRef({2, 3})) {
                throw std::runtime_error("LibTorchBackend: the model returned unexpected shapes on the warm-up input");
            }
        }
        torch::cuda::synchronize(device.index());   // surface asynchronous CUDA errors now, not in step 1
    }
};

LibTorchBackend::LibTorchBackend(const MLIPConfig& cfg) : impl_(std::make_unique<Impl>()) {
    Impl& s = *impl_;
    if (cfg.model_path.empty()) {
        throw std::runtime_error("LibTorchBackend: no model folder given (mlip_model)");
    }

    // 1. metadata.json
    s.metadata = load_mlip_metadata(cfg.model_path);
    const MLIPMetadata& meta = s.metadata;
    s.dtype = meta.dtype == "float32" ? torch::kFloat32 : torch::kFloat64;
    s.energy_to_md = meta.energy_unit == "eV" ? kEvToKcalPerMol : 1.0;   // "kcal/mol": already MD units

    // 6. Every MD type must map to an element the model knows. Checked before loading anything heavy.
    if (cfg.type_to_z.empty()) {
        throw std::runtime_error(
            "LibTorchBackend: the element of every MD atom type is required (mlip_type_to_z; dsMD fills it in from the "
            "coordinate file, the masses or the input file's mlip_elements: section); the "
            "model supports atomic numbers " + join_ints(meta.elements));
    }
    std::vector<int64_t> table;
    for (size_t type = 0; type < cfg.type_to_z.size(); ++type) {
        const int z = cfg.type_to_z[type];
        if (std::find(meta.elements.begin(), meta.elements.end(), z) == meta.elements.end()) {
            throw std::runtime_error("LibTorchBackend: MD atom type " + std::to_string(type) + " maps to atomic number " +
                                     std::to_string(z) + " (mlip_type_to_z), which model '" + meta.model_name +
                                     "' does not support; it supports " + join_ints(meta.elements));
        }
        table.push_back(z);
    }

    if (!torch::cuda::is_available()) {
        throw std::runtime_error("LibTorchBackend: LibTorch sees no CUDA device");
    }
    s.device = torch::Device(torch::kCUDA, c10::cuda::current_device());
    s.type_to_z = torch::tensor(table, torch::kInt64).to(s.device);

    // 2b. versions: what the model was converted with against what runs here (design 7.4)
    {
        int cuda_version = 0;
        cudaRuntimeGetVersion(&cuda_version);   // e.g. 11080 for CUDA 11.8
        const std::string cuda_text = std::to_string(cuda_version / 1000) + "." + std::to_string((cuda_version % 1000) / 10);
        const VersionReport versions = check_runtime_versions(meta, TORCH_VERSION, cuda_text);
        for (const std::string& warning : versions.warnings) std::cerr << "LibTorchBackend: warning: " << warning << std::endl;
        if (!versions.errors.empty()) {
            std::string message = "LibTorchBackend:";
            for (const std::string& error : versions.errors) message += " " + error + ".";
            throw std::runtime_error(message);
        }
    }

    // 3. operators the model calls, before it is loaded. cuEquivariance registers its operators from Python (design 11.3,
    // A.11-9): importing its module in the embedded interpreter is all the Python this backend runs; the model itself is
    // executed by LibTorch. The libraries (OpenEquivariance: a plain operator library) are dlopened after that.
    if (!meta.required_python_modules.empty()) {
        std::string python_torch;
        try {
            python_torch = import_python_modules(meta.required_python_modules);
        } catch (const std::exception& e) {
            throw std::runtime_error("LibTorchBackend: model '" + meta.model_name + "' needs the Python modules " +
                                     join_strings(meta.required_python_modules) + " imported before it can be loaded: " + e.what());
        }
        // "2.7.1+cu118" against TORCH_VERSION "2.7.1": two torch builds in one process would crash later, so stop here.
        if (python_torch.substr(0, python_torch.find('+')) != TORCH_VERSION) {
            throw std::runtime_error("LibTorchBackend: the embedded Python imports torch " + python_torch + ", but this program is "
                                     "linked with LibTorch " + TORCH_VERSION + "; activate the environment with the same torch "
                                     "(the one the model was converted in), or put its site-packages on PYTHONPATH");
        }
    }
    load_op_libs(meta.required_op_libs);

    // 4. the runner for this file format
    if (meta.backend == "torchscript") {
        s.runner = std::make_unique<TorchScriptRunner>(
            (std::filesystem::path(cfg.model_path) / "model.pt").string(), s.device);
    } else if (meta.backend == "aoti") {
        check_aoti_target(meta, s.device);
        s.runner = std::make_unique<AOTIRunner>(
            (std::filesystem::path(cfg.model_path) / "model.pt2").string(), s.device);
    } else {
        throw std::runtime_error("LibTorchBackend: backend '" + meta.backend + "' is not implemented");
    }

    // 7. warm-up
    s.warm_up();

    if (cfg.verbose) {
        std::cout << "LibTorchBackend: model " << meta.model_name << " (" << meta.model_family << ", " << meta.backend
                  << "), cutoff " << meta.cutoff << " A, ghost cutoff " << meta.ghost_cutoff << " A, " << meta.dtype
                  << " on " << s.device << std::endl;
    }
}

LibTorchBackend::~LibTorchBackend() = default;

double LibTorchBackend::cutoff() const { return impl_->metadata.cutoff; }
double LibTorchBackend::ghost_cutoff() const { return impl_->metadata.ghost_cutoff; }
bool LibTorchBackend::uses_edge_shifts() const { return impl_->metadata.edge_shifts; }
const MLIPMetadata& LibTorchBackend::metadata() const { return impl_->metadata; }

void LibTorchBackend::compute(const MLIPInputs& in, MLIPOutputs& out) {
    Impl& s = *impl_;
    if (in.host_memory) {
        throw std::runtime_error("LibTorchBackend::compute needs device memory (MLIPInputs::host_memory is set)");
    }
    const int64_t n = in.n_all;
    if (n <= 0) {
        if (out.need_energy) out.energy = 0.0;
        return;
    }
    const int64_t n_edges = in.n_edges;

    // Work on the MD stream (design 8): our tensor ops and the model's kernels queue behind the MD kernels that
    // produced the positions and ahead of whatever the MD code queues next.
    const c10::DeviceIndex device_index = s.device.index();
    c10::cuda::CUDAStream stream = in.stream ? c10::cuda::getStreamFromExternal(in.stream, device_index)
                                             : c10::cuda::getDefaultCUDAStream(device_index);
    c10::cuda::CUDAStreamGuard stream_guard(stream);

    // Our own operations record no graph; the wrapper inside the model turns grad mode on for the force.
    torch::NoGradGuard no_grad;

    const auto real64 = torch::TensorOptions().device(s.device).dtype(torch::kFloat64);
    const auto int32 = torch::TensorOptions().device(s.device).dtype(torch::kInt32);
    const auto int64 = torch::TensorOptions().device(s.device).dtype(torch::kInt64);

    // MD arrays are SoA double; the model wants [N, 3] in its dtype. Wrapping is free; stack copies on the GPU.
    torch::Tensor x = torch::from_blob(const_cast<double*>(in.d_x), {n}, real64);
    torch::Tensor y = torch::from_blob(const_cast<double*>(in.d_y), {n}, real64);
    torch::Tensor z = torch::from_blob(const_cast<double*>(in.d_z), {n}, real64);
    torch::Tensor pos = torch::stack({x, y, z}, 1).to(s.dtype);

    torch::Tensor md_types = torch::from_blob(const_cast<int*>(in.d_type), {n}, int32);
    torch::Tensor atomic_numbers = s.type_to_z.index_select(0, md_types.to(torch::kInt64));

    torch::Tensor edge_index = n_edges > 0
        ? torch::from_blob(const_cast<int64_t*>(in.d_edge_index), {2, n_edges}, int64)
        : torch::empty({2, 0}, int64);

    torch::Tensor edge_shift;
    if (s.metadata.edge_shifts) {
        // [E, 3] in the model's dtype; no shifts given = every edge is between atoms that are really there
        const auto real_model = torch::TensorOptions().device(s.device).dtype(s.dtype);
        edge_shift = (in.d_edge_shift && n_edges > 0)
            ? torch::from_blob(const_cast<double*>(in.d_edge_shift), {n_edges, 3}, real64).to(s.dtype)
            : torch::zeros({n_edges, 3}, real_model);
    }

    std::vector<torch::Tensor> result = s.run(pos, atomic_numbers, edge_index, edge_shift, in.n_owned);
    const torch::Tensor& energy = result[0];
    const torch::Tensor& atom_energy = result[1];
    const torch::Tensor& forces = result[2];
    if (atom_energy.numel() != n || forces.sizes() != torch::IntArrayRef({n, 3})) {
        throw std::runtime_error("LibTorchBackend: the model returned arrays that do not match the " +
                                 std::to_string(n) + " atoms it was given");
    }

    // Forces in MD units, added to the MD force arrays in place (ghost atoms included, for the reverse comm).
    torch::Tensor md_forces = forces.to(torch::kFloat64);
    double* out_ptrs[3] = {out.d_fx, out.d_fy, out.d_fz};
    for (int axis = 0; axis < 3; ++axis) {
        torch::from_blob(out_ptrs[axis], {n}, real64).add_(md_forces.select(1, axis), s.energy_to_md);
    }
    if (out.d_atom_energy) {
        torch::from_blob(out.d_atom_energy, {n}, real64).copy_(atom_energy.to(torch::kFloat64) * s.energy_to_md);
    }

    // The only host synchronization, and only when the caller wants the energy this step.
    if (out.need_energy) {
        out.energy = energy.item<double>() * s.energy_to_md;
        if (s.metadata.edge_shifts) {
            out.virial = result[3].item<double>() * s.energy_to_md;
            out.has_virial = true;
        }
    }
}
