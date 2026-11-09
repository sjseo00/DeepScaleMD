#include "LibTorchBackend.h"

#include "AOTIRunner.h"
#include "HaloExchangeOp.h"
#include "MLIPKspaceOp.h"
#include "OpLibLoader.h"
#include "PyTorchModelRunner.h"
#include "PythonModules.h"
#include "TorchModelRunner.h"
#include "TorchScriptRunner.h"

#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/version.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// 1 eV in kcal/mol; the same value as EV_TO_KCAL_PER_MOL in MLIP_tools/mlip_python/mlip_interface.py, so results of the
// python and libtorch backends compare directly (design A.6).
constexpr double kEvToKcalPerMol = 23.0605;

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

// cuEquivariance (cuequivariance_ops 0.13) times dozens of uniform_1d kernel variants for every new problem size and keeps
// the fastest. Under domain decomposition the ghost count of a rank changes at every neighbor rebuild, so it tunes again and
// again: water 24000 atoms on 4 GPUs ran at 154 ms/step instead of 66 ms/step (benchmarks/mlip/mace/modes/u1d_algos.sh).
// CUEQUIVARIANCE_OPS_U1D_ALGOS (read by libcue_ops.so, not documented) fixes the choice; 0 was the fastest on 1 and 4 GPUs and
// the energies are unchanged. A value already set in the environment is kept. Must run before the library is loaded.
// Returns the value in effect.
std::string fix_cueq_kernel_choice() {
    static const char* const name = "CUEQUIVARIANCE_OPS_U1D_ALGOS";
    if (const char* value = std::getenv(name)) return value;
    setenv(name, "0", 0);
    return "0";
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
        // A layer-exchange model takes the atoms after n_owned as ghost atoms and exchanges their features: both atoms are owned here.
        const int64_t n_owned = metadata.layer_exchange ? 2 : 1;
        torch::Tensor pos = torch::zeros({2, 3}, real);
        pos[1][0] = 0.5 * metadata.cutoff;
        torch::Tensor z = torch::full({2}, static_cast<int64_t>(metadata.elements.front()), integer);
        torch::Tensor edges = torch::tensor(std::vector<int64_t>{0, 1, 1, 0}, torch::kInt64).reshape({2, 2}).to(device);
        torch::Tensor shift = torch::zeros({2, 3}, real);
        torch::NoGradGuard no_grad;
        // A model with k-space terms needs a box and grid for dsmd::kspace: a periodic box around the two atoms, the model's grid.
        MLIPKspaceContext warm_kspace;
        for (int a = 0; a < 3; ++a) warm_kspace.box[a] = 2.0 * metadata.cutoff + 10.0;
        warm_kspace.grid_spacing = metadata.kspace.grid_spacing;
        warm_kspace.dispersion_grid_spacing = metadata.kspace.dispersion_grid_spacing;
        warm_kspace.order = metadata.kspace.order;
        KspaceScope kspace_scope(metadata.kspace.enabled ? &warm_kspace : nullptr);
        for (int call = 0; call < kWarmupCalls; ++call) {
            std::vector<torch::Tensor> out = run(pos, z, edges, shift, n_owned);
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
    // mlip_backend says how the model is run, and the folder has to be one for it (the yaml alone then tells the way).
    if (cfg.backend == "pytorch" && meta.backend != "pytorch") {
        throw std::runtime_error("LibTorchBackend: mlip_backend: pytorch needs a model folder converted with --backend pytorch, but '" +
                                 cfg.model_path + "' was converted with --backend " + meta.backend +
                                 ". Use mlip_backend: libtorch for it, or convert the model again with --backend pytorch");
    }
    if (cfg.backend == "libtorch" && meta.backend == "pytorch") {
        throw std::runtime_error("LibTorchBackend: mlip_backend: libtorch runs a TorchScript or AOTI file, but '" + cfg.model_path +
                                 "' was converted with --backend pytorch (no model file). Use mlip_backend: pytorch for it, or "
                                 "convert the model again with --backend torchscript");
    }
    s.dtype = meta.dtype == "float32" ? torch::kFloat32 : torch::kFloat64;
    s.energy_to_md = meta.energy_unit == "eV" ? kEvToKcalPerMol : 1.0;   // "kcal/mol": already MD units
    s.metadata.kspace.energy_to_md = s.energy_to_md;                      // the k-space virial correction is in model units

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

    // float32 matrix products on the TF32 tensor cores (advanced.tf32): about 3x the FP32 rate on an A100, with TF32's 10 mantissa bits
    // in the products (the sums stay float32). Global for the process; kernels of cuEquivariance / OpenEquivariance have their own
    // arithmetic and are not affected.
    if (cfg.allow_tf32) {
        const cudaDeviceProp& props = *at::cuda::getDeviceProperties(s.device.index());
        at::globalContext().setFloat32MatmulPrecision("high");
        if (props.major < 8) {
            std::cerr << "LibTorchBackend: warning: advanced.tf32 is set, but " << props.name << " (sm_" << props.major << props.minor
                      << ") has no TF32 tensor cores: nothing changes" << std::endl;
        } else if (cfg.verbose) {
            std::cerr << "[MLIP] advanced.tf32: float32 matrix products run in TF32 on " << props.name << " (the forces carry TF32 "
                      << "rounding: check step 0 against a run without it)" << std::endl;
        }
    }

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
    // A.11-9): importing its module in the embedded interpreter is all the Python this backend runs for a TorchScript or AOTI
    // model; the model itself is executed by LibTorch. The libraries (OpenEquivariance: a plain operator library) are dlopened
    // after that. A "pytorch" folder is run by the embedded Python itself: the interpreter is started here too, and its torch
    // checked against the LibTorch of this program.
    if (meta.accel == "cueq") {
        const bool user_set = std::getenv("CUEQUIVARIANCE_OPS_U1D_ALGOS") != nullptr;
        const std::string algos = fix_cueq_kernel_choice();
        // on stderr: std::cout is silenced during start-up
        if (cfg.verbose) {
            std::cerr << "LibTorchBackend: cuEquivariance kernel choice CUEQUIVARIANCE_OPS_U1D_ALGOS=" << algos
                      << (user_set ? " (set in the environment)" : " (dsMD default)") << std::endl;
        }
    }
    if (!meta.required_python_modules.empty() || meta.backend == "pytorch") {
        std::string python_torch;
        try {
            python_torch = import_python_modules(meta.required_python_modules);
        } catch (const std::exception& e) {
            throw std::runtime_error("LibTorchBackend: model '" + meta.model_name + "' needs the Python modules torch" +
                                     (meta.required_python_modules.empty() ? "" : ", " + join_strings(meta.required_python_modules)) +
                                     " imported before it can be loaded: " + e.what());
        }
        // "2.7.1+cu118" against TORCH_VERSION "2.7.1": two torch builds in one process would crash later, so stop here.
        if (python_torch.substr(0, python_torch.find('+')) != TORCH_VERSION) {
            throw std::runtime_error("LibTorchBackend: the embedded Python imports torch " + python_torch + ", but this program is "
                                     "linked with LibTorch " + TORCH_VERSION + "; activate the environment with the same torch "
                                     "(the one the model was converted in), or put its site-packages on PYTHONPATH");
        }
    }
    load_op_libs(meta.required_op_libs);
    ensure_halo_exchange_op_registered();   // dsmd::halo_exchange, called by a layer-exchange model
    ensure_kspace_op_registered();          // dsmd::kspace, called by a model converted with --kspace pme

    // 4. the runner for this file format
    if (meta.backend == "torchscript") {
        s.runner = std::make_unique<TorchScriptRunner>(
            (std::filesystem::path(cfg.model_path) / "model.pt").string(), s.device);
    } else if (meta.backend == "aoti") {
        check_aoti_target(meta, s.device);
        s.runner = std::make_unique<AOTIRunner>(
            (std::filesystem::path(cfg.model_path) / "model.pt2").string(), s.device);
    } else if (meta.backend == "pytorch") {
        // no model file: the converter's wrapper is built again and run by the embedded Python (mlip_convert/pytorch_runner.py)
        s.runner = std::make_unique<PyTorchModelRunner>(cfg.model_path, meta.python_path, s.device);
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
bool LibTorchBackend::gives_virial_diagonal() const { return impl_->metadata.virial_diagonal; }
int LibTorchBackend::edge_receiver_row() const { return impl_->metadata.edge_receiver_row; }
bool LibTorchBackend::exchanges_layers() const { return impl_->metadata.layer_exchange; }
bool LibTorchBackend::needs_whole_system() const { return impl_->metadata.whole_system; }
const MLIPKspaceSettings* LibTorchBackend::kspace_settings() const { return &impl_->metadata.kspace; }
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
    // The transport the model's dsmd::halo_exchange calls (forward pass and the backward pass of the force) use; none on a single rank.
    HaloTransportScope halo_scope(in.halo);
    // The context of dsmd::kspace (box, grid, grid sum) for a model with k-space terms; nullptr for every other model.
    KspaceScope kspace_scope(in.kspace);

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
            // [1] the trace, or with metadata virial_diagonal [3] the diagonal W_xx, W_yy, W_zz (one copy to the host)
            const torch::Tensor virial = result[3].reshape(-1).to(torch::kCPU, torch::kFloat64);
            const int64_t expected = s.metadata.virial_diagonal ? 3 : 1;
            if (virial.numel() != expected) {
                throw std::runtime_error("LibTorchBackend: the model returned a virial of " + std::to_string(virial.numel()) +
                                         " values, metadata.json says " + std::to_string(expected) + " (virial_diagonal)");
            }
            const double* w = virial.data_ptr<double>();
            if (expected == 3) {
                for (int a = 0; a < 3; ++a) out.virial_diag[a] = w[a] * s.energy_to_md;
                out.virial = out.virial_diag[0] + out.virial_diag[1] + out.virial_diag[2];
                out.has_virial_diag = true;
            } else {
                out.virial = w[0] * s.energy_to_md;
            }
            out.has_virial = true;
        }
    }
}
