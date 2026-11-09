#include "MLIPKspaceHost.h"

#include "pme_rank_split.h"

#include <cuda_runtime.h>
#include <mpi.h>

#include <climits>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

// The operator's grid, summed over the MD ranks (the PP communicator) in place, through host memory.
void sum_grid_over_ranks(void* data, int64_t count, int element_bytes, bool on_device, void* stream) {
    MPI_Comm comm = PmePpSplit::instance().pp_comm();
    int size = 1;
    MPI_Comm_size(comm, &size);
    if (size <= 1 || count == 0) return;
    const MPI_Datatype type = element_bytes == 8 ? MPI_DOUBLE : MPI_FLOAT;
    const size_t bytes = static_cast<size_t>(count) * static_cast<size_t>(element_bytes);
    if (count > static_cast<int64_t>(INT_MAX)) throw std::runtime_error("MLIP k-space: the grid has too many points for one MPI message");
    if (!on_device) {
        MPI_Allreduce(MPI_IN_PLACE, data, static_cast<int>(count), type, MPI_SUM, comm);
        return;
    }
    std::vector<unsigned char> host(bytes);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    if (cudaMemcpyAsync(host.data(), data, bytes, cudaMemcpyDeviceToHost, s) != cudaSuccess || cudaStreamSynchronize(s) != cudaSuccess) {
        throw std::runtime_error("MLIP k-space: copying the grid to the host failed");
    }
    MPI_Allreduce(MPI_IN_PLACE, host.data(), static_cast<int>(count), type, MPI_SUM, comm);
    if (cudaMemcpyAsync(data, host.data(), bytes, cudaMemcpyHostToDevice, s) != cudaSuccess || cudaStreamSynchronize(s) != cudaSuccess) {
        throw std::runtime_error("MLIP k-space: copying the grid back to the device failed");
    }
}

}  // namespace

std::unique_ptr<MLIPKspaceHost> MLIPKspaceHost::create(const MLIPKspaceSettings* model, const MLIPKspaceInput& input, bool periodic_all,
                                                       bool verbose) {
    const bool model_has = model != nullptr && model->enabled;
    if (!input.mode.empty() && input.mode != "none" && input.mode != "pme") {
        throw std::runtime_error("mlip_kspace: '" + input.mode + "' is not known (pme or none)");
    }
    const bool input_has = input.mode == "pme";
    if (!model_has && !input_has) {
        if (input.grid_spacing > 0.0 || input.dispersion_grid_spacing > 0.0 || input.order > 0) {
            throw std::runtime_error("mlip_kspace_grid_spacing / mlip_kspace_dispersion_grid_spacing / mlip_kspace_order are set, but the "
                                     "run has no k-space sums (mlip_kspace: pme, with a model converted with --kspace pme)");
        }
        return nullptr;
    }
    if (model_has && !input_has) {
        throw std::runtime_error("the MLIP model was converted with --kspace pme: its electrostatics and dispersion are split into "
                                 "real-space terms (in the model) and k-space sums (in dsMD). Add 'mlip_kspace: pme' to the input file "
                                 "(without it the long-range energy would be wrong), or convert the model again without --kspace");
    }
    if (!model_has && input_has) {
        throw std::runtime_error("mlip_kspace: pme is set, but the MLIP model has no k-space terms (metadata.json has no \"kspace\"). "
                                 "Convert it with --kspace pme (so3lr), or remove mlip_kspace from the input file");
    }
    if (!periodic_all) {
        throw std::runtime_error("mlip_kspace: pme needs a box that is periodic along x, y and z");
    }
    std::unique_ptr<MLIPKspaceHost> host(new MLIPKspaceHost());
    host->settings_ = *model;
    MLIPKspaceContext& c = host->context_;
    c.grid_spacing = input.grid_spacing > 0.0 ? input.grid_spacing : model->grid_spacing;
    c.dispersion_grid_spacing = input.dispersion_grid_spacing > 0.0 ? input.dispersion_grid_spacing : model->dispersion_grid_spacing;
    c.order = input.order > 0 ? input.order : model->order;
    if (!(c.grid_spacing > 0.0) || !(c.dispersion_grid_spacing > 0.0) || (c.order != 4 && c.order != 6)) {
        throw std::runtime_error("mlip_kspace: the grid spacings must be positive and the B-spline order 4 or 6");
    }
    if (verbose) {
        std::ostringstream s;
        s << "[MLIP] k-space sums (mlip_kspace: pme): smooth PME, B-spline order " << c.order;
        if (model->electrostatics) s << "; electrostatics smearing " << model->smearing << " A, grid spacing " << c.grid_spacing << " A";
        if (model->dispersion) {
            s << "; dispersion smearing " << model->dispersion_smearing << " A, grid spacing " << c.dispersion_grid_spacing << " A";
        }
        std::cout << s.str() << std::endl;
    }
    return host;
}

MLIPKspaceContext* MLIPKspaceHost::prepare(const double box[3], bool sum_over_ranks, int ranks) {
    for (int a = 0; a < 3; ++a) {
        context_.box[a] = box[a];
        context_.virial_correction[a] = 0.0;
    }
    context_.calls = 0;
    context_.grid_sum = sum_over_ranks ? &sum_grid_over_ranks : nullptr;
    context_.grid_ranks = sum_over_ranks ? ranks : 1;
    return &context_;
}
