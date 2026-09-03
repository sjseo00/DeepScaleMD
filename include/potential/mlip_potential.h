/**
 * mlip_potential.h
 *
 * IPotential adapter that wraps an IMlipPotential and connects it to the
 * general ForceAccumulator pipeline.
 * Lives in include/potential/ so the MD core only sees potential_interface.h
 * and this header — no MLIP framework headers leak in.
 */

#ifndef MLIP_POTENTIAL_H
#define MLIP_POTENTIAL_H

#include "potential_interface.h"
#include "../mlip/mlip_interface.h"
#include "../mlip/mlip_types.h"
#include "../mlip/mlip_registry.h"

#include <memory>
#include <string>

class MlipPotential : public IPotential {
public:
    explicit MlipPotential(const mlip::MlipModelConfig& config)
        : config_(config) {}

    // ---------------------------------------------------------------------------
    // IPotential interface
    // ---------------------------------------------------------------------------
    void initialize() override {
        // Create adapter via registry (no if-else chain in MD core)
        impl_ = MlipRegistry::instance().create(config_.adapter_name);

        // Execution context is filled in by the MD core before initialize()
        impl_->initialize(config_, exec_ctx_);

        caps_ = impl_->capabilities();
        validate_capabilities();
    }

    void finalize() override {
        if (impl_) {
            impl_->synchronize();
            impl_.reset();
        }
    }

    void compute(
        const SimulationStateView& state,
        ForceAccumulator&          acc,
        const ComputeContext&      ctx
    ) override {
        // Build mlip::MlipInputView from SimulationStateView (zero-copy)
        mlip::MlipInputView in;
        in.num_local  = state.num_local;
        in.num_ghost  = state.num_ghost;
        in.num_total  = state.num_total;
        in.stream     = ctx.stream;

        // positions [num_total, 3]
        in.positions.data      = const_cast<double*>(state.d_x);  // contiguous x,y,z not interleaved
        // NOTE: if MD stores x/y/z as separate arrays we need a staging buffer.
        // For now we rely on the ForceCalculatorCUDA interleaved positions buffer
        // accessed via a helper (see MaceAdapter::build_interleaved_positions).
        in.positions.dtype     = mlip::ScalarType::Float64;
        in.positions.ndim      = 2;
        in.positions.shape     = {state.num_total, 3, 0, 0};
        in.positions.device_id = ctx.device_id;

        in.species.data        = const_cast<int*>(state.d_type);
        in.species.dtype       = mlip::ScalarType::Int32;
        in.species.ndim        = 1;
        in.species.shape       = {state.num_total, 0, 0, 0};
        in.species.device_id   = ctx.device_id;

        in.neighbors.num_edges       = state.num_edges;
        in.neighbors.src             = state.d_edge_src;
        in.neighbors.dst             = state.d_edge_dst;
        in.neighbors.is_full_list    = true;
        in.neighbors.includes_ghost  = true;
        in.neighbors.directed        = true;
        in.neighbors.device_id       = ctx.device_id;

        // cell / pbc as device views are set by the adapter if needed
        // (host copies available in state.cell / state.pbc)

        // Build mlip::MlipOutputView pointing into the force accumulator
        mlip::MlipOutputView out;
        out.forces.data      = acc.d_fx;   // adapter writes [num_total, 3] interleaved
        out.forces.dtype     = mlip::ScalarType::Float64;
        out.forces.ndim      = 2;
        out.forces.shape     = {state.num_total, 3, 0, 0};
        out.forces.device_id = ctx.device_id;
        out.total_energy     = acc.d_pe;
        out.virial           = mlip::DeviceArrayView{acc.d_virial, mlip::ScalarType::Float64, 1, {9,0,0,0}, {1,0,0,0}, ctx.device_id};
        out.has_virial       = ctx.need_virial;
        out.stream           = ctx.stream;

        MlipComputeOptions opts;
        opts.compute_virial = ctx.need_virial;

        impl_->compute(in, out, opts);
    }

    std::string name() const override {
        return "MLIP[" + config_.adapter_name + "/" + config_.backend_name + "]";
    }

    // MD core must call this before initialize() to pass rank/device info.
    void set_execution_context(const mlip::MlipExecutionContext& ctx) {
        exec_ctx_ = ctx;
    }

    const mlip::MlipCapabilities& capabilities() const { return caps_; }
    double model_cutoff() const { return impl_ ? impl_->cutoff() : 0.0; }
    const mlip::NeighborRequirements& neighbor_requirements() const {
        return nbr_reqs_;
    }

    void post_initialize_neighbor_reqs() {
        if (impl_) nbr_reqs_ = impl_->neighbor_requirements();
    }

private:
    mlip::MlipModelConfig          config_;
    mlip::MlipExecutionContext     exec_ctx_;
    mlip::MlipCapabilities         caps_;
    mlip::NeighborRequirements     nbr_reqs_;
    std::unique_ptr<IMlipPotential> impl_;

    void validate_capabilities() {
        if (!caps_.supports_cuda)
            throw std::runtime_error("[MlipPotential] Model does not support CUDA execution.");
        if (!caps_.computes_forces)
            throw std::runtime_error("[MlipPotential] Model does not compute forces.");
    }
};

#endif // MLIP_POTENTIAL_H
