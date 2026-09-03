/**
 * mlip_interface.h
 *
 * Pure abstract interface that every MLIP adapter must implement.
 * The MD core depends only on this header — no framework-specific headers allowed.
 */

#ifndef MLIP_INTERFACE_H
#define MLIP_INTERFACE_H

#include "mlip_types.h"

// ---------------------------------------------------------------------------
// Options passed to compute() to control per-step behaviour.
// ---------------------------------------------------------------------------
struct MlipComputeOptions {
    bool compute_virial      = false;
    bool compute_per_atom_e  = false;
    bool validate_output     = false;  // enable NaN/shape checks (debug)
};

// ---------------------------------------------------------------------------
// Core interface.  All methods are per-rank; there is no global state.
// ---------------------------------------------------------------------------
class IMlipPotential {
public:
    virtual ~IMlipPotential() = default;

    /**
     * Load the model, set up runtime and device context.
     * Called once before the first compute() call.
     * Throws std::runtime_error on any failure.
     */
    virtual void initialize(
        const mlip::MlipModelConfig&      model_config,
        const mlip::MlipExecutionContext& context
    ) = 0;

    /**
     * Return capabilities of this model.
     * Valid only after initialize() succeeds.
     */
    virtual mlip::MlipCapabilities capabilities() const = 0;

    /**
     * Return the model's interaction cutoff in Angstroms.
     * Valid only after initialize() succeeds.
     */
    virtual double cutoff() const = 0;

    /**
     * Return neighbor list requirements so the MD core builds the right format.
     * Valid only after initialize() succeeds.
     */
    virtual mlip::NeighborRequirements neighbor_requirements() const = 0;

    /**
     * Enqueue the ML inference on the CUDA stream provided in mlip::MlipInputView.
     * Must NOT call cudaDeviceSynchronize() internally unless unavoidable.
     * The call returns as soon as all GPU work is enqueued.
     *
     * On success, output.forces is populated on the GPU.
     * Throws std::runtime_error with rank/device/model info on failure.
     */
    virtual void compute(
        const mlip::MlipInputView&       input,
        mlip::MlipOutputView&            output,
        const MlipComputeOptions&  options
    ) = 0;

    /**
     * Block until all previously enqueued GPU work is complete.
     * Called by the MD core when it needs results on the host
     * (e.g., logging, checkpointing) or before MPI communication.
     */
    virtual void synchronize() = 0;

    /**
     * Human-readable description (model name + backend).
     */
    virtual std::string description() const = 0;
};

#endif // MLIP_INTERFACE_H
