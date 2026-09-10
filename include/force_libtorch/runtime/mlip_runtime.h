/**
 * mlip_runtime.h
 *
 * Generic MLIP runtime.
 * Reads a YAML model spec, compiles an execution plan, and executes
 * preprocessing → backend → postprocessing entirely on the GPU each step.
 *
 * MD core depends on this class (via IMlipPotential adapter below).
 * No framework headers (torch, onnx, etc.) appear here.
 */

#pragma once
#ifndef MLIP_RUNTIME_H
#define MLIP_RUNTIME_H

#include "../mlip_types.h"
#include "../runtime/model_specification.h"
#include "../runtime/execution_plan.h"
#include "../backend/backend_interface.h"

#include <memory>
#include <string>

namespace mlip {

class MlipRuntime {
public:
    MlipRuntime() = default;
    ~MlipRuntime();

    /**
     * Initialize from a YAML model spec file.
     * Parses, validates, compiles execution plan, loads backend.
     * Throws on any error.
     */
    void initialize(
        const std::string&         yaml_path,
        const MlipExecutionContext& context
    );

    /**
     * Initialize from a pre-parsed spec (useful for tests).
     */
    void initialize(
        const MlipModelSpec&       spec,
        const MlipExecutionContext& context
    );

    /**
     * Execute one MD step: preprocess → backend → postprocess.
     * All data stays on GPU.
     * output.forces must be pre-zeroed by the caller.
     */
    void compute(
        const MlipInputView& input,
        MlipOutputView&      output
    );

    void synchronize();

    const MlipCapabilities&      capabilities()            const { return caps_; }
    const NeighborRequirements&  neighbor_requirements()   const;
    double                       cutoff()                  const;
    const MlipExecutionPlan&     execution_plan()          const { return plan_; }
    const MlipModelSpec&         model_spec()              const { return spec_; }

private:
    void load_backend();
    void validate_capabilities();

    void run_preprocessing(const MlipInputView& input,
                           std::vector<BackendTensorView>& tv_inputs);

    void run_postprocessing(std::vector<BackendTensorView>& tv_outputs,
                            MlipOutputView& output,
                            const MlipInputView& input);

    MlipModelSpec         spec_;
    MlipExecutionContext  ctx_;
    MlipExecutionPlan     plan_;
    MlipCapabilities      caps_;

    std::unique_ptr<IMlipBackend> backend_;
    bool initialized_ = false;

    // Per-input scratch GPU buffers for preprocess ops (e.g., full_directed_edge_list).
    // Each entry is {device_ptr, size_in_bytes}.
    std::vector<std::pair<void*, size_t>> scratch_bufs_;
};

} // namespace mlip
#endif // MLIP_RUNTIME_H
