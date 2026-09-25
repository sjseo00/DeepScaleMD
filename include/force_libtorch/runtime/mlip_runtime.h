
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

    void initialize(
        const std::string&         yaml_path,
        const MlipExecutionContext& context
    );

    void initialize(
        const MlipModelSpec&       spec,
        const MlipExecutionContext& context
    );

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

    std::vector<std::pair<void*, size_t>> scratch_bufs_;
};

} // namespace mlip
#endif // MLIP_RUNTIME_H
