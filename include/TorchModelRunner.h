#ifndef TORCH_MODEL_RUNNER_H
#define TORCH_MODEL_RUNNER_H

// LibTorchBackend internal: runs one converted model file. This header includes LibTorch, so only the LibTorch
// sources (built with ENABLE_LIBTORCH) include it; MLIPBackend.h and the MD code never do.

#include <torch/torch.h>

#include <vector>

class TorchModelRunner {
public:
    virtual ~TorchModelRunner() = default;

    // inputs:  {pos [N, 3], atomic_numbers [N] int64, edge_index [2, E] int64, n_owned [] int64} (design 3.1)
    // returns: {energy [1], atom_energy [N], forces [N, 3]} (design 3.2), in the model's dtype and units, on the
    //          device the inputs are on.
    virtual std::vector<torch::Tensor> run(const std::vector<torch::Tensor>& inputs) = 0;
};

#endif // TORCH_MODEL_RUNNER_H
