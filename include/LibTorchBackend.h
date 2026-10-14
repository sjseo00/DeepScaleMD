#ifndef LIBTORCH_BACKEND_H
#define LIBTORCH_BACKEND_H

// MLIPBackend that runs a converted model (mlip_convert) with LibTorch on the GPU, reading and writing the MD arrays in
// place (design 7.4, 8). No LibTorch header appears here, so the MD code that creates it (MLIPBackendFactory.cpp) needs
// no LibTorch include paths; the LibTorch types live in the Impl of LibTorchBackend.cpp. Built only with
// ENABLE_LIBTORCH.

#include "MLIPBackend.h"
#include "MLIPMetadata.h"

#include <memory>

class LibTorchBackend : public MLIPBackend {
public:
    // Steps of design 7.4: read metadata.json (cfg.model_path is the model folder), check that every MD type maps to
    // an atomic number the model supports (cfg.type_to_z), dlopen required_op_libs, load the model on the current
    // CUDA device, then call it a few times on a dummy input so first-call compilation happens now.
    // Throws std::runtime_error with a message naming the problem.
    explicit LibTorchBackend(const MLIPConfig& cfg);
    ~LibTorchBackend() override;

    // Inputs and outputs must be device pointers on the current CUDA device (MLIPInputs::host_memory == false).
    // Forces are added to out.d_f*, in kcal/mol/A. d_atom_energy (if not null) is overwritten with the per-atom
    // energies in kcal/mol. The energy is copied to the host, which synchronizes, only when out.need_energy is set.
    void compute(const MLIPInputs& in, MLIPOutputs& out) override;

    double cutoff() const override;
    double ghost_cutoff() const override;
    bool uses_edge_shifts() const override;
    const MLIPMetadata& metadata() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

#endif // LIBTORCH_BACKEND_H
