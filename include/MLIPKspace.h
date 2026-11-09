#ifndef MLIP_KSPACE_H
#define MLIP_KSPACE_H

// Reciprocal-space (k-space) sums for MLIPs whose long-range terms are split Ewald-style: so3lr converted with
// `mlip_convert so3lr --kspace pme` (MLIP_tools/mlip_convert/kspace_op.py, adapters/so3lr.py). The model computes per-atom charges and
// calls the operator dsmd::kspace (src/MLIPKspaceOp.cpp), which does smooth PME of them; the MD side (src/MLIPKspaceHost.cpp) gives
// the operator the box, the grid settings and the sum of the grid over the ranks, and adds the virial correction.
// Only for such models and only when the input file asks for it (mlip_kspace: pme): everything else never touches this.
//
// No LibTorch, MPI or CUDA types here.

#include <cstdint>
#include <string>

// metadata.json "kspace" of a model converted with --kspace pme (enabled = false: the model has no k-space term)
struct MLIPKspaceSettings {
    bool enabled = false;
    bool electrostatics = false;
    bool dispersion = false;
    double smearing = 0.0;                  // A, fixed in the model's real-space terms
    double dispersion_smearing = 0.0;       // A
    double grid_spacing = 0.0;              // A, default of the electrostatic grid
    double dispersion_grid_spacing = 0.0;   // A, default of the dispersion grid
    int order = 4;                          // B-spline order (4 or 6)
    double energy_to_md = 1.0;              // the model's energy unit in kcal/mol (set by the backend): the operator works in model units
};

// The input file: mlip_kspace (none | pme), and overrides of the grid settings (0: the model's defaults)
struct MLIPKspaceInput {
    std::string mode;                       // "" (not given) | "none" | "pme"
    double grid_spacing = 0.0;              // mlip_kspace_grid_spacing (A)
    double dispersion_grid_spacing = 0.0;   // mlip_kspace_dispersion_grid_spacing (A)
    int order = 0;                          // mlip_kspace_order (4 or 6)
};

// What the operator gets for one model call (set around it with KspaceScope, MLIPKspaceOp.h) and what it leaves.
struct MLIPKspaceContext {
    double box[3] = {0.0, 0.0, 0.0};        // A (orthorhombic, periodic)
    double grid_spacing = 0.0;
    double dispersion_grid_spacing = 0.0;
    int order = 4;
    int grid_ranks = 1;                     // ranks that share one grid (the grid part of the virial correction is divided by it)
    // Sums a grid over the ranks in place; nullptr when this rank holds all atoms. element_bytes 4 or 8; on_device: CUDA memory
    // written on `stream` (a cudaStream_t).
    void (*grid_sum)(void* data, int64_t count, int element_bytes, bool on_device, void* stream) = nullptr;
    // Out: W_true + sum_i x_i dE/dx_i of the k-space energy per axis (model energy unit, eV for so3lr), this rank's part, added up over the calls. The model's
    // strain derivative (its virial) has -sum_i x_i dE/dx_i of it, not the explicit dependence on the box; adding this gives -dE/dstrain.
    double virial_correction[3] = {0.0, 0.0, 0.0};
    int calls = 0;
};

#endif // MLIP_KSPACE_H
