#ifndef MLIP_STEP_CUH
#define MLIP_STEP_CUH

// Everything the MD side does around one MLIPBackend::compute() on the GPU (design 4, 5, 8):
//   * periodic images for the axes the MD does not decompose (it has no ghosts there: a single rank, or an axis with
//     one rank), so the model sees real neighbors across the periodic boundary,
//   * the edge list (EdgeBuilder) over owned + ghost + image atoms,
//   * zeroed work arrays for the model's forces, and afterwards adding them to the MD force arrays with each image's
//     force folded back onto the atom it copies, in a fixed order (no atomics),
//   * the MLIP's contribution to the virial, W = sum_i r_i . F_i over all atoms the model saw.
// With edge shifts (a model that takes periodic images as per-edge shift vectors, MLIPStepGeometry::edge_shifts) the images are
// only searched for neighbors: the model gets the MD atoms alone, every edge to an image becomes an edge to the atom it copies with
// the lattice vector as its shift, nothing is folded back, and the virial is the model's own (it includes the shift term).
// Only used when an MLIP backend that wants edges (cutoff() > 0) is selected; builds without LibTorch never compile it.
// Compile with -DUSE_CUDA (MLIPBackend.h).

#include "EdgeBuilder.cuh"
#include "MLIPBackend.h"

#include <cstdint>

#include <cuda_runtime.h>

struct MLIPStepGeometry {
    double box[3] = {0, 0, 0};          // periodic box lengths
    double lo[3] = {0, 0, 0};           // this rank's subdomain; the whole box on an undivided axis
    double hi[3] = {0, 0, 0};
    bool image_axis[3] = {false, false, false};   // periodic and not decomposed: make periodic images along it
    double depth = 0.0;                 // how far the model reaches: the backend's ghost_cutoff
    double slack = 0.0;                 // how far atoms may sit outside [lo, hi] (the Verlet skin) plus a margin
    bool edge_shifts = false;           // the backend takes periodic images as edge shifts (MLIPBackend::uses_edge_shifts)
    // MLIPBackend::edge_receiver_row(), or -1. With a row, only the edges are built whose receiver lies within
    // (depth - cutoff) + slack of the subdomain box [lo, hi]: the others cannot change the energy of an owned atom.
    int prune_row = -1;
    // A layer-exchange model (MLIPBackend::exchanges_layers): the edges go onto the owned atoms only (the features of the ghost
    // atoms come from their owners), whatever the distance. Needs prune_row.
    bool owned_receivers_only = false;
};

class MLIPStep {
public:
    MLIPStep();
    ~MLIPStep();
    MLIPStep(const MLIPStep&) = delete;
    MLIPStep& operator=(const MLIPStep&) = delete;

    // d_x, d_y, d_z, d_type: the MD arrays, n_md atoms = n_owned owned ones first, then ghosts. Queues everything on
    // `stream`. Synchronizes `stream` once for the edge count, and once more for the image count when there are image
    // axes. Throws std::runtime_error when an atom lies outside the expected region.
    void prepare(const double* d_x, const double* d_y, const double* d_z, const int* d_type, int64_t n_owned,
                 int64_t n_md, double cutoff, const MLIPStepGeometry& geometry, cudaStream_t stream);

    // Valid until the next prepare(). With images the position and type arrays are this object's own (wrapped copies of
    // the MD atoms followed by the images); without images they are the MD arrays themselves.
    MLIPInputs inputs() const;
    // The force arrays to hand to the backend (zeroed by prepare()).
    MLIPOutputs outputs(bool need_energy) const;

    int64_t n_atoms() const { return n_all_; }       // what the model sees: owned + ghost + images (with edge shifts: owned + ghost)
    int64_t n_images() const { return n_all_ - n_md_; }
    bool edge_shifts_active() const { return shift_active_; }   // this step's images are edge shifts
    int64_t n_edges() const { return edges_.edge_count(); }
    bool edges_pruned() const { return pruned_; }    // this step's edge list leaves out the edges that cannot matter

    // d_fx += the model's forces on the MD atoms, images folded onto the atoms they copy. With n_target >= 0 only the first
    // n_target MD atoms (the replicated step: this rank's own atoms come first, MLIPReplica).
    void add_forces(double* d_fx, double* d_fy, double* d_fz, cudaStream_t stream, int64_t n_target = -1);

    // sum_i r_i . F_i over every atom the model saw, with the model's forces (before folding), in kcal/mol. Synchronizes.
    // With edge shifts it is the virial the model returned (set_model_virial), which includes the shift term.
    double virial(cudaStream_t stream);

    // The backend's virial of the last compute() (MLIPOutputs::virial), needed by virial() when edge_shifts_active().
    void set_model_virial(bool has_virial, double virial) { model_virial_valid_ = has_virial; model_virial_ = virial; }

private:
    template <class T>
    void ensure(T*& ptr, size_t& capacity, size_t needed);

    EdgeBuilder edges_;
    cudaStream_t stream_ = nullptr;
    int64_t n_owned_ = 0, n_md_ = 0, n_all_ = 0;
    bool have_images_ = false;
    bool pruned_ = false;
    bool shift_active_ = false;         // images are edge shifts this step
    bool model_virial_valid_ = false;
    double model_virial_ = 0.0;
    const double* md_x_ = nullptr;
    const double* md_y_ = nullptr;
    const double* md_z_ = nullptr;
    const int* md_type_ = nullptr;

    // Extended atoms (when there are images).
    double* d_ext_x_ = nullptr;  size_t cap_ext_x_ = 0;
    double* d_ext_y_ = nullptr;  size_t cap_ext_y_ = 0;
    double* d_ext_z_ = nullptr;  size_t cap_ext_z_ = 0;
    int* d_ext_type_ = nullptr;  size_t cap_ext_type_ = 0;
    // With edge shifts: the real atom each extended atom copies, and its offset from it (EdgeBuilder::Images).
    int32_t* d_ext_owner_ = nullptr;  size_t cap_ext_owner_ = 0;
    double* d_ext_sx_ = nullptr;  size_t cap_ext_sx_ = 0;
    double* d_ext_sy_ = nullptr;  size_t cap_ext_sy_ = 0;
    double* d_ext_sz_ = nullptr;  size_t cap_ext_sz_ = 0;
    int64_t* d_counts_ = nullptr;   size_t cap_counts_ = 0;     // images per MD atom
    int64_t* d_offsets_ = nullptr;  size_t cap_offsets_ = 0;    // first image slot per MD atom
    // The model's forces and the virial scratch, over every atom the model saw.
    double* d_fx_ = nullptr;  size_t cap_fx_ = 0;
    double* d_fy_ = nullptr;  size_t cap_fy_ = 0;
    double* d_fz_ = nullptr;  size_t cap_fz_ = 0;
    double* d_terms_ = nullptr;  size_t cap_terms_ = 0;
    unsigned char* d_temp_ = nullptr;  size_t cap_temp_ = 0;
    unsigned char* d_ok_ = nullptr;    size_t cap_ok_ = 0;      // per node: may be the receiver of an edge that matters
    long long* d_total_ = nullptr;      // image count (device)
    long long* h_total_ = nullptr;      // pinned
    double* d_sum_ = nullptr;           // virial (device)
    double* h_sum_ = nullptr;           // pinned
};

#endif // MLIP_STEP_CUH
