#ifndef EDGE_BUILDER_CUH
#define EDGE_BUILDER_CUH

// GPU edge list for the MLIP (design 4): every ordered pair (i, j), i != j, with |r_j - r_i| < cutoff, over all atoms
// (owned and ghost, ghost-ghost pairs included), both directions. Cell list + count + prefix sum + fill; no
// nonzero()/boolean masks. Plain C++ interface (only cuda_runtime.h), so ordinary .cpp files can use it.

#include <cstdint>

#include <cuda_runtime.h>

class EdgeBuilder {
public:
    // Periodic images as shifts (MLIPStep with edge shifts). Atoms [0, n_centers) are real; every atom after them is a copy of a
    // real one (a periodic image). With this, build() makes only the edges whose center is a real atom, with the neighbor replaced
    // by the real atom it copies, and writes each edge's shift: the neighbor's position minus the position of the real atom
    // it stands for, so that  pos[j] - pos[i] + shift  is the vector to the neighbor that was found. A real atom's own image
    // is an edge (i, i) with a shift. All arrays are device arrays of n entries, indexed by the atom (before sorting).
    struct Images {
        int64_t n_centers = 0;
        const int32_t* d_owner = nullptr;      // the real atom each atom copies (itself for the first n_centers)
        const double* d_shift_x = nullptr;     // position of the atom minus that of its real atom (0 for the real atoms)
        const double* d_shift_y = nullptr;
        const double* d_shift_z = nullptr;
    };

    // Leaves out edges that cannot change the energy of the owned atoms (MLIPStep chooses them). d_receiver_ok: one byte per node
    // the model sees (the real atom: with Images the atom a neighbor copies, otherwise the atom itself); an edge is made only
    // when its receiver has a nonzero byte. row: which row of edge_index receives (0 = i, the center; 1 = j, the neighbor).
    struct Prune {
        const unsigned char* d_receiver_ok = nullptr;
        int row = 1;
    };

    EdgeBuilder();
    ~EdgeBuilder();
    EdgeBuilder(const EdgeBuilder&) = delete;
    EdgeBuilder& operator=(const EdgeBuilder&) = delete;

    // d_x, d_y, d_z: device arrays of n doubles. Everything is queued on `stream`. Returns the edge count E after one
    // device-to-host copy of it (the single synchronization of `stream`).
    //
    // lo / hi (host arrays of 3): optional box that contains every atom. Passing them avoids an extra reduction and an
    // extra synchronization; the MD code knows its subdomain plus ghost depth. An atom outside, or NaN, makes build()
    // throw. Without them the bounds are computed on the device.
    //
    // The result stays valid until the next build(): edge_index() points to int64 [2 * E] on the device, row 0 (E
    // entries) holds i, row 1 holds j. Edges are grouped by i in cell order; the order is deterministic.
    //
    // With `images`: only edges centered on a real atom, neighbors as real atoms, and edge_shift() (see Images).
    // With `prune`: only the edges whose receiver is marked (see Prune); nullptr keeps every edge.
    int64_t build(const double* d_x, const double* d_y, const double* d_z, int64_t n, double cutoff,
                  cudaStream_t stream, const double* lo = nullptr, const double* hi = nullptr,
                  const Images* images = nullptr, const Prune* prune = nullptr);

    const int64_t* edge_index() const { return d_edges_; }
    // double [3 * E] on the device, row-major [E, 3]; valid after a build() with images (else not written).
    const double* edge_shift() const { return d_shift_; }
    int64_t edge_count() const { return n_edges_; }

private:
    struct Status;   // {edge count, out-of-bounds flag}: written by kernels, read back by the one copy

    template <class T>
    void ensure(T*& ptr, size_t& capacity, size_t needed);

    // Per-atom and per-cell work arrays (capacity grows by 1.5x so steady-state steps allocate nothing).
    uint32_t* d_cell_ = nullptr;          size_t cap_cell_ = 0;
    uint32_t* d_cell_sorted_ = nullptr;   size_t cap_cell_sorted_ = 0;
    int32_t* d_atom_ = nullptr;           size_t cap_atom_ = 0;
    int32_t* d_atom_sorted_ = nullptr;    size_t cap_atom_sorted_ = 0;
    double* d_sx_ = nullptr;              size_t cap_sx_ = 0;
    double* d_sy_ = nullptr;              size_t cap_sy_ = 0;
    double* d_sz_ = nullptr;              size_t cap_sz_ = 0;
    int32_t* d_starts_ = nullptr;         size_t cap_starts_ = 0;
    int64_t* d_counts_ = nullptr;         size_t cap_counts_ = 0;
    int64_t* d_offsets_ = nullptr;        size_t cap_offsets_ = 0;
    unsigned char* d_temp_ = nullptr;     size_t cap_temp_ = 0;
    int64_t* d_edges_ = nullptr;          size_t cap_edges_ = 0;   // in int64 entries
    double* d_shift_ = nullptr;           size_t cap_shift_ = 0;   // [E, 3], with images only
    Status* d_status_ = nullptr;
    Status* h_status_ = nullptr;          // pinned
    int64_t n_edges_ = 0;
};

#endif // EDGE_BUILDER_CUH
