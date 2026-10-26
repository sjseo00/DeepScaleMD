#ifndef MLIP_HALO_TRANSPORT_H
#define MLIP_HALO_TRANSPORT_H

// How a model that exchanges node features between its message-passing layers (metadata `layer_exchange`, mlip_convert
// --layer-exchange, MLIP_tools/mlip_convert/halo_op.py) moves them between the ranks. The op dsmd::halo_exchange (HaloExchangeOp.cpp)
// calls it: forward() copies the features of every owned atom into the ghost copies of that atom on the other ranks, reverse()
// is its adjoint (the gradients of the ghost copies are added to the owner's row). No LibTorch types here, so the MD code
// implements it without LibTorch include paths: the MD's implementation is MDSystem's halo exchange of dsDecompX
// (Decomp::forward_exchange / reverse_exchange, md_simulation.cpp); tests use a host implementation.
// The ghost layout is the one of the MD's halo exchange of this step: ghost row k is the copy of the atom the dsDecompX plan
// says; the transport knows it, the model never does.

#include <cstdint>

// One tensor of one rank, row-major [n_owned + n_ghost, width]: the owned rows first, then the ghost rows.
struct MLIPHaloBuffer {
    void* data = nullptr;
    int64_t n_owned = 0;
    int64_t n_ghost = 0;
    int64_t width = 0;
    int element_bytes = 0;     // 4 (float) or 8 (double)
    bool on_device = false;    // false: host memory (tests)
    void* stream = nullptr;    // cudaStream_t of the model's kernels when on_device: the transport orders itself on it
};

class MLIPHaloTransport {
public:
    virtual ~MLIPHaloTransport() = default;
    // Rows [n_owned, n_owned + n_ghost) are overwritten with the rows of the owners; the owned rows are valid on entry.
    virtual void forward(const MLIPHaloBuffer& buffer) = 0;
    // The ghost rows are added to the rows of their owners (on this or another rank); on return the owned rows hold the sums
    // (the contents of the ghost rows are not defined any more).
    virtual void reverse(const MLIPHaloBuffer& buffer) = 0;
};

#endif // MLIP_HALO_TRANSPORT_H
