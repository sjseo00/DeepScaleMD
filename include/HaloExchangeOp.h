#ifndef HALO_EXCHANGE_OP_H
#define HALO_EXCHANGE_OP_H

// The LibTorch operator dsmd::halo_exchange(Tensor feats, int n_ghost) -> Tensor that a layer-exchange model (metadata
// `layer_exchange`) calls between its message-passing layers: [n_owned, F] -> [n_owned + n_ghost, F], the owned rows followed
// by the ghost rows, which the MLIPHaloTransport fills with the owners' rows; its autograd backward is the transport's
// reverse(). The Python side of the op (the same schema, for building and checking the model) is MLIP_tools/mlip_convert/halo_op.py.
//
// The operator is registered when this library is loaded; call ensure_halo_exchange_op_registered() once (from LibTorchBackend) so
// that a static link keeps this file. The transport the op uses is the one of the innermost HaloTransportScope: set it around
// the model call (forward AND the backward pass inside the model's wrapper both run inside it). Without a transport only
// n_ghost = 0 works (a single rank), anything else raises.
//
// No LibTorch header here.

#include "MLIPHaloTransport.h"

void ensure_halo_exchange_op_registered();

class HaloTransportScope {
public:
    explicit HaloTransportScope(MLIPHaloTransport* transport);
    ~HaloTransportScope();
    HaloTransportScope(const HaloTransportScope&) = delete;
    HaloTransportScope& operator=(const HaloTransportScope&) = delete;

private:
    MLIPHaloTransport* previous_;
};

#endif // HALO_EXCHANGE_OP_H
