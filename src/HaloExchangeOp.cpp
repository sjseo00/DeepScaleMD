#include "HaloExchangeOp.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <cstdint>
#include <stdexcept>

namespace {

MLIPHaloTransport* g_transport = nullptr;

MLIPHaloBuffer describe(at::Tensor& tensor, int64_t n_owned, int64_t n_ghost) {
    TORCH_CHECK(tensor.dim() == 2 && tensor.is_contiguous(), "dsmd::halo_exchange: a contiguous [rows, width] tensor is needed");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat || tensor.scalar_type() == at::kDouble,
                "dsmd::halo_exchange: the features must be float32 or float64");
    MLIPHaloBuffer buffer;
    buffer.data = tensor.data_ptr();
    buffer.n_owned = n_owned;
    buffer.n_ghost = n_ghost;
    buffer.width = tensor.size(1);
    buffer.element_bytes = static_cast<int>(tensor.element_size());
    buffer.on_device = tensor.is_cuda();
    buffer.stream = tensor.is_cuda() ? static_cast<void*>(c10::cuda::getCurrentCUDAStream(tensor.device().index()).stream()) : nullptr;
    return buffer;
}

// [n_owned, F] -> [n_owned + n_ghost, F]
at::Tensor exchange_forward(const at::Tensor& feats, int64_t n_ghost) {
    TORCH_CHECK(feats.dim() == 2, "dsmd::halo_exchange: feats must be [n_owned, width]");
    TORCH_CHECK(n_ghost >= 0, "dsmd::halo_exchange: n_ghost must not be negative");
    const int64_t n_owned = feats.size(0);
    at::Tensor out = at::empty({n_owned + n_ghost, feats.size(1)}, feats.options());
    out.narrow(0, 0, n_owned).copy_(feats);
    if (n_ghost > 0) {
        TORCH_CHECK(g_transport != nullptr,
                    "dsmd::halo_exchange: ", n_ghost, " ghost atoms but no halo transport is set: a layer-exchange model needs the "
                    "MD program's halo exchange (run it with several ranks through dsMD, not on its own)");
        g_transport->forward(describe(out, n_owned, n_ghost));
    }
    return out;
}

// [n_owned + n_ghost, F] gradient -> [n_owned, F]: the ghost rows are added to their owners.
at::Tensor exchange_reverse(const at::Tensor& grad_out, int64_t n_owned, int64_t n_ghost) {
    at::Tensor work = grad_out.contiguous().clone();    // the transport sums into the owner rows in place
    if (n_ghost > 0) {
        TORCH_CHECK(g_transport != nullptr, "dsmd::halo_exchange: no halo transport for the backward pass");
        g_transport->reverse(describe(work, n_owned, n_ghost));
    }
    return work.narrow(0, 0, n_owned).contiguous();
}

class HaloExchangeFunction : public torch::autograd::Function<HaloExchangeFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx, const at::Tensor& feats, int64_t n_ghost) {
        ctx->saved_data["n_owned"] = feats.size(0);
        ctx->saved_data["n_ghost"] = n_ghost;
        at::AutoDispatchBelowAutograd below_autograd;
        return exchange_forward(feats, n_ghost);
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_outputs) {
        const int64_t n_owned = ctx->saved_data["n_owned"].toInt();
        const int64_t n_ghost = ctx->saved_data["n_ghost"].toInt();
        return {exchange_reverse(grad_outputs[0], n_owned, n_ghost), at::Tensor()};
    }
};

at::Tensor exchange_autograd(const at::Tensor& feats, int64_t n_ghost) { return HaloExchangeFunction::apply(feats, n_ghost); }

}  // namespace

// FRAGMENT: other code may add operators to the namespace (the Python side of the op uses it for the same schema when a model is
// built in a process of its own).
TORCH_LIBRARY_FRAGMENT(dsmd, m) { m.def("halo_exchange(Tensor feats, int n_ghost) -> Tensor"); }

TORCH_LIBRARY_IMPL(dsmd, CompositeExplicitAutograd, m) { m.impl("halo_exchange", &exchange_forward); }

TORCH_LIBRARY_IMPL(dsmd, Autograd, m) { m.impl("halo_exchange", &exchange_autograd); }

void ensure_halo_exchange_op_registered() {}

HaloTransportScope::HaloTransportScope(MLIPHaloTransport* transport) : previous_(g_transport) { g_transport = transport; }

HaloTransportScope::~HaloTransportScope() { g_transport = previous_; }
