// SO3LR attention as one PyTorch operator with its own forward and backward CUDA kernels (accel so3lr_fused). The kernels are in
// so3lr_fused_kernels.cu; this file is the operator itself: registration, autograd, checks.
//
// Not part of the dsMD build: the converter compiles the two files of this folder as a torch extension (mlip_convert/accel.py, --accel so3lr_fused)
// and writes the library into metadata.json (required_op_libs); dsMD dlopens it before it loads the model, like OpenEquivariance.
//
// so3lr::attention(q1, k1, v1, w1, q2, k2, w2, cut, ylm, idx_i, idx_j, degree_index, norm1, norm2) -> (x_att, ev_att)
//   q1, k1, v1 [N, H, C1]   per-atom query/key/value of the invariant heads (q/k already RMS-normalized)
//   w1          [E, H, C1]  per-edge filter of the invariant heads
//   q2, k2      [N, D, C2]  per-atom query/key of the degree heads;  w2 [E, D, C2]
//   cut [E], ylm [E, M], idx_i / idx_j [E] int64 (i receives), degree_index [M] int64 (degree head of each ylm column)
//   alpha1[e, h] = sum_c q1[i,h,c] w1[e,h,c] k1[j,h,c] / norm1 * cut[e]      x_att[i, h, :] += alpha1[e, h] * v1[j, h, :]
//   alpha2[e, d] = sum_c q2[i,d,c] w2[e,d,c] k2[j,d,c] / norm2 * cut[e]      ev_att[i, m]   += alpha2[e, degree_index[m]] * ylm[e, m]
// This is exactly SO3kratesLayerSparse's AttentionBlock after the filters (mlip_convert/adapters/so3lr.py, _So3kratesLayer._attention).
// One warp per edge; the sums onto atoms are atomic adds. First derivatives only (forces and virial).
//
// so3lr::pair_energy(pos, shift, idx_i, idx_j, q, alpha, c6, params) -> energy [N]
//   SO3LR's long-range pair terms on the edge list (mlip_convert/adapters/so3lr.py, SO3LRAdapter._long_range), half of each directed
//   edge's energy added to its atom i:
//   r = |pos[j] - pos[i] + shift[e]| < lr_cut
//   electrostatics: 0.5 ke q_i q_j (erf(r/sigma)/r - shift - (1 - f(r)) fshift (r - rc)),  f = switch(r, cuton, rc)
//   dispersion:     0.5 hartree V sw(r),  sw = switch(r, rc - damping, rc),  V = -C6/(rb^6+p^6) - C8/(rb^8+p^8) - C10/(rb^10+p^10),
//                   rb = r / bohr, alpha_ij = (alpha_i + alpha_j)/2, C6 Casimir-Polder mixed from c6 and alpha, QDO damping p and
//                   C8 = 10 sigma^2 C6, C10 = 122.5 sigma^4 C6 with sigma the cubic fit of the vdW radius of alpha_ij
//   q, alpha, c6 [N] are per atom (alpha and c6 already scaled by the free-atom tables). The backward gives the gradients of pos,
//   shift (the virial), q, alpha and c6. params: the fields of So3lrPairParams (so3lr_pair_params.h) in order.
//   With --kspace pme (params 14 and 15 nonzero) these are the real-space parts of the Ewald split of so3lr: erf(r/(sqrt2 s))/r
//   is taken out of the electrostatic potential (shift and fshift are then those of the difference), and the geometric-mean C6
//   times the long-range r^-6 part, sqrt(c6_i c6_j) (1 - exp(-x)(1 + x + x^2/2)) / rb^6, is added to V before the switch.
//
// so3lr::ev_contraction(ev, idx_i, idx_j, degree_index, cg) -> [E, D]
//   out[e, d] = sum over the ylm columns m with degree_index[m] == d of cg[m] (ev[j, m] - ev[i, m])^2: the degree-wise l = 0
//   contraction of the Euclidean-variable differences that feeds SO3LR's spherical filters (make_l0_contraction_fn).
//
// so3lr::radial_filter(x, table, x0, inv_h) -> [E, C]
//   A function of one variable per edge (SO3LR: both radial filters of a layer as functions of the basis distance) read from a table:
//   table [n, 2, C] holds the C values and their C derivatives at the nodes x0 + k / inv_h, k = 0..n-1; between nodes cubic Hermite
//   interpolation, outside the nodes the end cells. The backward gives the derivative in x (the table is a constant).
//
// so3lr::attention_filtered(q1, k1, v1, q2, k2, cut, ylm, x, s, idx_i, idx_j, degree_index, table, x0, inv_h, norm1, norm2)
//   -> (x_att, ev_att)
//   so3lr::attention with the filters put together per edge inside the kernel: w = radial_filter(x, table) + s, columns [0, F) for
//   the invariant heads (w1), [F, 2F) for the degree heads (w2). x [E] basis distance, s [E, 2F] the spherical filter outputs
//   (matrix products before the call), table as radial_filter with C = 2F. No radial filter array and no filter sum is stored.
//   Needs 32 channels per head (q1 [N, H, 32], q2 [N, D, 32], H = D = F / 32 <= 8).
//   The backward gives q1, k1, v1, q2, k2, cut, ylm, x and s.
//
// so3lr::edge_geometry(vec, idx_i, idx_j, z, floor, nlh_a, nlh_b, columns, n_atoms, params) -> (ylm, d_basis, cut, e_rep)
//   The per-edge geometry of SO3LRAdapter.forward_whole_system from the short-range edge vectors vec [E, 3]: ylm [E, M] the
//   spherical harmonics l = 1..4 of the unit vector (so3lr's order, columns [M] picks them), d_basis [E] the distance with the NLH
//   floor of the element pair (floor [Z, Z], indexed by min(Z) - 1, max(Z) - 1 of z [N], the atomic numbers), cut [E] the PhysNet
//   cutoff, e_rep [n_atoms] the NLH repulsion (nlh_a, nlh_b [Z, Z, K]) of the edges summed onto atom i (zero without repulsion).
//   params: the fields of So3lrGeometryParams (so3lr_pair_params.h) in order. The backward gives vec.
//
// The kernels that add per-edge terms onto atoms (pair_energy, the ev_contraction backward) first sum the lanes of a warp that add
// to the same atom (dsMD's edge lists are grouped by atom i), then make one atomic add per run.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <cstdint>
#include <tuple>
#include <vector>

#include "so3lr_pair_params.h"

// so3lr_fused_kernels.cu (instantiated for float and double)
template <typename T>
cudaError_t so3lr_attention_forward_launch(const T* q1, const T* k1, const T* v1, const T* w1, const T* q2, const T* k2, const T* w2,
                                           const T* cut, const T* ylm, const int64_t* idx_i, const int64_t* idx_j,
                                           const int64_t* degree_index, int64_t n_edges, int H, int C1, int D, int C2, int M,
                                           double norm1, double norm2, T* x_att, T* ev_att, cudaStream_t stream);
template <typename T>
cudaError_t so3lr_attention_backward_launch(const T* g_x, const T* g_ev, const T* q1, const T* k1, const T* v1, const T* w1,
                                            const T* q2, const T* k2, const T* w2, const T* cut, const T* ylm, const int64_t* idx_i,
                                            const int64_t* idx_j, const int64_t* degree_index, int64_t n_edges, int H, int C1, int D,
                                            int C2, int M, double norm1, double norm2, T* gq1, T* gk1, T* gv1, T* gw1, T* gq2, T* gk2,
                                            T* gw2, T* gcut, T* gylm, cudaStream_t stream);

template <typename T>
cudaError_t so3lr_pair_forward_launch(const T* pos, const T* shift, const int64_t* idx_i, const int64_t* idx_j, const T* q,
                                      const T* alpha, const T* c6, int64_t n_edges, const So3lrPairParams& p, T* energy,
                                      cudaStream_t stream);
template <typename T>
cudaError_t so3lr_pair_backward_launch(const T* g_energy, const T* pos, const T* shift, const int64_t* idx_i, const int64_t* idx_j,
                                       const T* q, const T* alpha, const T* c6, int64_t n_edges, const So3lrPairParams& p, T* g_pos,
                                       T* g_shift, T* g_q, T* g_alpha, T* g_c6, cudaStream_t stream);

template <typename T>
cudaError_t so3lr_ev_contraction_forward_launch(const T* ev, const int64_t* idx_i, const int64_t* idx_j, const int64_t* degree_index,
                                                const T* cg, int64_t n_edges, int M, int D, T* out, cudaStream_t stream);
template <typename T>
cudaError_t so3lr_ev_contraction_backward_launch(const T* g_out, const T* ev, const int64_t* idx_i, const int64_t* idx_j,
                                                 const int64_t* degree_index, const T* cg, int64_t n_edges, int M, int D, T* g_ev,
                                                 cudaStream_t stream);

template <typename T>
cudaError_t so3lr_radial_forward_launch(const T* x, const T* table, int64_t n_edges, int n_nodes, int C, double x0, double inv_h, T* out,
                                        cudaStream_t stream);
template <typename T>
cudaError_t so3lr_radial_backward_launch(const T* g_out, const T* x, const T* table, int64_t n_edges, int n_nodes, int C, double x0,
                                         double inv_h, T* g_x, cudaStream_t stream);

template <typename T>
cudaError_t so3lr_geometry_forward_launch(const T* vec, const int64_t* idx_i, const int64_t* idx_j, const int64_t* z, const T* floor_d,
                                          const T* nlh_a, const T* nlh_b, int Zt, int Kn, const int64_t* columns, int M, int64_t n_edges,
                                          const So3lrGeometryParams& p, T* ylm, T* d_basis, T* cut, T* e_rep, cudaStream_t stream);
template <typename T>
cudaError_t so3lr_geometry_backward_launch(const T* g_ylm, const T* g_db, const T* g_cut, const T* g_erep, const T* vec,
                                           const int64_t* idx_i, const int64_t* idx_j, const int64_t* z, const T* floor_d, const T* nlh_a,
                                           const T* nlh_b, int Zt, int Kn, const int64_t* columns, int M, int64_t n_edges,
                                           const So3lrGeometryParams& p, T* g_vec, cudaStream_t stream);

template <typename T>
cudaError_t so3lr_attention_filtered_forward_launch(const T* q1, const T* k1, const T* v1, const T* q2, const T* k2, const T* cut,
                                                    const T* ylm, const T* x, const T* s, const int64_t* idx_i, const int64_t* idx_j,
                                                    const int64_t* degree_index, const T* table, int n_nodes, int F, int64_t n_edges,
                                                    int M, double x0, double inv_h, double norm1, double norm2, T* x_att, T* ev_att,
                                                    cudaStream_t stream);
template <typename T>
cudaError_t so3lr_attention_filtered_backward_launch(const T* g_x, const T* g_ev, const T* q1, const T* k1, const T* v1, const T* q2,
                                                     const T* k2, const T* cut, const T* ylm, const T* x, const T* s,
                                                     const int64_t* idx_i, const int64_t* idx_j, const int64_t* degree_index,
                                                     const T* table, int n_nodes, int F, int64_t n_edges, int M, double x0,
                                                     double inv_h, double norm1, double norm2, T* gq1, T* gk1, T* gv1, T* gq2, T* gk2,
                                                     T* gcut, T* gylm, T* gx, T* gs, cudaStream_t stream);

namespace {

constexpr int kMaxDegreeHeads = 16;   // the same limit as in so3lr_fused_kernels.cu

void check_inputs(const std::vector<at::Tensor>& floats, const at::Tensor& idx_i, const at::Tensor& idx_j, const at::Tensor& degree_index) {
    for (const auto& t : floats) {
        TORCH_CHECK(t.is_cuda(), "so3lr::attention: every tensor must be on a CUDA device");
        TORCH_CHECK(t.scalar_type() == floats[0].scalar_type(), "so3lr::attention: the float inputs must have one dtype");
    }
    for (const auto* t : {&idx_i, &idx_j, &degree_index}) {
        TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kLong, "so3lr::attention: index tensors must be int64 on the GPU");
    }
    TORCH_CHECK(floats[0].dim() == 3 && floats[4].dim() == 3, "so3lr::attention: q1 and q2 must be [N, heads, channels]");
    TORCH_CHECK(floats[4].size(1) <= kMaxDegreeHeads, "so3lr::attention: at most ", kMaxDegreeHeads, " degree heads");
}

std::tuple<at::Tensor, at::Tensor> attention_forward_cuda(const at::Tensor& q1_, const at::Tensor& k1_, const at::Tensor& v1_,
                                                          const at::Tensor& w1_, const at::Tensor& q2_, const at::Tensor& k2_,
                                                          const at::Tensor& w2_, const at::Tensor& cut_, const at::Tensor& ylm_,
                                                          const at::Tensor& idx_i_, const at::Tensor& idx_j_,
                                                          const at::Tensor& degree_index_, double norm1, double norm2) {
    const at::cuda::CUDAGuard guard(q1_.device());
    auto q1 = q1_.contiguous(), k1 = k1_.contiguous(), v1 = v1_.contiguous(), w1 = w1_.contiguous();
    auto q2 = q2_.contiguous(), k2 = k2_.contiguous(), w2 = w2_.contiguous(), cut = cut_.contiguous(), ylm = ylm_.contiguous();
    auto idx_i = idx_i_.contiguous(), idx_j = idx_j_.contiguous(), degree_index = degree_index_.contiguous();
    check_inputs({q1, k1, v1, w1, q2, k2, w2, cut, ylm}, idx_i, idx_j, degree_index);
    const int64_t n_atoms = q1.size(0), n_edges = idx_i.size(0);
    const int H = q1.size(1), C1 = q1.size(2), D = q2.size(1), C2 = q2.size(2), M = ylm.size(1);
    auto x_att = at::zeros_like(q1);
    auto ev_att = at::zeros({n_atoms, M}, q1.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(q1.scalar_type(), "so3lr_attention_forward", [&] {
        const cudaError_t err = so3lr_attention_forward_launch<scalar_t>(
            q1.data_ptr<scalar_t>(), k1.data_ptr<scalar_t>(), v1.data_ptr<scalar_t>(), w1.data_ptr<scalar_t>(), q2.data_ptr<scalar_t>(),
            k2.data_ptr<scalar_t>(), w2.data_ptr<scalar_t>(), cut.data_ptr<scalar_t>(), ylm.data_ptr<scalar_t>(),
            idx_i.data_ptr<int64_t>(), idx_j.data_ptr<int64_t>(), degree_index.data_ptr<int64_t>(), n_edges, H, C1, D, C2, M, norm1,
            norm2, x_att.data_ptr<scalar_t>(), ev_att.data_ptr<scalar_t>(), stream);
        TORCH_CHECK(err == cudaSuccess, "so3lr::attention forward kernel: ", cudaGetErrorString(err));
    });
    return {x_att, ev_att};
}

std::vector<at::Tensor> attention_backward_cuda(const at::Tensor& g_x_, const at::Tensor& g_ev_, const std::vector<at::Tensor>& saved,
                                                double norm1, double norm2) {
    const at::Tensor &q1 = saved[0], &k1 = saved[1], &v1 = saved[2], &w1 = saved[3], &q2 = saved[4], &k2 = saved[5];
    const at::Tensor &w2 = saved[6], &cut = saved[7], &ylm = saved[8], &idx_i = saved[9], &idx_j = saved[10];
    const at::Tensor& degree_index = saved[11];
    const at::cuda::CUDAGuard guard(q1.device());
    auto g_x = g_x_.defined() ? g_x_.contiguous() : at::zeros_like(q1);
    auto g_ev = g_ev_.defined() ? g_ev_.contiguous() : at::zeros({q1.size(0), ylm.size(1)}, q1.options());
    const int64_t n_edges = idx_i.size(0);
    const int H = q1.size(1), C1 = q1.size(2), D = q2.size(1), C2 = q2.size(2), M = ylm.size(1);
    auto gq1 = at::zeros_like(q1), gk1 = at::zeros_like(k1), gv1 = at::zeros_like(v1), gw1 = at::empty_like(w1);
    auto gq2 = at::zeros_like(q2), gk2 = at::zeros_like(k2), gw2 = at::empty_like(w2);
    auto gcut = at::empty_like(cut), gylm = at::empty_like(ylm);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(q1.scalar_type(), "so3lr_attention_backward", [&] {
        const cudaError_t err = so3lr_attention_backward_launch<scalar_t>(
            g_x.data_ptr<scalar_t>(), g_ev.data_ptr<scalar_t>(), q1.data_ptr<scalar_t>(), k1.data_ptr<scalar_t>(), v1.data_ptr<scalar_t>(),
            w1.data_ptr<scalar_t>(), q2.data_ptr<scalar_t>(), k2.data_ptr<scalar_t>(), w2.data_ptr<scalar_t>(), cut.data_ptr<scalar_t>(),
            ylm.data_ptr<scalar_t>(), idx_i.data_ptr<int64_t>(), idx_j.data_ptr<int64_t>(), degree_index.data_ptr<int64_t>(), n_edges, H,
            C1, D, C2, M, norm1, norm2, gq1.data_ptr<scalar_t>(), gk1.data_ptr<scalar_t>(), gv1.data_ptr<scalar_t>(),
            gw1.data_ptr<scalar_t>(), gq2.data_ptr<scalar_t>(), gk2.data_ptr<scalar_t>(), gw2.data_ptr<scalar_t>(),
            gcut.data_ptr<scalar_t>(), gylm.data_ptr<scalar_t>(), stream);
        TORCH_CHECK(err == cudaSuccess, "so3lr::attention backward kernel: ", cudaGetErrorString(err));
    });
    return {gq1, gk1, gv1, gw1, gq2, gk2, gw2, gcut, gylm};
}

class AttentionFunction : public torch::autograd::Function<AttentionFunction> {
public:
    static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx, const at::Tensor& q1, const at::Tensor& k1,
                                                  const at::Tensor& v1, const at::Tensor& w1, const at::Tensor& q2,
                                                  const at::Tensor& k2, const at::Tensor& w2, const at::Tensor& cut,
                                                  const at::Tensor& ylm, const at::Tensor& idx_i, const at::Tensor& idx_j,
                                                  const at::Tensor& degree_index, double norm1, double norm2) {
        at::AutoDispatchBelowADInplaceOrView below_autograd;
        auto out = attention_forward_cuda(q1, k1, v1, w1, q2, k2, w2, cut, ylm, idx_i, idx_j, degree_index, norm1, norm2);
        ctx->save_for_backward({q1.contiguous(), k1.contiguous(), v1.contiguous(), w1.contiguous(), q2.contiguous(), k2.contiguous(),
                                w2.contiguous(), cut.contiguous(), ylm.contiguous(), idx_i.contiguous(), idx_j.contiguous(),
                                degree_index.contiguous()});
        ctx->saved_data["norm1"] = norm1;
        ctx->saved_data["norm2"] = norm2;
        return {std::get<0>(out), std::get<1>(out)};
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grads) {
        const auto saved = ctx->get_saved_variables();
        auto g = attention_backward_cuda(grads[0], grads[1], saved, ctx->saved_data["norm1"].toDouble(),
                                         ctx->saved_data["norm2"].toDouble());
        return {g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8], at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                at::Tensor()};
    }
};

std::tuple<at::Tensor, at::Tensor> attention_autograd(const at::Tensor& q1, const at::Tensor& k1, const at::Tensor& v1,
                                                      const at::Tensor& w1, const at::Tensor& q2, const at::Tensor& k2,
                                                      const at::Tensor& w2, const at::Tensor& cut, const at::Tensor& ylm,
                                                      const at::Tensor& idx_i, const at::Tensor& idx_j, const at::Tensor& degree_index,
                                                      double norm1, double norm2) {
    auto out = AttentionFunction::apply(q1, k1, v1, w1, q2, k2, w2, cut, ylm, idx_i, idx_j, degree_index, norm1, norm2);
    return {out[0], out[1]};
}

So3lrPairParams pair_params(c10::ArrayRef<double> v) {
    TORCH_CHECK(v.size() == 13 || v.size() == 15, "so3lr::pair_energy: params must have 13 or 15 values (So3lrPairParams), got ",
                v.size());
    So3lrPairParams p;
    p.lr_cut = v[0];
    p.sigma = v[1];
    p.cuton = v[2];
    p.coul_shift = v[3];
    p.coul_fshift = v[4];
    p.ke = v[5];
    p.disp_on = v[6];
    p.disp_scale = v[7];
    p.bohr = v[8];
    p.hartree = v[9];
    p.vdw_factor = v[10];
    p.use_elec = v[11] != 0.0;
    p.use_disp = v[12] != 0.0;
    p.pme_smear = v.size() > 13 ? v[13] : 0.0;
    p.disp_smear = v.size() > 14 ? v[14] : 0.0;
    return p;
}

void check_pair_inputs(const at::Tensor& pos, const at::Tensor& shift, const at::Tensor& idx_i, const at::Tensor& idx_j,
                       const at::Tensor& q, const at::Tensor& alpha, const at::Tensor& c6) {
    for (const auto* t : {&pos, &shift, &q, &alpha, &c6}) {
        TORCH_CHECK(t->is_cuda() && t->scalar_type() == pos.scalar_type(), "so3lr::pair_energy: float inputs must be on the GPU, one dtype");
    }
    TORCH_CHECK(idx_i.is_cuda() && idx_j.is_cuda() && idx_i.scalar_type() == at::kLong && idx_j.scalar_type() == at::kLong,
                "so3lr::pair_energy: index tensors must be int64 on the GPU");
    TORCH_CHECK(pos.dim() == 2 && pos.size(1) == 3 && shift.dim() == 2 && shift.size(1) == 3 && shift.size(0) == idx_i.size(0),
                "so3lr::pair_energy: pos [N, 3], shift [E, 3]");
    TORCH_CHECK(q.numel() == pos.size(0) && alpha.numel() == pos.size(0) && c6.numel() == pos.size(0),
                "so3lr::pair_energy: q, alpha, c6 must have one value per atom");
}

at::Tensor pair_forward_cuda(const at::Tensor& pos_, const at::Tensor& shift_, const at::Tensor& idx_i_, const at::Tensor& idx_j_,
                             const at::Tensor& q_, const at::Tensor& alpha_, const at::Tensor& c6_, c10::ArrayRef<double> params) {
    const at::cuda::CUDAGuard guard(pos_.device());
    auto pos = pos_.contiguous(), shift = shift_.contiguous(), idx_i = idx_i_.contiguous(), idx_j = idx_j_.contiguous();
    auto q = q_.contiguous(), alpha = alpha_.contiguous(), c6 = c6_.contiguous();
    check_pair_inputs(pos, shift, idx_i, idx_j, q, alpha, c6);
    const So3lrPairParams p = pair_params(params);
    auto energy = at::zeros({pos.size(0)}, pos.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(pos.scalar_type(), "so3lr_pair_forward", [&] {
        const cudaError_t err = so3lr_pair_forward_launch<scalar_t>(
            pos.data_ptr<scalar_t>(), shift.data_ptr<scalar_t>(), idx_i.data_ptr<int64_t>(), idx_j.data_ptr<int64_t>(),
            q.data_ptr<scalar_t>(), alpha.data_ptr<scalar_t>(), c6.data_ptr<scalar_t>(), idx_i.size(0), p, energy.data_ptr<scalar_t>(),
            stream);
        TORCH_CHECK(err == cudaSuccess, "so3lr::pair_energy forward kernel: ", cudaGetErrorString(err));
    });
    return energy;
}

class PairFunction : public torch::autograd::Function<PairFunction> {
public:
    static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx, const at::Tensor& pos, const at::Tensor& shift,
                                                  const at::Tensor& idx_i, const at::Tensor& idx_j, const at::Tensor& q,
                                                  const at::Tensor& alpha, const at::Tensor& c6, c10::ArrayRef<double> params) {
        at::AutoDispatchBelowADInplaceOrView below_autograd;
        auto energy = pair_forward_cuda(pos, shift, idx_i, idx_j, q, alpha, c6, params);
        ctx->save_for_backward({pos.contiguous(), shift.contiguous(), idx_i.contiguous(), idx_j.contiguous(), q.contiguous(),
                                alpha.contiguous(), c6.contiguous()});
        ctx->saved_data["params"] = std::vector<double>(params.begin(), params.end());
        return {energy};
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grads) {
        const auto s = ctx->get_saved_variables();
        const at::Tensor &pos = s[0], &shift = s[1], &idx_i = s[2], &idx_j = s[3], &q = s[4], &alpha = s[5], &c6 = s[6];
        const auto values = ctx->saved_data["params"].toDoubleVector();
        const So3lrPairParams p = pair_params(values);
        const at::cuda::CUDAGuard guard(pos.device());
        auto g = grads[0].defined() ? grads[0].contiguous() : at::zeros({pos.size(0)}, pos.options());
        auto g_pos = at::zeros_like(pos), g_shift = at::empty_like(shift);
        auto g_q = at::zeros_like(q), g_alpha = at::zeros_like(alpha), g_c6 = at::zeros_like(c6);
        auto stream = at::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES(pos.scalar_type(), "so3lr_pair_backward", [&] {
            const cudaError_t err = so3lr_pair_backward_launch<scalar_t>(
                g.data_ptr<scalar_t>(), pos.data_ptr<scalar_t>(), shift.data_ptr<scalar_t>(), idx_i.data_ptr<int64_t>(),
                idx_j.data_ptr<int64_t>(), q.data_ptr<scalar_t>(), alpha.data_ptr<scalar_t>(), c6.data_ptr<scalar_t>(), idx_i.size(0), p,
                g_pos.data_ptr<scalar_t>(), g_shift.data_ptr<scalar_t>(), g_q.data_ptr<scalar_t>(), g_alpha.data_ptr<scalar_t>(),
                g_c6.data_ptr<scalar_t>(), stream);
            TORCH_CHECK(err == cudaSuccess, "so3lr::pair_energy backward kernel: ", cudaGetErrorString(err));
        });
        return {g_pos, g_shift, at::Tensor(), at::Tensor(), g_q, g_alpha, g_c6, at::Tensor()};
    }
};

at::Tensor pair_autograd(const at::Tensor& pos, const at::Tensor& shift, const at::Tensor& idx_i, const at::Tensor& idx_j,
                         const at::Tensor& q, const at::Tensor& alpha, const at::Tensor& c6, c10::ArrayRef<double> params) {
    return PairFunction::apply(pos, shift, idx_i, idx_j, q, alpha, c6, params)[0];
}

at::Tensor ev_contraction_forward_cuda(const at::Tensor& ev_, const at::Tensor& idx_i_, const at::Tensor& idx_j_,
                                       const at::Tensor& degree_index_, const at::Tensor& cg_, int64_t num_heads) {
    const at::cuda::CUDAGuard guard(ev_.device());
    auto ev = ev_.contiguous(), idx_i = idx_i_.contiguous(), idx_j = idx_j_.contiguous();
    auto degree_index = degree_index_.contiguous(), cg = cg_.contiguous();
    TORCH_CHECK(ev.is_cuda() && cg.is_cuda() && cg.scalar_type() == ev.scalar_type() && ev.dim() == 2,
                "so3lr::ev_contraction: ev [N, M] and cg [M] on the GPU, one dtype");
    TORCH_CHECK(idx_i.scalar_type() == at::kLong && idx_j.scalar_type() == at::kLong && degree_index.scalar_type() == at::kLong,
                "so3lr::ev_contraction: int64 indices");
    TORCH_CHECK(num_heads <= kMaxDegreeHeads, "so3lr::ev_contraction: at most ", kMaxDegreeHeads, " degree heads");
    const int M = ev.size(1), D = static_cast<int>(num_heads);
    auto out = at::empty({idx_i.size(0), num_heads}, ev.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(ev.scalar_type(), "so3lr_ev_contraction_forward", [&] {
        const cudaError_t err = so3lr_ev_contraction_forward_launch<scalar_t>(
            ev.data_ptr<scalar_t>(), idx_i.data_ptr<int64_t>(), idx_j.data_ptr<int64_t>(), degree_index.data_ptr<int64_t>(),
            cg.data_ptr<scalar_t>(), idx_i.size(0), M, D, out.data_ptr<scalar_t>(), stream);
        TORCH_CHECK(err == cudaSuccess, "so3lr::ev_contraction forward kernel: ", cudaGetErrorString(err));
    });
    return out;
}

class EvContractionFunction : public torch::autograd::Function<EvContractionFunction> {
public:
    static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx, const at::Tensor& ev, const at::Tensor& idx_i,
                                                  const at::Tensor& idx_j, const at::Tensor& degree_index, const at::Tensor& cg,
                                                  int64_t num_heads) {
        at::AutoDispatchBelowADInplaceOrView below_autograd;
        auto out = ev_contraction_forward_cuda(ev, idx_i, idx_j, degree_index, cg, num_heads);
        ctx->save_for_backward({ev.contiguous(), idx_i.contiguous(), idx_j.contiguous(), degree_index.contiguous(), cg.contiguous()});
        return {out};
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grads) {
        const auto s = ctx->get_saved_variables();
        const at::Tensor &ev = s[0], &idx_i = s[1], &idx_j = s[2], &degree_index = s[3], &cg = s[4];
        const at::cuda::CUDAGuard guard(ev.device());
        auto g_ev = at::zeros_like(ev);
        if (grads[0].defined()) {
            auto g = grads[0].contiguous();
            auto stream = at::cuda::getCurrentCUDAStream();
            AT_DISPATCH_FLOATING_TYPES(ev.scalar_type(), "so3lr_ev_contraction_backward", [&] {
                const cudaError_t err = so3lr_ev_contraction_backward_launch<scalar_t>(
                    g.data_ptr<scalar_t>(), ev.data_ptr<scalar_t>(), idx_i.data_ptr<int64_t>(), idx_j.data_ptr<int64_t>(),
                    degree_index.data_ptr<int64_t>(), cg.data_ptr<scalar_t>(), idx_i.size(0), ev.size(1), g.size(1),
                    g_ev.data_ptr<scalar_t>(), stream);
                TORCH_CHECK(err == cudaSuccess, "so3lr::ev_contraction backward kernel: ", cudaGetErrorString(err));
            });
        }
        return {g_ev, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

at::Tensor ev_contraction_autograd(const at::Tensor& ev, const at::Tensor& idx_i, const at::Tensor& idx_j, const at::Tensor& degree_index,
                                   const at::Tensor& cg, int64_t num_heads) {
    return EvContractionFunction::apply(ev, idx_i, idx_j, degree_index, cg, num_heads)[0];
}

void check_radial_inputs(const at::Tensor& x, const at::Tensor& table) {
    TORCH_CHECK(x.is_cuda() && table.is_cuda() && x.scalar_type() == table.scalar_type(),
                "so3lr::radial_filter: x and table on the GPU, one dtype");
    TORCH_CHECK(x.dim() == 1 && table.dim() == 3 && table.size(1) == 2 && table.size(0) >= 2,
                "so3lr::radial_filter: x [E], table [nodes >= 2, 2, C]");
}

at::Tensor radial_forward_cuda(const at::Tensor& x_, const at::Tensor& table_, double x0, double inv_h) {
    const at::cuda::CUDAGuard guard(x_.device());
    auto x = x_.contiguous(), table = table_.contiguous();
    check_radial_inputs(x, table);
    const int n_nodes = table.size(0), C = table.size(2);
    auto out = at::empty({x.size(0), C}, x.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "so3lr_radial_forward", [&] {
        const cudaError_t err = so3lr_radial_forward_launch<scalar_t>(x.data_ptr<scalar_t>(), table.data_ptr<scalar_t>(), x.size(0), n_nodes,
                                                                      C, x0, inv_h, out.data_ptr<scalar_t>(), stream);
        TORCH_CHECK(err == cudaSuccess, "so3lr::radial_filter forward kernel: ", cudaGetErrorString(err));
    });
    return out;
}

class RadialFunction : public torch::autograd::Function<RadialFunction> {
public:
    static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx, const at::Tensor& x, const at::Tensor& table,
                                                  double x0, double inv_h) {
        at::AutoDispatchBelowADInplaceOrView below_autograd;
        auto out = radial_forward_cuda(x, table, x0, inv_h);
        ctx->save_for_backward({x.contiguous(), table.contiguous()});
        ctx->saved_data["x0"] = x0;
        ctx->saved_data["inv_h"] = inv_h;
        return {out};
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grads) {
        const auto s = ctx->get_saved_variables();
        const at::Tensor &x = s[0], &table = s[1];
        const at::cuda::CUDAGuard guard(x.device());
        auto g_x = at::zeros_like(x);
        if (grads[0].defined()) {
            auto g = grads[0].contiguous();
            auto stream = at::cuda::getCurrentCUDAStream();
            AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "so3lr_radial_backward", [&] {
                const cudaError_t err = so3lr_radial_backward_launch<scalar_t>(
                    g.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), table.data_ptr<scalar_t>(), x.size(0), table.size(0), table.size(2),
                    ctx->saved_data["x0"].toDouble(), ctx->saved_data["inv_h"].toDouble(), g_x.data_ptr<scalar_t>(), stream);
                TORCH_CHECK(err == cudaSuccess, "so3lr::radial_filter backward kernel: ", cudaGetErrorString(err));
            });
        }
        return {g_x, at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

at::Tensor radial_autograd(const at::Tensor& x, const at::Tensor& table, double x0, double inv_h) {
    return RadialFunction::apply(x, table, x0, inv_h)[0];
}

// ---- attention_filtered

struct FilteredShape {
    int64_t n_atoms, n_edges;
    int F, M, n_nodes;
};

FilteredShape check_filtered_inputs(const std::vector<at::Tensor>& floats, const at::Tensor& idx_i, const at::Tensor& idx_j,
                                    const at::Tensor& degree_index) {
    // floats: q1, k1, v1, q2, k2, cut, ylm, x, s, table
    for (const auto& t : floats) {
        TORCH_CHECK(t.is_cuda() && t.scalar_type() == floats[0].scalar_type(),
                    "so3lr::attention_filtered: the float inputs must be on the GPU, one dtype");
    }
    for (const auto* t : {&idx_i, &idx_j, &degree_index}) {
        TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kLong, "so3lr::attention_filtered: index tensors must be int64 on the GPU");
    }
    const at::Tensor &q1 = floats[0], &q2 = floats[3], &ylm = floats[6], &x = floats[7], &s = floats[8], &table = floats[9];
    TORCH_CHECK(q1.dim() == 3 && q2.dim() == 3 && q1.size(2) == 32 && q2.size(2) == 32 && q1.size(1) == q2.size(1) &&
                    q1.size(1) <= 8,
                "so3lr::attention_filtered: q1 [N, H, 32] and q2 [N, D, 32] with H = D <= 8");
    FilteredShape r;
    r.n_atoms = q1.size(0);
    r.n_edges = idx_i.size(0);
    r.F = static_cast<int>(q1.size(1) * 32);
    r.M = static_cast<int>(ylm.size(1));
    r.n_nodes = static_cast<int>(table.size(0));
    TORCH_CHECK(x.dim() == 1 && x.size(0) == r.n_edges && s.dim() == 2 && s.size(0) == r.n_edges && s.size(1) == 2 * r.F,
                "so3lr::attention_filtered: x [E], s [E, 2F]");
    TORCH_CHECK(table.dim() == 3 && table.size(1) == 2 && table.size(2) == 2 * r.F && r.n_nodes >= 2,
                "so3lr::attention_filtered: table [nodes, 2, 2F]");
    return r;
}

std::tuple<at::Tensor, at::Tensor> filtered_forward_cuda(const at::Tensor& q1_, const at::Tensor& k1_, const at::Tensor& v1_,
                                                         const at::Tensor& q2_, const at::Tensor& k2_, const at::Tensor& cut_,
                                                         const at::Tensor& ylm_, const at::Tensor& x_, const at::Tensor& s_,
                                                         const at::Tensor& idx_i_, const at::Tensor& idx_j_,
                                                         const at::Tensor& degree_index_, const at::Tensor& table_, double x0,
                                                         double inv_h, double norm1, double norm2) {
    const at::cuda::CUDAGuard guard(q1_.device());
    auto q1 = q1_.contiguous(), k1 = k1_.contiguous(), v1 = v1_.contiguous(), q2 = q2_.contiguous(), k2 = k2_.contiguous();
    auto cut = cut_.contiguous(), ylm = ylm_.contiguous(), x = x_.contiguous(), s = s_.contiguous();
    auto idx_i = idx_i_.contiguous(), idx_j = idx_j_.contiguous(), degree_index = degree_index_.contiguous();
    auto table = table_.contiguous();
    const FilteredShape r = check_filtered_inputs({q1, k1, v1, q2, k2, cut, ylm, x, s, table}, idx_i, idx_j, degree_index);
    auto x_att = at::zeros_like(q1);
    auto ev_att = at::zeros({r.n_atoms, r.M}, q1.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(q1.scalar_type(), "so3lr_attention_filtered_forward", [&] {
        const cudaError_t err = so3lr_attention_filtered_forward_launch<scalar_t>(
            q1.data_ptr<scalar_t>(), k1.data_ptr<scalar_t>(), v1.data_ptr<scalar_t>(), q2.data_ptr<scalar_t>(), k2.data_ptr<scalar_t>(),
            cut.data_ptr<scalar_t>(), ylm.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(), s.data_ptr<scalar_t>(), idx_i.data_ptr<int64_t>(),
            idx_j.data_ptr<int64_t>(), degree_index.data_ptr<int64_t>(), table.data_ptr<scalar_t>(), r.n_nodes, r.F, r.n_edges, r.M, x0,
            inv_h, norm1, norm2, x_att.data_ptr<scalar_t>(), ev_att.data_ptr<scalar_t>(), stream);
        TORCH_CHECK(err == cudaSuccess, "so3lr::attention_filtered forward kernel: ", cudaGetErrorString(err));
    });
    return {x_att, ev_att};
}

class FilteredFunction : public torch::autograd::Function<FilteredFunction> {
public:
    static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx, const at::Tensor& q1, const at::Tensor& k1,
                                                  const at::Tensor& v1, const at::Tensor& q2, const at::Tensor& k2, const at::Tensor& cut,
                                                  const at::Tensor& ylm, const at::Tensor& x, const at::Tensor& s,
                                                  const at::Tensor& idx_i, const at::Tensor& idx_j, const at::Tensor& degree_index,
                                                  const at::Tensor& table, double x0, double inv_h, double norm1, double norm2) {
        at::AutoDispatchBelowADInplaceOrView below_autograd;
        auto out = filtered_forward_cuda(q1, k1, v1, q2, k2, cut, ylm, x, s, idx_i, idx_j, degree_index, table, x0, inv_h, norm1, norm2);
        ctx->save_for_backward({q1.contiguous(), k1.contiguous(), v1.contiguous(), q2.contiguous(), k2.contiguous(), cut.contiguous(),
                                ylm.contiguous(), x.contiguous(), s.contiguous(), idx_i.contiguous(), idx_j.contiguous(),
                                degree_index.contiguous(), table.contiguous()});
        ctx->saved_data["x0"] = x0;
        ctx->saved_data["inv_h"] = inv_h;
        ctx->saved_data["norm1"] = norm1;
        ctx->saved_data["norm2"] = norm2;
        return {std::get<0>(out), std::get<1>(out)};
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grads) {
        const auto v = ctx->get_saved_variables();
        const at::Tensor &q1 = v[0], &k1 = v[1], &v1 = v[2], &q2 = v[3], &k2 = v[4], &cut = v[5], &ylm = v[6], &x = v[7], &s = v[8];
        const at::Tensor &idx_i = v[9], &idx_j = v[10], &degree_index = v[11], &table = v[12];
        const FilteredShape r = check_filtered_inputs({q1, k1, v1, q2, k2, cut, ylm, x, s, table}, idx_i, idx_j, degree_index);
        const at::cuda::CUDAGuard guard(q1.device());
        auto g_x = grads[0].defined() ? grads[0].contiguous() : at::zeros_like(q1);
        auto g_ev = grads[1].defined() ? grads[1].contiguous() : at::zeros({r.n_atoms, r.M}, q1.options());
        auto gq1 = at::zeros_like(q1), gk1 = at::zeros_like(k1), gv1 = at::zeros_like(v1), gq2 = at::zeros_like(q2), gk2 = at::zeros_like(k2);
        auto gcut = at::empty_like(cut), gylm = at::empty_like(ylm), gx = at::empty_like(x), gs = at::empty_like(s);
        auto stream = at::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES(q1.scalar_type(), "so3lr_attention_filtered_backward", [&] {
            const cudaError_t err = so3lr_attention_filtered_backward_launch<scalar_t>(
                g_x.data_ptr<scalar_t>(), g_ev.data_ptr<scalar_t>(), q1.data_ptr<scalar_t>(), k1.data_ptr<scalar_t>(),
                v1.data_ptr<scalar_t>(), q2.data_ptr<scalar_t>(), k2.data_ptr<scalar_t>(), cut.data_ptr<scalar_t>(), ylm.data_ptr<scalar_t>(),
                x.data_ptr<scalar_t>(), s.data_ptr<scalar_t>(), idx_i.data_ptr<int64_t>(), idx_j.data_ptr<int64_t>(),
                degree_index.data_ptr<int64_t>(), table.data_ptr<scalar_t>(), r.n_nodes, r.F, r.n_edges, r.M,
                ctx->saved_data["x0"].toDouble(), ctx->saved_data["inv_h"].toDouble(), ctx->saved_data["norm1"].toDouble(),
                ctx->saved_data["norm2"].toDouble(), gq1.data_ptr<scalar_t>(), gk1.data_ptr<scalar_t>(), gv1.data_ptr<scalar_t>(),
                gq2.data_ptr<scalar_t>(), gk2.data_ptr<scalar_t>(), gcut.data_ptr<scalar_t>(), gylm.data_ptr<scalar_t>(),
                gx.data_ptr<scalar_t>(), gs.data_ptr<scalar_t>(), stream);
            TORCH_CHECK(err == cudaSuccess, "so3lr::attention_filtered backward kernel: ", cudaGetErrorString(err));
        });
        return {gq1, gk1, gv1, gq2, gk2, gcut, gylm, gx, gs, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                at::Tensor(), at::Tensor(), at::Tensor()};
    }
};

std::tuple<at::Tensor, at::Tensor> filtered_autograd(const at::Tensor& q1, const at::Tensor& k1, const at::Tensor& v1, const at::Tensor& q2,
                                                     const at::Tensor& k2, const at::Tensor& cut, const at::Tensor& ylm, const at::Tensor& x,
                                                     const at::Tensor& s, const at::Tensor& idx_i, const at::Tensor& idx_j,
                                                     const at::Tensor& degree_index, const at::Tensor& table, double x0, double inv_h,
                                                     double norm1, double norm2) {
    auto out = FilteredFunction::apply(q1, k1, v1, q2, k2, cut, ylm, x, s, idx_i, idx_j, degree_index, table, x0, inv_h, norm1, norm2);
    return {out[0], out[1]};
}

// ---- edge_geometry

So3lrGeometryParams geometry_params(c10::ArrayRef<double> v) {
    TORCH_CHECK(v.size() == 5, "so3lr::edge_geometry: params must have 5 values (So3lrGeometryParams), got ", v.size());
    So3lrGeometryParams p;
    p.sr_cutoff = v[0];
    p.floor_width = v[1];
    p.ke = v[2];
    p.repulsion_switch = v[3];
    p.use_repulsion = v[4] != 0.0;
    return p;
}

void check_geometry_inputs(const at::Tensor& vec, const at::Tensor& idx_i, const at::Tensor& idx_j, const at::Tensor& z,
                           const at::Tensor& floor_d, const at::Tensor& nlh_a, const at::Tensor& nlh_b, const at::Tensor& columns) {
    TORCH_CHECK(vec.is_cuda() && vec.dim() == 2 && vec.size(1) == 3, "so3lr::edge_geometry: vec [E, 3] on the GPU");
    for (const auto* t : {&floor_d, &nlh_a, &nlh_b}) {
        TORCH_CHECK(t->is_cuda() && t->scalar_type() == vec.scalar_type(), "so3lr::edge_geometry: tables on the GPU, the dtype of vec");
    }
    for (const auto* t : {&idx_i, &idx_j, &z, &columns}) {
        TORCH_CHECK(t->is_cuda() && t->scalar_type() == at::kLong, "so3lr::edge_geometry: index tensors must be int64 on the GPU");
    }
    TORCH_CHECK(idx_i.size(0) == vec.size(0) && idx_j.size(0) == vec.size(0), "so3lr::edge_geometry: one vector per edge");
    TORCH_CHECK(floor_d.dim() == 2 && floor_d.size(0) == floor_d.size(1) && nlh_a.dim() == 3 && nlh_a.size(0) == floor_d.size(0) &&
                    nlh_a.size(1) == floor_d.size(0) && nlh_b.sizes() == nlh_a.sizes(),
                "so3lr::edge_geometry: floor [Z, Z], nlh_a and nlh_b [Z, Z, K]");
}

std::vector<at::Tensor> geometry_forward_cuda(const at::Tensor& vec_, const at::Tensor& idx_i_, const at::Tensor& idx_j_, const at::Tensor& z_,
                                              const at::Tensor& floor_, const at::Tensor& nlh_a_, const at::Tensor& nlh_b_,
                                              const at::Tensor& columns_, int64_t n_atoms, c10::ArrayRef<double> params) {
    const at::cuda::CUDAGuard guard(vec_.device());
    auto vec = vec_.contiguous(), idx_i = idx_i_.contiguous(), idx_j = idx_j_.contiguous(), z = z_.contiguous();
    auto floor_d = floor_.contiguous(), nlh_a = nlh_a_.contiguous(), nlh_b = nlh_b_.contiguous(), columns = columns_.contiguous();
    check_geometry_inputs(vec, idx_i, idx_j, z, floor_d, nlh_a, nlh_b, columns);
    const So3lrGeometryParams p = geometry_params(params);
    const int64_t E = vec.size(0);
    const int M = static_cast<int>(columns.size(0));
    auto ylm = at::empty({E, M}, vec.options());
    auto d_basis = at::empty({E}, vec.options()), cut = at::empty({E}, vec.options());
    auto e_rep = at::zeros({n_atoms}, vec.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(vec.scalar_type(), "so3lr_geometry_forward", [&] {
        const cudaError_t err = so3lr_geometry_forward_launch<scalar_t>(
            vec.data_ptr<scalar_t>(), idx_i.data_ptr<int64_t>(), idx_j.data_ptr<int64_t>(), z.data_ptr<int64_t>(),
            floor_d.data_ptr<scalar_t>(), nlh_a.data_ptr<scalar_t>(), nlh_b.data_ptr<scalar_t>(), static_cast<int>(floor_d.size(0)),
            static_cast<int>(nlh_a.size(2)), columns.data_ptr<int64_t>(), M, E, p, ylm.data_ptr<scalar_t>(), d_basis.data_ptr<scalar_t>(),
            cut.data_ptr<scalar_t>(), e_rep.data_ptr<scalar_t>(), stream);
        TORCH_CHECK(err == cudaSuccess, "so3lr::edge_geometry forward kernel: ", cudaGetErrorString(err));
    });
    return {ylm, d_basis, cut, e_rep};
}

class GeometryFunction : public torch::autograd::Function<GeometryFunction> {
public:
    static torch::autograd::variable_list forward(torch::autograd::AutogradContext* ctx, const at::Tensor& vec, const at::Tensor& idx_i,
                                                  const at::Tensor& idx_j, const at::Tensor& z, const at::Tensor& floor_d,
                                                  const at::Tensor& nlh_a, const at::Tensor& nlh_b, const at::Tensor& columns,
                                                  int64_t n_atoms, c10::ArrayRef<double> params) {
        at::AutoDispatchBelowADInplaceOrView below_autograd;
        auto out = geometry_forward_cuda(vec, idx_i, idx_j, z, floor_d, nlh_a, nlh_b, columns, n_atoms, params);
        ctx->save_for_backward({vec.contiguous(), idx_i.contiguous(), idx_j.contiguous(), z.contiguous(), floor_d.contiguous(),
                                nlh_a.contiguous(), nlh_b.contiguous(), columns.contiguous()});
        ctx->saved_data["params"] = std::vector<double>(params.begin(), params.end());
        return out;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grads) {
        const auto s = ctx->get_saved_variables();
        const at::Tensor &vec = s[0], &idx_i = s[1], &idx_j = s[2], &z = s[3], &floor_d = s[4], &nlh_a = s[5], &nlh_b = s[6], &columns = s[7];
        const So3lrGeometryParams p = geometry_params(ctx->saved_data["params"].toDoubleVector());
        const at::cuda::CUDAGuard guard(vec.device());
        const int64_t E = vec.size(0);
        const int M = static_cast<int>(columns.size(0));
        auto g_ylm = grads[0].defined() ? grads[0].contiguous() : at::zeros({E, M}, vec.options());
        auto g_db = grads[1].defined() ? grads[1].contiguous() : at::zeros({E}, vec.options());
        auto g_cut = grads[2].defined() ? grads[2].contiguous() : at::zeros({E}, vec.options());
        auto g_erep = grads[3].defined() ? grads[3].contiguous() : at::zeros({z.size(0)}, vec.options());
        auto g_vec = at::empty_like(vec);
        auto stream = at::cuda::getCurrentCUDAStream();
        AT_DISPATCH_FLOATING_TYPES(vec.scalar_type(), "so3lr_geometry_backward", [&] {
            const cudaError_t err = so3lr_geometry_backward_launch<scalar_t>(
                g_ylm.data_ptr<scalar_t>(), g_db.data_ptr<scalar_t>(), g_cut.data_ptr<scalar_t>(), g_erep.data_ptr<scalar_t>(),
                vec.data_ptr<scalar_t>(), idx_i.data_ptr<int64_t>(), idx_j.data_ptr<int64_t>(), z.data_ptr<int64_t>(),
                floor_d.data_ptr<scalar_t>(), nlh_a.data_ptr<scalar_t>(), nlh_b.data_ptr<scalar_t>(), static_cast<int>(floor_d.size(0)),
                static_cast<int>(nlh_a.size(2)), columns.data_ptr<int64_t>(), M, E, p, g_vec.data_ptr<scalar_t>(), stream);
            TORCH_CHECK(err == cudaSuccess, "so3lr::edge_geometry backward kernel: ", cudaGetErrorString(err));
        });
        return {g_vec, at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
                at::Tensor()};
    }
};

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> geometry_autograd(const at::Tensor& vec, const at::Tensor& idx_i,
                                                                             const at::Tensor& idx_j, const at::Tensor& z,
                                                                             const at::Tensor& floor_d, const at::Tensor& nlh_a,
                                                                             const at::Tensor& nlh_b, const at::Tensor& columns,
                                                                             int64_t n_atoms, c10::ArrayRef<double> params) {
    auto out = GeometryFunction::apply(vec, idx_i, idx_j, z, floor_d, nlh_a, nlh_b, columns, n_atoms, params);
    return {out[0], out[1], out[2], out[3]};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> geometry_forward_op(const at::Tensor& vec, const at::Tensor& idx_i,
                                                                               const at::Tensor& idx_j, const at::Tensor& z,
                                                                               const at::Tensor& floor_d, const at::Tensor& nlh_a,
                                                                               const at::Tensor& nlh_b, const at::Tensor& columns,
                                                                               int64_t n_atoms, c10::ArrayRef<double> params) {
    auto out = geometry_forward_cuda(vec, idx_i, idx_j, z, floor_d, nlh_a, nlh_b, columns, n_atoms, params);
    return {out[0], out[1], out[2], out[3]};
}

}  // namespace

TORCH_LIBRARY(so3lr, m) {
    m.def("attention(Tensor q1, Tensor k1, Tensor v1, Tensor w1, Tensor q2, Tensor k2, Tensor w2, Tensor cut, Tensor ylm, "
          "Tensor idx_i, Tensor idx_j, Tensor degree_index, float norm1, float norm2) -> (Tensor, Tensor)");
    m.def("pair_energy(Tensor pos, Tensor shift, Tensor idx_i, Tensor idx_j, Tensor q, Tensor alpha, Tensor c6, float[] params) -> Tensor");
    m.def("ev_contraction(Tensor ev, Tensor idx_i, Tensor idx_j, Tensor degree_index, Tensor cg, int num_heads) -> Tensor");
    m.def("radial_filter(Tensor x, Tensor table, float x0, float inv_h) -> Tensor");
    m.def("attention_filtered(Tensor q1, Tensor k1, Tensor v1, Tensor q2, Tensor k2, Tensor cut, Tensor ylm, Tensor x, Tensor s, "
          "Tensor idx_i, Tensor idx_j, Tensor degree_index, Tensor table, float x0, float inv_h, float norm1, float norm2) "
          "-> (Tensor, Tensor)");
    m.def("edge_geometry(Tensor vec, Tensor idx_i, Tensor idx_j, Tensor z, Tensor floor, Tensor nlh_a, Tensor nlh_b, Tensor columns, "
          "int n_atoms, float[] params) -> (Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(so3lr, CUDA, m) {
    m.impl("attention", attention_forward_cuda);
    m.impl("pair_energy", pair_forward_cuda);
    m.impl("ev_contraction", ev_contraction_forward_cuda);
    m.impl("radial_filter", radial_forward_cuda);
    m.impl("attention_filtered", filtered_forward_cuda);
    m.impl("edge_geometry", geometry_forward_op);
}

TORCH_LIBRARY_IMPL(so3lr, Autograd, m) {
    m.impl("attention", attention_autograd);
    m.impl("pair_energy", pair_autograd);
    m.impl("ev_contraction", ev_contraction_autograd);
    m.impl("radial_filter", radial_autograd);
    m.impl("attention_filtered", filtered_autograd);
    m.impl("edge_geometry", geometry_autograd);
}
