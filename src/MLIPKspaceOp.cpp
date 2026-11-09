// dsmd::kspace: smooth particle-mesh Ewald of the per-atom charges a model computes (so3lr --kspace pme), for the electrostatic kernel
// and jaxpme's r^-6 dispersion kernel. A line-by-line ATen version of pme() / kspace_terms() in
// MLIP_tools/mlip_convert/kspace_op.py (the formulas are documented there); the converter checks the model with that file, dsMD runs
// it with this one, so the two must stay the same algorithm (grid sizes, B-splines, kernels, corrections, virial correction).
//
//   pos [n, 3] (A, this rank's owned atoms), charges [n, 2] (q in e; sqrt(C6) in atomic units),
//   params [ke, smearing_e, smearing_d, bohr, hartree, use_elec, use_disp]  ->  energy [n] (eV)
// The derivatives (positions, both charges) are computed in the forward pass and returned by the backward, which needs no
// communication. Only this file's operator touches the context (MLIPKspace.h); no MPI here: the grid sum is the context's callback.

#include "MLIPKspaceOp.h"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/autograd.h>
#include <torch/library.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

MLIPKspaceContext* g_context = nullptr;

constexpr double kPi = 3.14159265358979323846;

bool fft_friendly(int64_t n) {
    for (int64_t f : {2, 3, 5, 7}) {
        while (n % f == 0) n /= f;
    }
    return n == 1;
}

// Points along one axis: at most `spacing` apart, at least 2 x order, factors 2, 3, 5, 7 only (kspace_op.grid_size)
int64_t grid_size(double length, double spacing, int order) {
    int64_t n = std::max<int64_t>(static_cast<int64_t>(std::ceil(length / spacing - 1e-9)), 2 * order);
    while (!fft_friendly(n)) ++n;
    return n;
}

double binomial(int n, int k) {
    double r = 1.0;
    for (int i = 1; i <= k; ++i) r = r * (n - k + i) / i;
    return r;
}

double factorial(int n) {
    double r = 1.0;
    for (int i = 2; i <= n; ++i) r *= i;
    return r;
}

// Cardinal B-spline M_order(y), support [0, order) (kspace_op.bspline)
at::Tensor bspline(const at::Tensor& y, int order) {
    at::Tensor out = at::zeros_like(y);
    for (int j = 0; j <= order; ++j) {
        const double sign = (j % 2 == 0) ? 1.0 : -1.0;
        out = out + sign * binomial(order, j) * at::pow(at::clamp_min(y - j, 0.0), order - 1);
    }
    out = out / factorial(order - 1);
    return at::where((y >= 0) & (y < order), out, at::zeros_like(out));
}

// |sum_{k=0}^{order-2} M_order(k+1) exp(2 pi i m k / n)|^2, m = 0 .. n-1 (kspace_op.bspline_moduli)
at::Tensor bspline_moduli(int64_t n, int order, const at::TensorOptions& options) {
    std::vector<double> w(static_cast<size_t>(order - 1));
    for (int k = 0; k < order - 1; ++k) {
        double v = 0.0;
        const double y = k + 1.0;
        for (int j = 0; j <= order; ++j) {
            const double sign = (j % 2 == 0) ? 1.0 : -1.0;
            v += sign * binomial(order, j) * std::pow(std::max(y - j, 0.0), order - 1);
        }
        w[static_cast<size_t>(k)] = v / factorial(order - 1);
    }
    at::Tensor out = at::empty({n}, at::TensorOptions().dtype(at::kDouble));
    double* o = out.data_ptr<double>();
    for (int64_t m = 0; m < n; ++m) {
        double re = 0.0, im = 0.0;
        for (int k = 0; k < order - 1; ++k) {
            const double angle = 2.0 * kPi * static_cast<double>(m) * k / static_cast<double>(n);
            re += w[static_cast<size_t>(k)] * std::cos(angle);
            im += w[static_cast<size_t>(k)] * std::sin(angle);
        }
        o[m] = re * re + im * im;
    }
    return out.to(options);
}

struct GridResult {
    at::Tensor phi;      // [n] potential incl. self and background corrections (units of the kind)
    at::Tensor dphi;     // [n, 3] per A
    double virial[3];    // W of the mesh + background energy of the whole system, without the prefactor
};

GridResult pme(const at::Tensor& pos, const at::Tensor& c, const double box[3], double spacing, int order, double smearing,
               bool coulomb, double unit, const MLIPKspaceContext& context) {
    const auto opts = pos.options();
    const auto lopts = opts.dtype(at::kLong);
    int64_t ng[3];
    double length[3];
    for (int a = 0; a < 3; ++a) {
        ng[a] = grid_size(box[a], spacing, order);
        length[a] = box[a] * unit;
    }
    const double volume = length[0] * length[1] * length[2];
    const at::Tensor ngt = at::tensor({static_cast<double>(ng[0]), static_cast<double>(ng[1]), static_cast<double>(ng[2])}, opts);
    const at::Tensor boxt = at::tensor({box[0], box[1], box[2]}, opts);
    at::Tensor u = pos / boxt * ngt;
    u = u - at::floor(u / ngt) * ngt;
    at::Tensor base = at::floor(u);
    const at::Tensor f = u - base;
    const at::Tensor base_l = base.to(at::kLong);
    const at::Tensor k = at::arange(order, opts);
    const at::Tensor y = f.unsqueeze(2) + k;                                          // [n, 3, order]
    const at::Tensor w = bspline(y, order);
    const at::Tensor dw = bspline(y, order - 1) - bspline(y - 1.0, order - 1);
    const at::Tensor kk = at::arange(order, lopts);
    const at::Tensor ngl = at::tensor({ng[0], ng[1], ng[2]}, lopts);
    const at::Tensor idx = at::remainder(base_l.unsqueeze(2) - kk, ngl.view({1, 3, 1}));   // [n, 3, order]
    const at::Tensor ix = idx.select(1, 0).view({-1, order, 1, 1});
    const at::Tensor iy = idx.select(1, 1).view({-1, 1, order, 1});
    const at::Tensor iz = idx.select(1, 2).view({-1, 1, 1, order});
    const at::Tensor flat = (ix * ng[1] + iy) * ng[2] + iz;                              // [n, order, order, order]
    const at::Tensor wx = w.select(1, 0).view({-1, order, 1, 1});
    const at::Tensor wy = w.select(1, 1).view({-1, 1, order, 1});
    const at::Tensor wz = w.select(1, 2).view({-1, 1, 1, order});
    const at::Tensor w3 = wx * wy * wz;
    const int64_t total = ng[0] * ng[1] * ng[2];
    at::Tensor grid = at::zeros({total}, opts).index_add_(0, flat.reshape(-1), (c.view({-1, 1, 1, 1}) * w3).reshape(-1));
    grid = grid.view({ng[0], ng[1], ng[2]}).contiguous();
    if (context.grid_sum != nullptr) {
        void* stream = grid.is_cuda() ? static_cast<void*>(c10::cuda::getCurrentCUDAStream(grid.device().index()).stream()) : nullptr;
        context.grid_sum(grid.data_ptr(), total, static_cast<int>(grid.element_size()), grid.is_cuda(), stream);
    }
    const double charge_sum = grid.sum().item<double>();

    const at::Tensor s_hat = at::fft_rfftn(grid);
    const at::Tensor mx = at::fft_fftfreq(ng[0], 1.0, opts) * static_cast<double>(ng[0]);
    const at::Tensor my = at::fft_fftfreq(ng[1], 1.0, opts) * static_cast<double>(ng[1]);
    const at::Tensor mz = at::fft_rfftfreq(ng[2], 1.0, opts) * static_cast<double>(ng[2]);
    const at::Tensor kx2 = at::pow((2.0 * kPi / length[0]) * mx, 2).view({-1, 1, 1});
    const at::Tensor ky2 = at::pow((2.0 * kPi / length[1]) * my, 2).view({1, -1, 1});
    const at::Tensor kz2 = at::pow((2.0 * kPi / length[2]) * mz, 2).view({1, 1, -1});
    const at::Tensor k2 = kx2 + ky2 + kz2;
    const at::Tensor zero = k2 == 0;
    const at::Tensor k2s = at::where(zero, at::ones_like(k2), k2);
    const double s = smearing * unit;
    at::Tensor g, ratio;
    double self_c, background;
    if (coulomb) {
        g = 4.0 * kPi * at::exp(-0.5 * s * s * k2s) / k2s;
        ratio = -2.0 / k2s - s * s;
        self_c = std::sqrt(2.0 / kPi) / s;
        background = kPi * s * s;
    } else {
        const at::Tensor x = 0.5 * s * s * k2s;
        const at::Tensor sx = at::sqrt(x);
        const at::Tensor ex = at::exp(-x);
        const at::Tensor g12 = std::sqrt(kPi) * at::erfc(sx);
        const at::Tensor g_m12 = -2.0 * (g12 - ex / sx);
        const at::Tensor g_m32 = (-2.0 / 3.0) * (g_m12 - ex / (x * sx));
        const double factor = std::pow(kPi, 1.5) / 2.0 * std::pow(2.0 * s * s, -1.5);
        g = factor * g_m32 * x * sx;
        ratio = s * s * (1.5 / x - ex / (g_m32 * x * x * sx));
        const double t = 2.0 * s * s;
        self_c = 1.0 / (6.0 * t * t * t);
        background = std::pow(kPi, 1.5) * std::pow(t, -1.5) / (-6.0);
    }
    g = at::where(zero, at::zeros_like(g), g);
    ratio = at::where(zero, at::zeros_like(ratio), ratio);
    const int64_t nz_half = ng[2] / 2 + 1;
    const at::Tensor b = 1.0 / (bspline_moduli(ng[0], order, opts).view({-1, 1, 1}) * bspline_moduli(ng[1], order, opts).view({1, -1, 1}) *
                                bspline_moduli(ng[2], order, opts).slice(0, 0, nz_half).view({1, 1, -1}));
    const at::Tensor gb = g * b;
    at::Tensor weight = at::full({nz_half}, 2.0, opts);
    weight.index_put_({0}, 1.0);
    if (ng[2] % 2 == 0) weight.index_put_({nz_half - 1}, 1.0);
    const at::Tensor re_im = at::view_as_real(s_hat);
    const at::Tensor s2 = re_im.select(-1, 0).pow(2) + re_im.select(-1, 1).pow(2);
    const at::Tensor e_k = (0.5 / volume) * gb * s2 * weight.view({1, 1, -1});
    const double e_background = -background * charge_sum * charge_sum / volume;
    GridResult r;
    r.virial[0] = (e_k * (1.0 + kx2 * ratio)).sum().item<double>() + e_background;
    r.virial[1] = (e_k * (1.0 + ky2 * ratio)).sum().item<double>() + e_background;
    r.virial[2] = (e_k * (1.0 + kz2 * ratio)).sum().item<double>() + e_background;

    const at::Tensor psi = at::fft_irfftn(gb * s_hat, c10::IntArrayRef({ng[0], ng[1], ng[2]})) * (static_cast<double>(total) / volume);
    const at::Tensor vals = psi.reshape({-1}).index({flat});
    at::Tensor phi = (vals * w3).sum({1, 2, 3});
    const at::Tensor dux = (vals * dw.select(1, 0).view({-1, order, 1, 1}) * wy * wz).sum({1, 2, 3});
    const at::Tensor duy = (vals * wx * dw.select(1, 1).view({-1, 1, order, 1}) * wz).sum({1, 2, 3});
    const at::Tensor duz = (vals * wx * wy * dw.select(1, 2).view({-1, 1, 1, order})).sum({1, 2, 3});
    r.dphi = at::stack({dux * (ng[0] / box[0]), duy * (ng[1] / box[1]), duz * (ng[2] / box[2])}, 1);
    r.phi = phi - c * self_c - 2.0 * background * charge_sum / volume;
    return r;
}

struct Terms {
    at::Tensor energy, d_pos, d_charges;
    double correction[3] = {0.0, 0.0, 0.0};
};

MLIPKspaceContext& require_context() {
    TORCH_CHECK(g_context != nullptr,
                "dsmd::kspace: no k-space context (box, grid) is set: the model was converted with --kspace pme, so the dsMD input needs "
                "mlip_kspace: pme");
    return *g_context;
}

Terms kspace_terms(const at::Tensor& pos_in, const at::Tensor& charges_in, c10::ArrayRef<double> params, MLIPKspaceContext& context) {
    TORCH_CHECK(params.size() == 7, "dsmd::kspace: params must have 7 values, got ", params.size());
    TORCH_CHECK(pos_in.dim() == 2 && pos_in.size(1) == 3 && charges_in.dim() == 2 && charges_in.size(1) == 2 &&
                    charges_in.size(0) == pos_in.size(0),
                "dsmd::kspace: pos [n, 3] and charges [n, 2]");
    const at::Tensor pos = pos_in.contiguous();
    const at::Tensor charges = charges_in.contiguous();
    const double ke = params[0], s_e = params[1], s_d = params[2], bohr = params[3], hartree = params[4];
    const bool use_elec = params[5] != 0.0, use_disp = params[6] != 0.0;
    Terms t;
    t.energy = at::zeros({pos.size(0)}, pos.options());
    t.d_pos = at::zeros_like(pos);
    t.d_charges = at::zeros_like(charges);
    for (int column = 0; column < 2; ++column) {
        if ((column == 0 && !use_elec) || (column == 1 && !use_disp)) continue;
        const bool coulomb = column == 0;
        const double smearing = coulomb ? s_e : s_d;
        const double unit = coulomb ? 1.0 : 1.0 / bohr;
        const double spacing = coulomb ? context.grid_spacing : context.dispersion_grid_spacing;
        const double prefactor = coulomb ? ke : -hartree;
        const at::Tensor c = charges.select(1, column);
        GridResult r = pme(pos, c, context.box, spacing, context.order, smearing, coulomb, unit, context);
        t.energy = t.energy + 0.5 * prefactor * c * r.phi;
        t.d_charges.select(1, column).copy_(prefactor * r.phi);
        const at::Tensor de_dx = prefactor * c.unsqueeze(1) * r.dphi;
        t.d_pos = t.d_pos + de_dx;
        const at::Tensor xdx = (pos * de_dx).sum(0).to(at::kDouble).cpu();
        for (int a = 0; a < 3; ++a) {
            t.correction[a] += prefactor * r.virial[a] / context.grid_ranks + xdx[a].item<double>();
        }
    }
    return t;
}

at::Tensor kspace_forward(const at::Tensor& pos, const at::Tensor& charges, c10::ArrayRef<double> params) {
    MLIPKspaceContext& context = require_context();
    Terms t = kspace_terms(pos.detach(), charges.detach(), params, context);
    for (int a = 0; a < 3; ++a) context.virial_correction[a] += t.correction[a];
    ++context.calls;
    return t.energy;
}

class KspaceFunction : public torch::autograd::Function<KspaceFunction> {
public:
    static at::Tensor forward(torch::autograd::AutogradContext* ctx, const at::Tensor& pos, const at::Tensor& charges,
                              c10::ArrayRef<double> params) {
        MLIPKspaceContext& context = require_context();
        at::AutoDispatchBelowAutograd below_autograd;
        Terms t = kspace_terms(pos.detach(), charges.detach(), params, context);
        for (int a = 0; a < 3; ++a) context.virial_correction[a] += t.correction[a];
        ++context.calls;
        ctx->save_for_backward({t.d_pos, t.d_charges});
        return t.energy;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grads) {
        const auto saved = ctx->get_saved_variables();
        const at::Tensor& grad = grads[0];
        // the energy enters the total with one common weight on every row (kspace_op._Kspace.backward)
        at::Tensor g = grad.numel() > 0 ? grad.slice(0, 0, 1).view({1, 1}) : at::zeros({1, 1}, saved[0].options());
        if (grad.numel() > 0) {
            const double g0 = grad[0].item<double>();
            const double spread = (grad - grad[0]).abs().max().item<double>();
            TORCH_CHECK(spread <= 1e-6 * std::max(1.0, std::abs(g0)),
                        "dsmd::kspace: the per-atom energies must enter the total with one common weight");
        }
        return {g * saved[0], g * saved[1], at::Tensor()};
    }
};

at::Tensor kspace_autograd(const at::Tensor& pos, const at::Tensor& charges, c10::ArrayRef<double> params) {
    return KspaceFunction::apply(pos, charges, params);
}

}  // namespace

// FRAGMENT: the Python side (kspace_op.py) registers the same schema in a converter process; inside dsMD this one is registered first.
TORCH_LIBRARY_FRAGMENT(dsmd, m) { m.def("kspace(Tensor pos, Tensor charges, float[] params) -> Tensor"); }

TORCH_LIBRARY_IMPL(dsmd, CompositeExplicitAutograd, m) { m.impl("kspace", &kspace_forward); }

TORCH_LIBRARY_IMPL(dsmd, Autograd, m) { m.impl("kspace", &kspace_autograd); }

void ensure_kspace_op_registered() {}

KspaceScope::KspaceScope(MLIPKspaceContext* context) : previous_(g_context) { g_context = context; }

KspaceScope::~KspaceScope() { g_context = previous_; }
