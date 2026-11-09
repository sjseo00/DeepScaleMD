// CUDA kernels of the SO3LR operators (accel so3lr_fused: attention and pair terms): kernels and plain launch functions only. The PyTorch operator,
// its autograd and the checks are in so3lr_fused_ops.cpp, compiled by the host compiler (nvcc 11.5 with gcc 11 cannot compile the
// torch headers). Both files are built by the converter as one torch extension (mlip_convert/accel.py), not by the dsMD CMake build.
// What the operator computes: see so3lr_fused_ops.cpp.
#include <cstdint>
#include <cuda_runtime.h>

#include "so3lr_pair_params.h"

namespace {

constexpr int kMaxDegreeHeads = 16;
constexpr int kWarpsPerBlock = 8;

template <typename T>
__device__ __forceinline__ T warp_sum(T value) {
    for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor_sync(0xffffffffu, value, offset);
    return value;
}

// Runs of equal keys on consecutive lanes of a warp (one edge per lane, key = the atom an edge adds to). The edge lists of dsMD are
// grouped by atom i, so the lanes of a warp mostly add to one atom; summing a run first and letting its first lane do one atomic add
// replaces up to 32 atomic adds on the same address. Any key order is correct (a key that comes back later is another run).
// Every lane of the warp must call these (no lane may have returned).
struct WarpRun {
    int end;      // last lane of this lane's run
    bool head;    // first lane of its run: it holds the run's sum after run_sum
};

__device__ __forceinline__ WarpRun warp_run(int64_t key) {
    const int lane = threadIdx.x & 31;
    const int64_t next = __shfl_down_sync(0xffffffffu, key, 1);
    const unsigned last = __ballot_sync(0xffffffffu, lane == 31 || next != key);   // bit l: lane l ends a run
    WarpRun run;
    run.end = __ffs(last & (0xffffffffu << lane)) - 1;
    run.head = lane == 0 || ((last >> (lane - 1)) & 1u);
    return run;
}

// After this, the head lane of each run holds the sum of value over its run (other lanes hold partial sums).
template <typename T>
__device__ __forceinline__ T run_sum(T value, const WarpRun& run) {
    const int lane = threadIdx.x & 31;
    for (int offset = 1; offset < 32; offset <<= 1) {
        const T other = __shfl_down_sync(0xffffffffu, value, offset);
        if (lane + offset <= run.end) value += other;
    }
    return value;
}

template <typename T>
__global__ void attention_forward_kernel(const T* __restrict__ q1, const T* __restrict__ k1, const T* __restrict__ v1,
                                         const T* __restrict__ w1, const T* __restrict__ q2, const T* __restrict__ k2,
                                         const T* __restrict__ w2, const T* __restrict__ cut, const T* __restrict__ ylm,
                                         const int64_t* __restrict__ idx_i, const int64_t* __restrict__ idx_j,
                                         const int64_t* __restrict__ degree_index, int64_t n_edges, int H, int C1, int D, int C2,
                                         int M, T inv_norm1, T inv_norm2, T* __restrict__ x_att, T* __restrict__ ev_att) {
    const int lane = threadIdx.x & 31;
    const int64_t e = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
    if (e >= n_edges) return;
    const int64_t i = idx_i[e];
    const int64_t j = idx_j[e];
    const T c = cut[e];
    for (int h = 0; h < H; ++h) {
        const int64_t ni = (i * H + h) * C1, nj = (j * H + h) * C1, ne = (e * H + h) * C1;
        T part = T(0);
        for (int f = lane; f < C1; f += 32) part += q1[ni + f] * w1[ne + f] * k1[nj + f];
        const T alpha = warp_sum(part) * inv_norm1 * c;
        for (int f = lane; f < C1; f += 32) atomicAdd(&x_att[ni + f], alpha * v1[nj + f]);
    }
    T alpha2[kMaxDegreeHeads];
    for (int d = 0; d < D; ++d) {
        const int64_t ni = (i * D + d) * C2, nj = (j * D + d) * C2, ne = (e * D + d) * C2;
        T part = T(0);
        for (int f = lane; f < C2; f += 32) part += q2[ni + f] * w2[ne + f] * k2[nj + f];
        alpha2[d] = warp_sum(part) * inv_norm2 * c;
    }
    for (int m = lane; m < M; m += 32) atomicAdd(&ev_att[i * M + m], alpha2[degree_index[m]] * ylm[e * M + m]);
}

template <typename T>
__global__ void attention_backward_kernel(const T* __restrict__ g_x, const T* __restrict__ g_ev, const T* __restrict__ q1,
                                          const T* __restrict__ k1, const T* __restrict__ v1, const T* __restrict__ w1,
                                          const T* __restrict__ q2, const T* __restrict__ k2, const T* __restrict__ w2,
                                          const T* __restrict__ cut, const T* __restrict__ ylm, const int64_t* __restrict__ idx_i,
                                          const int64_t* __restrict__ idx_j, const int64_t* __restrict__ degree_index,
                                          int64_t n_edges, int H, int C1, int D, int C2, int M, T inv_norm1, T inv_norm2,
                                          T* __restrict__ gq1, T* __restrict__ gk1, T* __restrict__ gv1, T* __restrict__ gw1,
                                          T* __restrict__ gq2, T* __restrict__ gk2, T* __restrict__ gw2, T* __restrict__ gcut,
                                          T* __restrict__ gylm) {
    const int lane = threadIdx.x & 31;
    const int64_t e = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
    if (e >= n_edges) return;
    const int64_t i = idx_i[e];
    const int64_t j = idx_j[e];
    const T c = cut[e];
    T grad_cut = T(0);
    for (int h = 0; h < H; ++h) {
        const int64_t ni = (i * H + h) * C1, nj = (j * H + h) * C1, ne = (e * H + h) * C1;
        T raw_part = T(0), dalpha_part = T(0);
        for (int f = lane; f < C1; f += 32) {
            raw_part += q1[ni + f] * w1[ne + f] * k1[nj + f];
            dalpha_part += g_x[ni + f] * v1[nj + f];
        }
        const T raw = warp_sum(raw_part);
        const T dalpha = warp_sum(dalpha_part);
        const T alpha = raw * inv_norm1 * c;
        const T s = dalpha * inv_norm1 * c;
        grad_cut += dalpha * raw * inv_norm1;
        for (int f = lane; f < C1; f += 32) {
            const T q = q1[ni + f], w = w1[ne + f], k = k1[nj + f];
            atomicAdd(&gv1[nj + f], alpha * g_x[ni + f]);
            atomicAdd(&gq1[ni + f], s * w * k);
            atomicAdd(&gk1[nj + f], s * q * w);
            gw1[ne + f] = s * q * k;
        }
    }
    T dalpha2[kMaxDegreeHeads];
    T alpha2[kMaxDegreeHeads];
    for (int d = 0; d < D; ++d) dalpha2[d] = T(0);
    for (int m = lane; m < M; m += 32) dalpha2[degree_index[m]] += g_ev[i * M + m] * ylm[e * M + m];
    for (int d = 0; d < D; ++d) {
        dalpha2[d] = warp_sum(dalpha2[d]);
        const int64_t ni = (i * D + d) * C2, nj = (j * D + d) * C2, ne = (e * D + d) * C2;
        T raw_part = T(0);
        for (int f = lane; f < C2; f += 32) raw_part += q2[ni + f] * w2[ne + f] * k2[nj + f];
        const T raw = warp_sum(raw_part);
        alpha2[d] = raw * inv_norm2 * c;
        const T s = dalpha2[d] * inv_norm2 * c;
        grad_cut += dalpha2[d] * raw * inv_norm2;
        for (int f = lane; f < C2; f += 32) {
            const T q = q2[ni + f], w = w2[ne + f], k = k2[nj + f];
            atomicAdd(&gq2[ni + f], s * w * k);
            atomicAdd(&gk2[nj + f], s * q * w);
            gw2[ne + f] = s * q * k;
        }
    }
    for (int m = lane; m < M; m += 32) gylm[e * M + m] = alpha2[degree_index[m]] * g_ev[i * M + m];
    if (lane == 0) gcut[e] = grad_cut;
}

}  // namespace

template <typename T>
cudaError_t so3lr_attention_forward_launch(const T* q1, const T* k1, const T* v1, const T* w1, const T* q2, const T* k2, const T* w2,
                                           const T* cut, const T* ylm, const int64_t* idx_i, const int64_t* idx_j,
                                           const int64_t* degree_index, int64_t n_edges, int H, int C1, int D, int C2, int M,
                                           double norm1, double norm2, T* x_att, T* ev_att, cudaStream_t stream) {
    if (n_edges == 0) return cudaSuccess;
    const dim3 block(32 * kWarpsPerBlock);
    const dim3 grid(static_cast<unsigned>((n_edges + kWarpsPerBlock - 1) / kWarpsPerBlock));
    attention_forward_kernel<T><<<grid, block, 0, stream>>>(q1, k1, v1, w1, q2, k2, w2, cut, ylm, idx_i, idx_j, degree_index, n_edges,
                                                          H, C1, D, C2, M, static_cast<T>(1.0 / norm1), static_cast<T>(1.0 / norm2),
                                                          x_att, ev_att);
    return cudaGetLastError();
}

template <typename T>
cudaError_t so3lr_attention_backward_launch(const T* g_x, const T* g_ev, const T* q1, const T* k1, const T* v1, const T* w1,
                                            const T* q2, const T* k2, const T* w2, const T* cut, const T* ylm, const int64_t* idx_i,
                                            const int64_t* idx_j, const int64_t* degree_index, int64_t n_edges, int H, int C1, int D,
                                            int C2, int M, double norm1, double norm2, T* gq1, T* gk1, T* gv1, T* gw1, T* gq2, T* gk2,
                                            T* gw2, T* gcut, T* gylm, cudaStream_t stream) {
    if (n_edges == 0) return cudaSuccess;
    const dim3 block(32 * kWarpsPerBlock);
    const dim3 grid(static_cast<unsigned>((n_edges + kWarpsPerBlock - 1) / kWarpsPerBlock));
    attention_backward_kernel<T><<<grid, block, 0, stream>>>(g_x, g_ev, q1, k1, v1, w1, q2, k2, w2, cut, ylm, idx_i, idx_j, degree_index,
                                                           n_edges, H, C1, D, C2, M, static_cast<T>(1.0 / norm1),
                                                           static_cast<T>(1.0 / norm2), gq1, gk1, gv1, gw1, gq2, gk2, gw2, gcut, gylm);
    return cudaGetLastError();
}

template cudaError_t so3lr_attention_forward_launch<float>(const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const int64_t*, const int64_t*, const int64_t*, int64_t, int, int, int, int,
    int, double, double, float*, float*, cudaStream_t);
template cudaError_t so3lr_attention_forward_launch<double>(const double*, const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*, const int64_t*, const int64_t*, const int64_t*, int64_t, int, int, int,
    int, int, double, double, double*, double*, cudaStream_t);
template cudaError_t so3lr_attention_backward_launch<float>(const float*, const float*, const float*, const float*, const float*,
    const float*, const float*, const float*, const float*, const float*, const float*, const int64_t*, const int64_t*, const int64_t*,
    int64_t, int, int, int, int, int, double, double, float*, float*, float*, float*, float*, float*, float*, float*, float*,
    cudaStream_t);
template cudaError_t so3lr_attention_backward_launch<double>(const double*, const double*, const double*, const double*, const double*,
    const double*, const double*, const double*, const double*, const double*, const double*, const int64_t*, const int64_t*,
    const int64_t*, int64_t, int, int, int, int, int, double, double, double*, double*, double*, double*, double*, double*, double*,
    double*, double*, cudaStream_t);

// ---------------------------------------------------------------------------------------------------------------------- pair terms
// so3lr::pair_energy: SO3LR's long-range pair terms on the edge list, one thread per edge (see so3lr_fused_ops.cpp for the formulas).

namespace {

constexpr int kPairThreads = 256;

// so3lr switching_fn: 1 below on, 0 above off, exp(-1/u) steps in between; value and derivative in x
template <typename T>
__device__ __forceinline__ void switch_fn(T x, T on, T off, T& s, T& ds) {
    const T width = off - on;
    const T c = (x - on) / width;
    const T a = T(1) - c;
    T sa = T(0), dsa = T(0), sc = T(0), dsc = T(0);
    if (a > T(0)) { sa = exp(-T(1) / a); dsa = sa / (a * a); }
    if (c > T(0)) { sc = exp(-T(1) / c); dsc = sc / (c * c); }
    const T sum = sa + sc;
    s = sa / sum;
    ds = (-dsa * sc - sa * dsc) / (sum * sum) / width;
}

// Energy of one directed edge (counted half, as so3lr's 'sparse' neighbor list) and, with GRAD, its derivatives in r and in the
// per-atom inputs of both ends.
template <typename T, bool GRAD>
__device__ __forceinline__ void pair_terms(T r, T qi, T qj, T ai, T aj, T ci, T cj, const So3lrPairParams& p, T& e, T& de_dr, T& de_dqi,
                                           T& de_dqj, T& de_dai, T& de_daj, T& de_dci, T& de_dcj) {
    e = de_dr = de_dqi = de_dqj = de_dai = de_daj = de_dci = de_dcj = T(0);
    const T rc = T(p.lr_cut);
    if (!(r < rc)) return;
    if (p.use_elec) {      // erf(r/sigma)/r, shifted force, switched between cuton and the cutoff
        const T sig = T(p.sigma);
        const T erfv = erf(r / sig);
        T pot = erfv / r;
        T erfs = T(0);
        if (p.pme_smear > 0.0) {            // Ewald split: the smeared part is in the k-space sum
            erfs = erf(r / T(p.pme_smear));
            pot -= erfs / r;
        }
        T f, df;
        switch_fn<T>(r, T(p.cuton), rc, f, df);
        const T fshift = T(p.coul_fshift);
        const T g = pot - T(p.coul_shift) - (T(1) - f) * fshift * (r - rc);
        const T pre = T(0.5) * T(p.ke);
        e += pre * qi * qj * g;
        if (GRAD) {
            const T x = r / sig;
            T dpot = T(1.1283791670955126) / sig * exp(-x * x) / r - erfv / (r * r);   // 2/sqrt(pi)
            if (p.pme_smear > 0.0) {
                const T ps = T(p.pme_smear);
                const T xs = r / ps;
                dpot -= T(1.1283791670955126) / ps * exp(-xs * xs) / r - erfs / (r * r);
            }
            const T dg = dpot + df * fshift * (r - rc) - (T(1) - f) * fshift;
            de_dr += pre * qi * qj * dg;
            de_dqi += pre * qj * g;
            de_dqj += pre * qi * g;
        }
    }
    if (p.use_disp) {      // QDO-damped C6/C8/C10 with Casimir-Polder mixing, switched off over the last damping width
        const T A = T(0.5) * (ai + aj);
        const T den = ai * ai * cj + aj * aj * ci;
        const T num = T(2) * ci * cj * ai * aj;
        const T C6 = num / den;
        const T a17 = pow(A, T(1) / T(7));
        const T R = T(p.vdw_factor) * a17;
        const T sg = ((T(-0.00078893) * R + T(0.04125273)) * R + T(0.24428889)) * R - T(0.00433008);
        const T s2 = sg * sg;
        const T pp = T(p.disp_scale) * T(2.0 * 2.54) * a17;
        const T rb = r / T(p.bohr);
        const T rb2 = rb * rb, rb6 = rb2 * rb2 * rb2, rb8 = rb6 * rb2, rb10 = rb8 * rb2;
        const T p2 = pp * pp, p6 = p2 * p2 * p2, p8 = p6 * p2, p10 = p8 * p2;
        const T d6 = rb6 + p6, d8 = rb8 + p8, d10 = rb10 + p10;
        // gamma = 1 / (2 sigma^2): C8 = 5 C6 / gamma = 10 sigma^2 C6, C10 = 245/8 C6 / gamma^2 = 122.5 sigma^4 C6
        const T tsum = T(1) / d6 + T(10) * s2 / d8 + T(122.5) * s2 * s2 / d10;
        T V = -C6 * tsum;
        // Ewald split of the dispersion: + sqrt(c6_i c6_j) h(x) / rb^6, h = 1 - exp(-x)(1 + x + x^2/2), x = rb^2 / (2 s^2)
        T c6g = T(0), hlr = T(0), dhlr_drb = T(0);
        if (p.disp_smear > 0.0) {
            const T sb2 = T(p.disp_smear) * T(p.disp_smear);
            const T rb2_ = (r / T(p.bohr)) * (r / T(p.bohr));
            const T xx = rb2_ / (T(2) * sb2);
            const T ex = exp(-xx);
            const T h = T(1) - ex * (T(1) + xx + T(0.5) * xx * xx);
            const T rbv = r / T(p.bohr);
            const T rb6_ = rb2_ * rb2_ * rb2_;
            c6g = sqrt(ci * cj);
            hlr = h / rb6_;
            dhlr_drb = (ex * T(0.5) * xx * xx) * (rbv / sb2) / rb6_ - T(6) * h / (rb6_ * rbv);
            V += c6g * hlr;
        }
        T sw, dsw;
        switch_fn<T>(r, T(p.disp_on), rc, sw, dsw);
        const T hh = T(0.5) * T(p.hartree);
        e += hh * V * sw;
        if (GRAD) {
            const T rb5 = rb2 * rb2 * rb, rb7 = rb5 * rb2, rb9 = rb7 * rb2;
            const T dV_drb = C6 * (T(6) * rb5 / (d6 * d6) + T(80) * s2 * rb7 / (d8 * d8) + T(1225) * s2 * s2 * rb9 / (d10 * d10))
                             + c6g * dhlr_drb;
            de_dr += hh * (dV_drb / T(p.bohr) * sw + V * dsw);
            const T p5 = p2 * p2 * pp, p7 = p5 * p2, p9 = p7 * p2;
            const T dT_dsg = T(20) * sg / d8 + T(490) * s2 * sg / d10;
            const T dT_dp = -(T(6) * p5 / (d6 * d6) + T(80) * s2 * p7 / (d8 * d8) + T(1225) * s2 * s2 * p9 / (d10 * d10));
            const T dsg_dR = (T(3) * T(-0.00078893) * R + T(2) * T(0.04125273)) * R + T(0.24428889);
            const T dV_dA = -C6 * (dT_dsg * dsg_dR * R + dT_dp * pp) / (T(7) * A);
            const T dV_dC6 = -tsum;
            const T den2 = den * den;
            const T dC6_dci = (T(2) * cj * ai * aj * den - num * aj * aj) / den2;
            const T dC6_dcj = (T(2) * ci * ai * aj * den - num * ai * ai) / den2;
            const T dC6_dai = (T(2) * ci * cj * aj * den - num * T(2) * ai * cj) / den2;
            const T dC6_daj = (T(2) * ci * cj * ai * den - num * T(2) * aj * ci) / den2;
            const T k = hh * sw;
            de_dai += k * (T(0.5) * dV_dA + dV_dC6 * dC6_dai);
            de_daj += k * (T(0.5) * dV_dA + dV_dC6 * dC6_daj);
            de_dci += k * dV_dC6 * dC6_dci;
            de_dcj += k * dV_dC6 * dC6_dcj;
            if (c6g > T(0)) {           // d sqrt(c6_i c6_j) / dc6_i = c6g / (2 c6_i)
                de_dci += k * hlr * c6g / (T(2) * ci);
                de_dcj += k * hlr * c6g / (T(2) * cj);
            }
        }
    }
}

template <typename T>
__global__ void pair_forward_kernel(const T* __restrict__ pos, const T* __restrict__ shift, const int64_t* __restrict__ idx_i,
                                    const int64_t* __restrict__ idx_j, const T* __restrict__ q, const T* __restrict__ alpha,
                                    const T* __restrict__ c6, int64_t n_edges, So3lrPairParams p, T* __restrict__ energy) {
    const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool valid = e < n_edges;
    const int64_t i = valid ? idx_i[e] : -1;
    T en = T(0);
    if (valid) {
        const int64_t j = idx_j[e];
        const T vx = pos[3 * j] - pos[3 * i] + shift[3 * e];
        const T vy = pos[3 * j + 1] - pos[3 * i + 1] + shift[3 * e + 1];
        const T vz = pos[3 * j + 2] - pos[3 * i + 2] + shift[3 * e + 2];
        const T r = sqrt(vx * vx + vy * vy + vz * vz);
        T d0, d1, d2, d3, d4, d5, d6;
        pair_terms<T, false>(r, q[i], q[j], alpha[i], alpha[j], c6[i], c6[j], p, en, d0, d1, d2, d3, d4, d5, d6);
    }
    const WarpRun run = warp_run(i);
    en = run_sum(en, run);
    if (run.head && i >= 0 && en != T(0)) atomicAdd(&energy[i], en);
}

template <typename T>
__global__ void pair_backward_kernel(const T* __restrict__ g_energy, const T* __restrict__ pos, const T* __restrict__ shift,
                                     const int64_t* __restrict__ idx_i, const int64_t* __restrict__ idx_j, const T* __restrict__ q,
                                     const T* __restrict__ alpha, const T* __restrict__ c6, int64_t n_edges, So3lrPairParams p,
                                     T* __restrict__ g_pos, T* __restrict__ g_shift, T* __restrict__ g_q, T* __restrict__ g_alpha,
                                     T* __restrict__ g_c6) {
    const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool valid = e < n_edges;
    const int64_t i = valid ? idx_i[e] : -1;
    // this edge's terms on atom i (summed over the run of lanes with the same i, then one atomic add) and on atom j (atomic adds)
    T fx = T(0), fy = T(0), fz = T(0), gqi = T(0), gai = T(0), gci = T(0);
    if (valid) {
        const int64_t j = idx_j[e];
        const T g = g_energy[i];
        if (g == T(0)) {
            g_shift[3 * e] = g_shift[3 * e + 1] = g_shift[3 * e + 2] = T(0);
        } else {
            const T vx = pos[3 * j] - pos[3 * i] + shift[3 * e];
            const T vy = pos[3 * j + 1] - pos[3 * i + 1] + shift[3 * e + 1];
            const T vz = pos[3 * j + 2] - pos[3 * i + 2] + shift[3 * e + 2];
            const T r = sqrt(vx * vx + vy * vy + vz * vz);
            T en, de_dr, dqi, dqj, dai, daj, dci, dcj;
            pair_terms<T, true>(r, q[i], q[j], alpha[i], alpha[j], c6[i], c6[j], p, en, de_dr, dqi, dqj, dai, daj, dci, dcj);
            const T s = g * de_dr / r;
            fx = s * vx;
            fy = s * vy;
            fz = s * vz;
            g_shift[3 * e] = fx;
            g_shift[3 * e + 1] = fy;
            g_shift[3 * e + 2] = fz;
            if (s != T(0)) {
                atomicAdd(&g_pos[3 * j], fx);
                atomicAdd(&g_pos[3 * j + 1], fy);
                atomicAdd(&g_pos[3 * j + 2], fz);
            }
            if (p.use_elec && dqj != T(0)) atomicAdd(&g_q[j], g * dqj);
            if (p.use_disp && (daj != T(0) || dcj != T(0))) {
                atomicAdd(&g_alpha[j], g * daj);
                atomicAdd(&g_c6[j], g * dcj);
            }
            gqi = g * dqi;
            gai = g * dai;
            gci = g * dci;
        }
    }
    const WarpRun run = warp_run(i);
    fx = run_sum(fx, run);
    fy = run_sum(fy, run);
    fz = run_sum(fz, run);
    gqi = run_sum(gqi, run);
    gai = run_sum(gai, run);
    gci = run_sum(gci, run);
    if (run.head && i >= 0) {
        if (fx != T(0) || fy != T(0) || fz != T(0)) {
            atomicAdd(&g_pos[3 * i], -fx);
            atomicAdd(&g_pos[3 * i + 1], -fy);
            atomicAdd(&g_pos[3 * i + 2], -fz);
        }
        if (p.use_elec && gqi != T(0)) atomicAdd(&g_q[i], gqi);
        if (p.use_disp && (gai != T(0) || gci != T(0))) {
            atomicAdd(&g_alpha[i], gai);
            atomicAdd(&g_c6[i], gci);
        }
    }
}

}  // namespace

template <typename T>
cudaError_t so3lr_pair_forward_launch(const T* pos, const T* shift, const int64_t* idx_i, const int64_t* idx_j, const T* q,
                                      const T* alpha, const T* c6, int64_t n_edges, const So3lrPairParams& p, T* energy,
                                      cudaStream_t stream) {
    if (n_edges == 0) return cudaSuccess;
    const unsigned blocks = static_cast<unsigned>((n_edges + kPairThreads - 1) / kPairThreads);
    pair_forward_kernel<T><<<blocks, kPairThreads, 0, stream>>>(pos, shift, idx_i, idx_j, q, alpha, c6, n_edges, p, energy);
    return cudaGetLastError();
}

template <typename T>
cudaError_t so3lr_pair_backward_launch(const T* g_energy, const T* pos, const T* shift, const int64_t* idx_i, const int64_t* idx_j,
                                       const T* q, const T* alpha, const T* c6, int64_t n_edges, const So3lrPairParams& p, T* g_pos,
                                       T* g_shift, T* g_q, T* g_alpha, T* g_c6, cudaStream_t stream) {
    if (n_edges == 0) return cudaSuccess;
    const unsigned blocks = static_cast<unsigned>((n_edges + kPairThreads - 1) / kPairThreads);
    pair_backward_kernel<T><<<blocks, kPairThreads, 0, stream>>>(g_energy, pos, shift, idx_i, idx_j, q, alpha, c6, n_edges, p, g_pos,
                                                               g_shift, g_q, g_alpha, g_c6);
    return cudaGetLastError();
}

template cudaError_t so3lr_pair_forward_launch<float>(const float*, const float*, const int64_t*, const int64_t*, const float*,
    const float*, const float*, int64_t, const So3lrPairParams&, float*, cudaStream_t);
template cudaError_t so3lr_pair_forward_launch<double>(const double*, const double*, const int64_t*, const int64_t*, const double*,
    const double*, const double*, int64_t, const So3lrPairParams&, double*, cudaStream_t);
template cudaError_t so3lr_pair_backward_launch<float>(const float*, const float*, const float*, const int64_t*, const int64_t*,
    const float*, const float*, const float*, int64_t, const So3lrPairParams&, float*, float*, float*, float*, float*, cudaStream_t);
template cudaError_t so3lr_pair_backward_launch<double>(const double*, const double*, const double*, const int64_t*, const int64_t*,
    const double*, const double*, const double*, int64_t, const So3lrPairParams&, double*, double*, double*, double*, double*,
    cudaStream_t);

// ---------------------------------------------------------------------------------------------------------------- ev contraction
// so3lr::ev_contraction: out[e, d] = sum over the ylm columns m of degree head d of cg[m] (ev[j, m] - ev[i, m])^2, one thread per edge.

namespace {

constexpr int kEvThreads = 256;
constexpr int kEvMaxHeads = 16;

template <typename T>
__global__ void ev_contraction_forward_kernel(const T* __restrict__ ev, const int64_t* __restrict__ idx_i, const int64_t* __restrict__ idx_j,
                                              const int64_t* __restrict__ degree_index, const T* __restrict__ cg, int64_t n_edges, int M,
                                              int D, T* __restrict__ out) {
    const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;
    const int64_t i = idx_i[e], j = idx_j[e];
    T acc[kEvMaxHeads];
    for (int d = 0; d < D; ++d) acc[d] = T(0);
    for (int m = 0; m < M; ++m) {
        const T diff = ev[j * M + m] - ev[i * M + m];
        acc[degree_index[m]] += cg[m] * diff * diff;
    }
    for (int d = 0; d < D; ++d) out[e * D + d] = acc[d];
}

template <typename T>
__global__ void ev_contraction_backward_kernel(const T* __restrict__ g_out, const T* __restrict__ ev, const int64_t* __restrict__ idx_i,
                                               const int64_t* __restrict__ idx_j, const int64_t* __restrict__ degree_index,
                                               const T* __restrict__ cg, int64_t n_edges, int M, int D, T* __restrict__ g_ev) {
    const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool valid = e < n_edges;
    const int64_t i = valid ? idx_i[e] : -1;
    const int64_t j = valid ? idx_j[e] : -1;
    const WarpRun run = warp_run(i);       // the terms on atom i are summed over the run of lanes with the same i (see warp_run)
    for (int m = 0; m < M; ++m) {
        T v = T(0);
        if (valid) {
            const T g = g_out[e * D + degree_index[m]];
            v = T(2) * cg[m] * (ev[j * M + m] - ev[i * M + m]) * g;
            if (v != T(0)) atomicAdd(&g_ev[j * M + m], v);
        }
        v = run_sum(v, run);
        if (run.head && i >= 0 && v != T(0)) atomicAdd(&g_ev[i * M + m], -v);
    }
}

}  // namespace

template <typename T>
cudaError_t so3lr_ev_contraction_forward_launch(const T* ev, const int64_t* idx_i, const int64_t* idx_j, const int64_t* degree_index,
                                                const T* cg, int64_t n_edges, int M, int D, T* out, cudaStream_t stream) {
    if (n_edges == 0) return cudaSuccess;
    if (D > kEvMaxHeads) return cudaErrorInvalidValue;
    const unsigned blocks = static_cast<unsigned>((n_edges + kEvThreads - 1) / kEvThreads);
    ev_contraction_forward_kernel<T><<<blocks, kEvThreads, 0, stream>>>(ev, idx_i, idx_j, degree_index, cg, n_edges, M, D, out);
    return cudaGetLastError();
}

template <typename T>
cudaError_t so3lr_ev_contraction_backward_launch(const T* g_out, const T* ev, const int64_t* idx_i, const int64_t* idx_j,
                                                 const int64_t* degree_index, const T* cg, int64_t n_edges, int M, int D, T* g_ev,
                                                 cudaStream_t stream) {
    if (n_edges == 0) return cudaSuccess;
    const unsigned blocks = static_cast<unsigned>((n_edges + kEvThreads - 1) / kEvThreads);
    ev_contraction_backward_kernel<T><<<blocks, kEvThreads, 0, stream>>>(g_out, ev, idx_i, idx_j, degree_index, cg, n_edges, M, D, g_ev);
    return cudaGetLastError();
}

template cudaError_t so3lr_ev_contraction_forward_launch<float>(const float*, const int64_t*, const int64_t*, const int64_t*, const float*,
    int64_t, int, int, float*, cudaStream_t);
template cudaError_t so3lr_ev_contraction_forward_launch<double>(const double*, const int64_t*, const int64_t*, const int64_t*,
    const double*, int64_t, int, int, double*, cudaStream_t);
template cudaError_t so3lr_ev_contraction_backward_launch<float>(const float*, const float*, const int64_t*, const int64_t*,
    const int64_t*, const float*, int64_t, int, int, float*, cudaStream_t);
template cudaError_t so3lr_ev_contraction_backward_launch<double>(const double*, const double*, const int64_t*, const int64_t*,
    const int64_t*, const double*, int64_t, int, int, double*, cudaStream_t);

// ---------------------------------------------------------------------------------------------------------------- radial filter
// so3lr::radial_filter: the radial filters of a layer as a table in the basis distance, cubic Hermite interpolation; one warp per edge.

namespace {

// Hermite cell of x: node k, offset u in [0, 1]; positions outside the table are clamped to its ends.
template <typename T>
__device__ __forceinline__ void hermite_cell(T x, T x0, T inv_h, int n_nodes, int& k, T& u) {
    T t = (x - x0) * inv_h;
    t = t < T(0) ? T(0) : (t > T(n_nodes - 1) ? T(n_nodes - 1) : t);
    k = static_cast<int>(t);
    if (k > n_nodes - 2) k = n_nodes - 2;
    u = t - T(k);
}

template <typename T>
__global__ void radial_forward_kernel(const T* __restrict__ x, const T* __restrict__ table, int64_t n_edges, int n_nodes, int C, T x0,
                                      T inv_h, T* __restrict__ out) {
    const int lane = threadIdx.x & 31;
    const int64_t e = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
    if (e >= n_edges) return;
    int k;
    T u;
    hermite_cell(x[e], x0, inv_h, n_nodes, k, u);
    const T h = T(1) / inv_h;
    const T u2 = u * u, u3 = u2 * u;
    const T h00 = T(2) * u3 - T(3) * u2 + T(1), h10 = (u3 - T(2) * u2 + u) * h, h01 = T(3) * u2 - T(2) * u3, h11 = (u3 - u2) * h;
    const T* a = table + static_cast<int64_t>(k) * 2 * C;      // node k: C values, then C derivatives
    const T* b = a + 2 * C;                                    // node k + 1
    for (int c = lane; c < C; c += 32) out[e * C + c] = h00 * a[c] + h10 * a[C + c] + h01 * b[c] + h11 * b[C + c];
}

template <typename T>
__global__ void radial_backward_kernel(const T* __restrict__ g_out, const T* __restrict__ x, const T* __restrict__ table, int64_t n_edges,
                                       int n_nodes, int C, T x0, T inv_h, T* __restrict__ g_x) {
    const int lane = threadIdx.x & 31;
    const int64_t e = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
    if (e >= n_edges) return;
    int k;
    T u;
    hermite_cell(x[e], x0, inv_h, n_nodes, k, u);
    const T u2 = u * u;
    // derivatives of the Hermite basis in x (d/du times inv_h; the derivative terms carry h, which cancels)
    const T d00 = (T(6) * u2 - T(6) * u) * inv_h, d10 = T(3) * u2 - T(4) * u + T(1), d01 = (T(6) * u - T(6) * u2) * inv_h,
            d11 = T(3) * u2 - T(2) * u;
    const T* a = table + static_cast<int64_t>(k) * 2 * C;
    const T* b = a + 2 * C;
    T part = T(0);
    for (int c = lane; c < C; c += 32) part += g_out[e * C + c] * (d00 * a[c] + d10 * a[C + c] + d01 * b[c] + d11 * b[C + c]);
    part = warp_sum(part);
    if (lane == 0) g_x[e] = part;
}

}  // namespace

template <typename T>
cudaError_t so3lr_radial_forward_launch(const T* x, const T* table, int64_t n_edges, int n_nodes, int C, double x0, double inv_h, T* out,
                                        cudaStream_t stream) {
    if (n_edges == 0) return cudaSuccess;
    const dim3 block(32 * kWarpsPerBlock);
    const dim3 grid(static_cast<unsigned>((n_edges + kWarpsPerBlock - 1) / kWarpsPerBlock));
    radial_forward_kernel<T><<<grid, block, 0, stream>>>(x, table, n_edges, n_nodes, C, static_cast<T>(x0), static_cast<T>(inv_h), out);
    return cudaGetLastError();
}

template <typename T>
cudaError_t so3lr_radial_backward_launch(const T* g_out, const T* x, const T* table, int64_t n_edges, int n_nodes, int C, double x0,
                                         double inv_h, T* g_x, cudaStream_t stream) {
    if (n_edges == 0) return cudaSuccess;
    const dim3 block(32 * kWarpsPerBlock);
    const dim3 grid(static_cast<unsigned>((n_edges + kWarpsPerBlock - 1) / kWarpsPerBlock));
    radial_backward_kernel<T><<<grid, block, 0, stream>>>(g_out, x, table, n_edges, n_nodes, C, static_cast<T>(x0), static_cast<T>(inv_h),
                                                          g_x);
    return cudaGetLastError();
}

template cudaError_t so3lr_radial_forward_launch<float>(const float*, const float*, int64_t, int, int, double, double, float*, cudaStream_t);
template cudaError_t so3lr_radial_forward_launch<double>(const double*, const double*, int64_t, int, int, double, double, double*,
    cudaStream_t);
template cudaError_t so3lr_radial_backward_launch<float>(const float*, const float*, const float*, int64_t, int, int, double, double, float*,
    cudaStream_t);
template cudaError_t so3lr_radial_backward_launch<double>(const double*, const double*, const double*, int64_t, int, int, double, double,
    double*, cudaStream_t);

// ---------------------------------------------------------------------------------------------------------- filtered attention
// so3lr::attention_filtered: the attention of so3lr::attention with the filters put together per edge inside the kernel:
// w = radial table (in the basis distance) + s (the spherical filters, computed before as matrix products), both filters at once,
// so that no radial filter array and no filter sum is stored. One warp per edge, 32 channels per head (lane = channel),
// H = D = F / 32 heads; see so3lr_fused_ops.cpp.

namespace {

constexpr int kMaxSlots = 8;          // F / 32: heads (= degree heads), F <= 256

// The filters of edge e for this lane's channels (channel t * 32 + lane of filter 1 and of filter 2); with GRAD also the derivatives
// of the radial parts in the basis distance.
template <typename T, bool GRAD>
__device__ __forceinline__ void edge_filters(int64_t e, int lane, const T* __restrict__ x, const T* __restrict__ s,
                                             const T* __restrict__ table, int n_nodes, T x0, T inv_h, int F, int FS,
                                             T (&w1)[kMaxSlots], T (&w2)[kMaxSlots], T (&dr1)[kMaxSlots], T (&dr2)[kMaxSlots]) {
    int k;
    T u;
    hermite_cell(x[e], x0, inv_h, n_nodes, k, u);
    const T h = T(1) / inv_h;
    const T u2 = u * u, u3 = u2 * u;
    const T h00 = T(2) * u3 - T(3) * u2 + T(1), h10 = (u3 - T(2) * u2 + u) * h, h01 = T(3) * u2 - T(2) * u3, h11 = (u3 - u2) * h;
    const T d00 = (T(6) * u2 - T(6) * u) * inv_h, d10 = T(3) * u2 - T(4) * u + T(1), d01 = -d00, d11 = T(3) * u2 - T(2) * u;
    const int C = 2 * F;
    const T* a = table + static_cast<int64_t>(k) * 2 * C;     // node k: C values, then C derivatives (columns: filter 1, filter 2)
    const T* b = a + 2 * C;                                   // node k + 1
    const T* se = s + e * C;
#pragma unroll
    for (int t = 0; t < kMaxSlots; ++t) {
        if (t < FS) {
            const int c1 = t * 32 + lane, c2 = F + c1;
            w1[t] = h00 * a[c1] + h10 * a[C + c1] + h01 * b[c1] + h11 * b[C + c1] + se[c1];
            w2[t] = h00 * a[c2] + h10 * a[C + c2] + h01 * b[c2] + h11 * b[C + c2] + se[c2];
            if (GRAD) {
                dr1[t] = d00 * a[c1] + d10 * a[C + c1] + d01 * b[c1] + d11 * b[C + c1];
                dr2[t] = d00 * a[c2] + d10 * a[C + c2] + d01 * b[c2] + d11 * b[C + c2];
            }
        }
    }
}

template <typename T>
__global__ void attention_filtered_forward_kernel(const T* __restrict__ q1, const T* __restrict__ k1, const T* __restrict__ v1,
                                                  const T* __restrict__ q2, const T* __restrict__ k2, const T* __restrict__ cut,
                                                  const T* __restrict__ ylm, const T* __restrict__ x, const T* __restrict__ s,
                                                  const int64_t* __restrict__ idx_i, const int64_t* __restrict__ idx_j,
                                                  const int64_t* __restrict__ degree_index, const T* __restrict__ table, int n_nodes,
                                                  int F, int64_t n_edges, int M, T x0, T inv_h, T inv_norm1, T inv_norm2,
                                                  T* __restrict__ x_att, T* __restrict__ ev_att) {
    const int lane = threadIdx.x & 31;
    const int64_t e = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
    if (e >= n_edges) return;
    const int FS = F / 32;
    T w1[kMaxSlots], w2[kMaxSlots], dr1[kMaxSlots], dr2[kMaxSlots];
    edge_filters<T, false>(e, lane, x, s, table, n_nodes, x0, inv_h, F, FS, w1, w2, dr1, dr2);
    const int64_t i = idx_i[e];
    const int64_t j = idx_j[e];
    const T c = cut[e];
    T alpha2[kMaxSlots];
#pragma unroll
    for (int t = 0; t < kMaxSlots; ++t) {
        alpha2[t] = T(0);
        if (t < FS) {
            const int64_t ni = (i * FS + t) * 32 + lane, nj = (j * FS + t) * 32 + lane;
            const T alpha1 = warp_sum(q1[ni] * w1[t] * k1[nj]) * inv_norm1 * c;
            atomicAdd(&x_att[ni], alpha1 * v1[nj]);
            alpha2[t] = warp_sum(q2[ni] * w2[t] * k2[nj]) * inv_norm2 * c;
        }
    }
    for (int m = lane; m < M; m += 32) {
        const int d = static_cast<int>(degree_index[m]);
        T a2 = T(0);
#pragma unroll
        for (int t = 0; t < kMaxSlots; ++t) {
            if (t == d) a2 = alpha2[t];
        }
        atomicAdd(&ev_att[i * M + m], a2 * ylm[e * M + m]);
    }
}

template <typename T>
__global__ void attention_filtered_backward_kernel(const T* __restrict__ g_x, const T* __restrict__ g_ev, const T* __restrict__ q1,
                                                   const T* __restrict__ k1, const T* __restrict__ v1, const T* __restrict__ q2,
                                                   const T* __restrict__ k2, const T* __restrict__ cut, const T* __restrict__ ylm,
                                                   const T* __restrict__ x, const T* __restrict__ s, const int64_t* __restrict__ idx_i,
                                                   const int64_t* __restrict__ idx_j, const int64_t* __restrict__ degree_index,
                                                   const T* __restrict__ table, int n_nodes, int F, int64_t n_edges, int M, T x0,
                                                   T inv_h, T inv_norm1, T inv_norm2, T* __restrict__ gq1, T* __restrict__ gk1,
                                                   T* __restrict__ gv1, T* __restrict__ gq2, T* __restrict__ gk2, T* __restrict__ gcut,
                                                   T* __restrict__ gylm, T* __restrict__ gx, T* __restrict__ gs) {
    const int lane = threadIdx.x & 31;
    const int64_t e = static_cast<int64_t>(blockIdx.x) * kWarpsPerBlock + (threadIdx.x >> 5);
    if (e >= n_edges) return;
    const int FS = F / 32;
    T w1[kMaxSlots], w2[kMaxSlots], dr1[kMaxSlots], dr2[kMaxSlots];
    edge_filters<T, true>(e, lane, x, s, table, n_nodes, x0, inv_h, F, FS, w1, w2, dr1, dr2);
    const int64_t i = idx_i[e];
    const int64_t j = idx_j[e];
    const T c = cut[e];
    T* gse = gs + e * 2 * F;
    T grad_cut = T(0), gd = T(0);
    // invariant heads
#pragma unroll
    for (int t = 0; t < kMaxSlots; ++t) {
        if (t < FS) {
            const int64_t ni = (i * FS + t) * 32 + lane, nj = (j * FS + t) * 32 + lane;
            const T q = q1[ni], k = k1[nj], gxi = g_x[ni], v = v1[nj];
            const T raw = warp_sum(q * w1[t] * k);
            const T dalpha = warp_sum(gxi * v);
            const T alpha = raw * inv_norm1 * c;
            const T sc = dalpha * inv_norm1 * c;
            grad_cut += dalpha * raw * inv_norm1;
            atomicAdd(&gv1[nj], alpha * gxi);
            atomicAdd(&gq1[ni], sc * w1[t] * k);
            atomicAdd(&gk1[nj], sc * q * w1[t]);
            const T gw = sc * q * k;
            gse[t * 32 + lane] = gw;
            gd += gw * dr1[t];
        }
    }
    // degree heads: dL/dalpha2 from ev_att, then as above
    T dalpha2[kMaxSlots], alpha2[kMaxSlots];
#pragma unroll
    for (int t = 0; t < kMaxSlots; ++t) dalpha2[t] = alpha2[t] = T(0);
    for (int m = lane; m < M; m += 32) {
        const int d = static_cast<int>(degree_index[m]);
        const T v = g_ev[i * M + m] * ylm[e * M + m];
#pragma unroll
        for (int t = 0; t < kMaxSlots; ++t) {
            if (t == d) dalpha2[t] += v;
        }
    }
#pragma unroll
    for (int t = 0; t < kMaxSlots; ++t) {
        if (t < FS) {
            const T da = warp_sum(dalpha2[t]);
            const int64_t ni = (i * FS + t) * 32 + lane, nj = (j * FS + t) * 32 + lane;
            const T q = q2[ni], k = k2[nj];
            const T raw = warp_sum(q * w2[t] * k);
            alpha2[t] = raw * inv_norm2 * c;
            const T sc = da * inv_norm2 * c;
            grad_cut += da * raw * inv_norm2;
            atomicAdd(&gq2[ni], sc * w2[t] * k);
            atomicAdd(&gk2[nj], sc * q * w2[t]);
            const T gw = sc * q * k;
            gse[F + t * 32 + lane] = gw;
            gd += gw * dr2[t];
        }
    }
    for (int m = lane; m < M; m += 32) {
        const int d = static_cast<int>(degree_index[m]);
        T a2 = T(0);
#pragma unroll
        for (int t = 0; t < kMaxSlots; ++t) {
            if (t == d) a2 = alpha2[t];
        }
        gylm[e * M + m] = a2 * g_ev[i * M + m];
    }
    gd = warp_sum(gd);
    if (lane == 0) {
        gcut[e] = grad_cut;
        gx[e] = gd;
    }
}

}  // namespace

template <typename T>
cudaError_t so3lr_attention_filtered_forward_launch(const T* q1, const T* k1, const T* v1, const T* q2, const T* k2, const T* cut,
                                                    const T* ylm, const T* x, const T* s, const int64_t* idx_i, const int64_t* idx_j,
                                                    const int64_t* degree_index, const T* table, int n_nodes, int F, int64_t n_edges,
                                                    int M, double x0, double inv_h, double norm1, double norm2, T* x_att, T* ev_att,
                                                    cudaStream_t stream) {
    if (n_edges == 0) return cudaSuccess;
    if (F % 32 != 0 || F / 32 > kMaxSlots) return cudaErrorInvalidValue;
    const dim3 block(32 * kWarpsPerBlock);
    const dim3 grid(static_cast<unsigned>((n_edges + kWarpsPerBlock - 1) / kWarpsPerBlock));
    attention_filtered_forward_kernel<T><<<grid, block, 0, stream>>>(
        q1, k1, v1, q2, k2, cut, ylm, x, s, idx_i, idx_j, degree_index, table, n_nodes, F, n_edges, M, static_cast<T>(x0),
        static_cast<T>(inv_h), static_cast<T>(1.0 / norm1), static_cast<T>(1.0 / norm2), x_att, ev_att);
    return cudaGetLastError();
}

template <typename T>
cudaError_t so3lr_attention_filtered_backward_launch(const T* g_x, const T* g_ev, const T* q1, const T* k1, const T* v1, const T* q2,
                                                     const T* k2, const T* cut, const T* ylm, const T* x, const T* s,
                                                     const int64_t* idx_i, const int64_t* idx_j, const int64_t* degree_index,
                                                     const T* table, int n_nodes, int F, int64_t n_edges, int M, double x0,
                                                     double inv_h, double norm1, double norm2, T* gq1, T* gk1, T* gv1, T* gq2, T* gk2,
                                                     T* gcut, T* gylm, T* gx, T* gs, cudaStream_t stream) {
    if (n_edges == 0) return cudaSuccess;
    if (F % 32 != 0 || F / 32 > kMaxSlots) return cudaErrorInvalidValue;
    const dim3 block(32 * kWarpsPerBlock);
    const dim3 grid(static_cast<unsigned>((n_edges + kWarpsPerBlock - 1) / kWarpsPerBlock));
    attention_filtered_backward_kernel<T><<<grid, block, 0, stream>>>(
        g_x, g_ev, q1, k1, v1, q2, k2, cut, ylm, x, s, idx_i, idx_j, degree_index, table, n_nodes, F, n_edges, M, static_cast<T>(x0),
        static_cast<T>(inv_h), static_cast<T>(1.0 / norm1), static_cast<T>(1.0 / norm2), gq1, gk1, gv1, gq2, gk2, gcut, gylm, gx, gs);
    return cudaGetLastError();
}

#define SO3LR_FILTERED_INSTANTIATE(T)                                                                                                      \
    template cudaError_t so3lr_attention_filtered_forward_launch<T>(const T*, const T*, const T*, const T*, const T*, const T*, const T*, \
        const T*, const T*, const int64_t*, const int64_t*, const int64_t*, const T*, int, int, int64_t, int, double, double, double,     \
        double, T*, T*, cudaStream_t);                                                                                                    \
    template cudaError_t so3lr_attention_filtered_backward_launch<T>(const T*, const T*, const T*, const T*, const T*, const T*, const T*, \
        const T*, const T*, const T*, const T*, const int64_t*, const int64_t*, const int64_t*, const T*, int, int, int64_t, int, double,  \
        double, double, double, T*, T*, T*, T*, T*, T*, T*, T*, T*, cudaStream_t);
SO3LR_FILTERED_INSTANTIATE(float)
SO3LR_FILTERED_INSTANTIATE(double)
#undef SO3LR_FILTERED_INSTANTIATE

// ------------------------------------------------------------------------------------------------------------- edge geometry
// so3lr::edge_geometry: per short-range edge, from its vector: the spherical harmonics (l = 1..4, so3lr's order, selected columns),
// the basis distance (NLH distance floor), the PhysNet cutoff and the NLH repulsion summed onto atom i. One thread per edge.

namespace {

constexpr int kGeoThreads = 256;
constexpr int kNumYlm = 24;           // l = 1..4

// so3lr sph_ops.spherical_basis of a unit vector (x, y, z) for l = 1..4 (_spherical_harmonics in adapters/so3lr.py)
template <typename T>
__device__ __forceinline__ void ylm_values(T x, T y, T z, T (&Y)[kNumYlm]) {
    const double pi = 3.14159265358979323846;
    const T c1 = T(sqrt(3.0 / (4.0 * pi))), A = T(0.5 * sqrt(15.0 / pi)), B = T(0.25 * sqrt(5.0 / pi)), C = T(0.25 * sqrt(15.0 / pi));
    const T D = T(0.25 * sqrt(35.0 / (2.0 * pi))), E = T(0.5 * sqrt(105.0 / pi)), F = T(0.25 * sqrt(21.0 / (2.0 * pi)));
    const T G = T(0.25 * sqrt(7.0 / pi)), H = T(0.25 * sqrt(105.0 / pi)), I = T(0.75 * sqrt(35.0 / pi));
    const T J = T(0.75 * sqrt(35.0 / (2.0 * pi))), K = T(0.75 * sqrt(5.0 / pi)), L = T(0.75 * sqrt(5.0 / (2.0 * pi)));
    const T P = T(3.0 / 16.0 * sqrt(1.0 / pi)), Q = T(3.0 / 8.0 * sqrt(5.0 / pi)), R = T(3.0 / 16.0 * sqrt(35.0 / pi));
    const T x2 = x * x, y2 = y * y, z2 = z * z;
    Y[0] = c1 * y; Y[1] = c1 * z; Y[2] = c1 * x;
    Y[3] = A * x * y; Y[4] = A * y * z; Y[5] = B * (T(3) * z2 - T(1)); Y[6] = A * x * z; Y[7] = C * (x2 - y2);
    Y[8] = D * y * (T(3) * x2 - y2); Y[9] = E * x * y * z; Y[10] = F * y * (T(5) * z2 - T(1)); Y[11] = G * (T(5) * z2 * z - T(3) * z);
    Y[12] = F * x * (T(5) * z2 - T(1)); Y[13] = H * (x2 - y2) * z; Y[14] = D * x * (x2 - T(3) * y2);
    Y[15] = I * x * y * (x2 - y2); Y[16] = J * y * (T(3) * x2 - y2) * z; Y[17] = K * x * y * (T(7) * z2 - T(1));
    Y[18] = L * y * (T(7) * z2 * z - T(3) * z); Y[19] = P * (T(35) * z2 * z2 - T(30) * z2 + T(3)); Y[20] = L * x * (T(7) * z2 * z - T(3) * z);
    Y[21] = Q * (x2 - y2) * (T(7) * z2 - T(1)); Y[22] = J * x * (x2 - T(3) * y2) * z;
    Y[23] = R * (x2 * (x2 - T(3) * y2) - y2 * (T(3) * x2 - y2));
}

// sum over the 24 functions of g[n] * grad Y_n at (x, y, z), the functions taken as polynomials in x, y, z
template <typename T>
__device__ __forceinline__ void ylm_gradient(T x, T y, T z, const T (&g)[kNumYlm], T& gx, T& gy, T& gz) {
    const double pi = 3.14159265358979323846;
    const T c1 = T(sqrt(3.0 / (4.0 * pi))), A = T(0.5 * sqrt(15.0 / pi)), B = T(0.25 * sqrt(5.0 / pi)), C = T(0.25 * sqrt(15.0 / pi));
    const T D = T(0.25 * sqrt(35.0 / (2.0 * pi))), E = T(0.5 * sqrt(105.0 / pi)), F = T(0.25 * sqrt(21.0 / (2.0 * pi)));
    const T G = T(0.25 * sqrt(7.0 / pi)), H = T(0.25 * sqrt(105.0 / pi)), I = T(0.75 * sqrt(35.0 / pi));
    const T J = T(0.75 * sqrt(35.0 / (2.0 * pi))), K = T(0.75 * sqrt(5.0 / pi)), L = T(0.75 * sqrt(5.0 / (2.0 * pi)));
    const T P = T(3.0 / 16.0 * sqrt(1.0 / pi)), Q = T(3.0 / 8.0 * sqrt(5.0 / pi)), R = T(3.0 / 16.0 * sqrt(35.0 / pi));
    const T x2 = x * x, y2 = y * y, z2 = z * z;
    gx = g[2] * c1 + g[3] * A * y + g[6] * A * z + g[7] * T(2) * C * x + g[8] * T(6) * D * x * y + g[9] * E * y * z
       + g[12] * F * (T(5) * z2 - T(1)) + g[13] * T(2) * H * x * z + g[14] * D * (T(3) * x2 - T(3) * y2)
       + g[15] * I * (T(3) * x2 * y - y2 * y) + g[16] * T(6) * J * x * y * z + g[17] * K * y * (T(7) * z2 - T(1))
       + g[20] * L * (T(7) * z2 * z - T(3) * z) + g[21] * T(2) * Q * x * (T(7) * z2 - T(1)) + g[22] * J * (T(3) * x2 - T(3) * y2) * z
       + g[23] * R * (T(4) * x2 * x - T(12) * x * y2);
    gy = g[0] * c1 + g[3] * A * x + g[4] * A * z - g[7] * T(2) * C * y + g[8] * D * (T(3) * x2 - T(3) * y2) + g[9] * E * x * z
       + g[10] * F * (T(5) * z2 - T(1)) - g[13] * T(2) * H * y * z - g[14] * T(6) * D * x * y
       + g[15] * I * (x2 * x - T(3) * x * y2) + g[16] * J * (T(3) * x2 - T(3) * y2) * z + g[17] * K * x * (T(7) * z2 - T(1))
       + g[18] * L * (T(7) * z2 * z - T(3) * z) - g[21] * T(2) * Q * y * (T(7) * z2 - T(1)) - g[22] * T(6) * J * x * y * z
       + g[23] * R * (T(4) * y2 * y - T(12) * x2 * y);
    gz = g[1] * c1 + g[4] * A * y + g[5] * T(6) * B * z + g[6] * A * x + g[9] * E * x * y + g[10] * T(10) * F * y * z
       + g[11] * G * (T(15) * z2 - T(3)) + g[12] * T(10) * F * x * z + g[13] * H * (x2 - y2) + g[16] * J * y * (T(3) * x2 - y2)
       + g[17] * T(14) * K * x * y * z + g[18] * L * y * (T(21) * z2 - T(3)) + g[19] * P * (T(140) * z2 * z - T(60) * z)
       + g[20] * L * x * (T(21) * z2 - T(3)) + g[21] * T(14) * Q * z * (x2 - y2) + g[22] * J * x * (x2 - T(3) * y2);
}

// the radial quantities of one edge: basis distance, cutoff, repulsion pair energy, and with GRAD their derivatives in d
template <typename T, bool GRAD>
__device__ __forceinline__ void edge_radial(T d, int64_t zi, int64_t zj, const T* __restrict__ floor_d, const T* __restrict__ nlh_a,
                                            const T* __restrict__ nlh_b, int Zt, int Kn, const So3lrGeometryParams& p, T& db, T& cut,
                                            T& erep, T& ddb, T& dcut, T& derep) {
    const int64_t lo = (zi < zj ? zi : zj) - 1, hi = (zi < zj ? zj : zi) - 1;
    const T w = T(p.floor_width);
    const T d_off = floor_d[lo * Zt + hi] + w;
    if (d >= d_off) {
        db = d;
        ddb = T(1);
    } else {
        const T th = tanh((d_off - d) / w);
        db = d_off - w * th;
        ddb = T(1) - th * th;
    }
    const T rc = T(p.sr_cutoff);
    const T t = d / rc, t2 = t * t, t3 = t2 * t;
    cut = T(1) - T(6) * t3 * t2 + T(15) * t2 * t2 - T(10) * t3;
    dcut = (-T(30) * t2 * t2 + T(60) * t3 - T(30) * t2) / rc;
    erep = derep = T(0);
    if (p.use_repulsion) {
        const T* a = nlh_a + (lo * Zt + hi) * Kn;
        const T* b = nlh_b + (lo * Zt + hi) * Kn;
        T y = T(0), dy = T(0);
        for (int k = 0; k < Kn; ++k) {
            const T ex = a[k] * exp(-b[k] * d);
            y += ex;
            dy -= b[k] * ex;
        }
        T sw, dsw;
        switch_fn<T>(d, T(0), T(p.repulsion_switch), sw, dsw);
        const T pre = T(0.5) * T(p.ke) * T(zi * zj);
        erep = pre * sw * cut * y / d;
        if (GRAD) derep = pre * (dsw * cut * y / d + sw * dcut * y / d + sw * cut * dy / d - sw * cut * y / (d * d));
    }
}

template <typename T>
__global__ void geometry_forward_kernel(const T* __restrict__ vec, const int64_t* __restrict__ idx_i, const int64_t* __restrict__ idx_j,
                                        const int64_t* __restrict__ z, const T* __restrict__ floor_d, const T* __restrict__ nlh_a,
                                        const T* __restrict__ nlh_b, int Zt, int Kn, const int64_t* __restrict__ columns, int M,
                                        int64_t n_edges, So3lrGeometryParams p, T* __restrict__ ylm, T* __restrict__ d_basis,
                                        T* __restrict__ cut_out, T* __restrict__ e_rep) {
    const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool valid = e < n_edges;
    const int64_t i = valid ? idx_i[e] : -1;
    T erep = T(0);
    if (valid) {
        const T vx = vec[3 * e], vy = vec[3 * e + 1], vz = vec[3 * e + 2];
        const T d = sqrt(vx * vx + vy * vy + vz * vz);
        T Y[kNumYlm];
        ylm_values(vx / d, vy / d, vz / d, Y);
        for (int m = 0; m < M; ++m) {
            const int col = static_cast<int>(columns[m]);
            T v = T(0);
#pragma unroll
            for (int n = 0; n < kNumYlm; ++n) {
                if (n == col) v = Y[n];
            }
            ylm[e * M + m] = v;
        }
        T db, cut, ddb, dcut, derep;
        edge_radial<T, false>(d, z[i], z[idx_j[e]], floor_d, nlh_a, nlh_b, Zt, Kn, p, db, cut, erep, ddb, dcut, derep);
        d_basis[e] = db;
        cut_out[e] = cut;
    }
    const WarpRun run = warp_run(i);
    erep = run_sum(erep, run);
    if (run.head && i >= 0 && erep != T(0)) atomicAdd(&e_rep[i], erep);
}

template <typename T>
__global__ void geometry_backward_kernel(const T* __restrict__ g_ylm, const T* __restrict__ g_db, const T* __restrict__ g_cut,
                                         const T* __restrict__ g_erep, const T* __restrict__ vec, const int64_t* __restrict__ idx_i,
                                         const int64_t* __restrict__ idx_j, const int64_t* __restrict__ z, const T* __restrict__ floor_d,
                                         const T* __restrict__ nlh_a, const T* __restrict__ nlh_b, int Zt, int Kn,
                                         const int64_t* __restrict__ columns, int M, int64_t n_edges, So3lrGeometryParams p,
                                         T* __restrict__ g_vec) {
    const int64_t e = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= n_edges) return;
    const int64_t i = idx_i[e];
    const T vx = vec[3 * e], vy = vec[3 * e + 1], vz = vec[3 * e + 2];
    const T d = sqrt(vx * vx + vy * vy + vz * vz);
    const T ux = vx / d, uy = vy / d, uz = vz / d;
    T g[kNumYlm];
#pragma unroll
    for (int n = 0; n < kNumYlm; ++n) g[n] = T(0);
    for (int m = 0; m < M; ++m) {
        const int col = static_cast<int>(columns[m]);
        const T gm = g_ylm[e * M + m];
#pragma unroll
        for (int n = 0; n < kNumYlm; ++n) {
            if (n == col) g[n] += gm;
        }
    }
    T gux, guy, guz;
    ylm_gradient(ux, uy, uz, g, gux, guy, guz);
    T db, cut, erep, ddb, dcut, derep;
    edge_radial<T, true>(d, z[i], z[idx_j[e]], floor_d, nlh_a, nlh_b, Zt, Kn, p, db, cut, erep, ddb, dcut, derep);
    const T gd = g_db[e] * ddb + g_cut[e] * dcut + g_erep[i] * derep;
    const T dot = gux * ux + guy * uy + guz * uz;
    g_vec[3 * e] = (gux - dot * ux) / d + gd * ux;
    g_vec[3 * e + 1] = (guy - dot * uy) / d + gd * uy;
    g_vec[3 * e + 2] = (guz - dot * uz) / d + gd * uz;
}

}  // namespace

template <typename T>
cudaError_t so3lr_geometry_forward_launch(const T* vec, const int64_t* idx_i, const int64_t* idx_j, const int64_t* z, const T* floor_d,
                                          const T* nlh_a, const T* nlh_b, int Zt, int Kn, const int64_t* columns, int M, int64_t n_edges,
                                          const So3lrGeometryParams& p, T* ylm, T* d_basis, T* cut, T* e_rep, cudaStream_t stream) {
    if (n_edges == 0) return cudaSuccess;
    const unsigned blocks = static_cast<unsigned>((n_edges + kGeoThreads - 1) / kGeoThreads);
    geometry_forward_kernel<T><<<blocks, kGeoThreads, 0, stream>>>(vec, idx_i, idx_j, z, floor_d, nlh_a, nlh_b, Zt, Kn, columns, M, n_edges,
                                                                   p, ylm, d_basis, cut, e_rep);
    return cudaGetLastError();
}

template <typename T>
cudaError_t so3lr_geometry_backward_launch(const T* g_ylm, const T* g_db, const T* g_cut, const T* g_erep, const T* vec,
                                           const int64_t* idx_i, const int64_t* idx_j, const int64_t* z, const T* floor_d, const T* nlh_a,
                                           const T* nlh_b, int Zt, int Kn, const int64_t* columns, int M, int64_t n_edges,
                                           const So3lrGeometryParams& p, T* g_vec, cudaStream_t stream) {
    if (n_edges == 0) return cudaSuccess;
    const unsigned blocks = static_cast<unsigned>((n_edges + kGeoThreads - 1) / kGeoThreads);
    geometry_backward_kernel<T><<<blocks, kGeoThreads, 0, stream>>>(g_ylm, g_db, g_cut, g_erep, vec, idx_i, idx_j, z, floor_d, nlh_a,
                                                                    nlh_b, Zt, Kn, columns, M, n_edges, p, g_vec);
    return cudaGetLastError();
}

#define SO3LR_GEOMETRY_INSTANTIATE(T)                                                                                                     \
    template cudaError_t so3lr_geometry_forward_launch<T>(const T*, const int64_t*, const int64_t*, const int64_t*, const T*, const T*,  \
        const T*, int, int, const int64_t*, int, int64_t, const So3lrGeometryParams&, T*, T*, T*, T*, cudaStream_t);                     \
    template cudaError_t so3lr_geometry_backward_launch<T>(const T*, const T*, const T*, const T*, const T*, const int64_t*,             \
        const int64_t*, const int64_t*, const T*, const T*, const T*, int, int, const int64_t*, int, int64_t, const So3lrGeometryParams&, \
        T*, cudaStream_t);
SO3LR_GEOMETRY_INSTANTIATE(float)
SO3LR_GEOMETRY_INSTANTIATE(double)
#undef SO3LR_GEOMETRY_INSTANTIATE
