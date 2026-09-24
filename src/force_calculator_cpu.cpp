
#include "../include/force_calculator_cpu.h"
#include "../include/md_system.h"
#include "../include/neighbor_list.h"
#include "../include/topology.h"
#include "../include/mpi_utils.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cctype>
#include <mpi.h>
#include <iomanip>
#include <cstring>
#include <stdexcept>
#include <string>
#include <fftw3.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

#if defined(__AVX512F__)
constexpr int kCpuSimdLen = 8;  // 8 doubles per AVX-512 vector
#elif defined(__AVX__) || defined(__AVX2__)
constexpr int kCpuSimdLen = 4;  // 4 doubles per AVX/AVX2 vector
#else
constexpr int kCpuSimdLen = 1;  // scalar fallback
#endif

bool env_flag_enabled_local(const char* name) {
    const char* v = std::getenv(name);
    if (!v || !*v) return false;
    std::string s(v);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s == "1" || s == "true" || s == "yes" || s == "on";
}

// Real-space Ewald lookup tables (see force_calculator_cpu.h's
// ewald_rs_*_table_ member comment for the full rationale). Builds a table
// of the r^2-dependent energy/force factors -- erfc(alpha*r)/r and its force
// derivative for normal pairs, or the erf(alpha*r)/r analog for excluded
// pairs -- over [r2_min, cutoff^2), using the exact std::erf/erfc (this is a
// one-time build, so exactness costs nothing). +3 extra nodes past cutoff^2
// so the top bin's linear interpolation (which reads index+1) never touches
// uninitialized memory for r2 arbitrarily close to the cutoff.
// PERF FIX (layout): energy and force are written INTERLEAVED into a single
// table -- node i occupies table[2*i] (energy) and table[2*i+1] (force) --
// see the ewald_rs_table_ member comment in force_calculator_cpu.h. A lookup
// needs both factors at both node i and node i+1, which is now four
// consecutive doubles: one cache line instead of two lines in two arrays.
int build_real_space_ewald_table(
    Real cutoff, Real alpha, Real r2_min, Real spacing, bool use_erf,
    std::vector<Real>& table) {
    const Real two_alpha_over_sqrt_pi = 1.1283791670955126 * alpha;
    const Real cutoff_sq = cutoff * cutoff;
    const int n = static_cast<int>(std::ceil((cutoff_sq - r2_min) / spacing)) + 3;
    table.assign(static_cast<size_t>(2 * n), 0.0);
    for (int i = 0; i < n; ++i) {
        Real r2 = r2_min + static_cast<Real>(i) * spacing;
        Real r = std::sqrt(r2);
        Real z = alpha * r;
        Real exp_term = std::exp(-z * z);
        if (use_erf) {
            Real erf_z = std::erf(z);
            table[static_cast<size_t>(2 * i)] = -erf_z / r;
            table[static_cast<size_t>(2 * i + 1)] =
                -(erf_z / (r2 * r) - two_alpha_over_sqrt_pi * exp_term / r2);
        } else {
            Real erfc_z = std::erfc(z);
            table[static_cast<size_t>(2 * i)] = erfc_z / r;
            table[static_cast<size_t>(2 * i + 1)] =
                erfc_z / (r2 * r) + two_alpha_over_sqrt_pi * exp_term / r2;
        }
    }
    return n;
}

// Linear interpolation lookup into a table built by build_real_space_ewald_
// table above. r2 is assumed to already satisfy r2_min <= r2 < cutoff^2 (the
// caller's cutoff check guarantees the upper bound; the lower bound is
// enforced by clamping idx_f below, which only matters for the pathological
// near-zero-distance case the 1e-8 floor used to guard against).
inline void lookup_real_space_ewald(
    const Real* table,
    Real r2_min, Real inv_spacing, Real r2,
    Real& energy_factor, Real& force_factor) {
    Real idx_f = (r2 - r2_min) * inv_spacing;
    if (idx_f < 0.0) idx_f = 0.0;
    int idx = static_cast<int>(idx_f);
    Real frac = idx_f - static_cast<Real>(idx);
    // e0,f0,e1,f1 -- four consecutive doubles (see build_real_space_ewald_table).
    const Real* node = table + 2 * static_cast<size_t>(idx);
    energy_factor = node[0] + frac * (node[2] - node[0]);
    force_factor  = node[1] + frac * (node[3] - node[1]);
}

// Scalar LJ(12-6) pair force/energy, shared by every scalar CPU nonbonded
inline void compute_lj_pair(Real r2, Real c6, Real c12, Real cutoff_sq,
                            bool switching_enabled, Real switch_sq, Real inv_denom,
                            Real& force_mag, Real& pe_pair) {
    const Real r2inv = 1.0 / r2;
    const Real r6inv = r2inv * r2inv * r2inv;
    const Real attractive = c6 * r6inv;
    const Real repulsive = c12 * r6inv * r6inv;
    force_mag = r2inv * (12.0 * repulsive - 6.0 * attractive);
    pe_pair = repulsive - attractive;
    if (switching_enabled && r2 > switch_sq) {
        const Real a = cutoff_sq - r2;
        const Real b = cutoff_sq + 2.0 * r2 - 3.0 * switch_sq;
        const Real switch_value = a * a * b * inv_denom;
        const Real force_correction = 12.0 * pe_pair * a * (r2 - switch_sq) * inv_denom;
        force_mag = switch_value * force_mag + force_correction;
        pe_pair *= switch_value;
    }
}

// Scalar erfc(alpha*r)-screened Ewald real-space Coulomb pair force/energy,
inline void compute_coulomb_pair_real_space(const Real* table, Real r2_min, Real inv_spacing,
                                            Real r2, Real qi, Real qj, Real k_coulomb,
                                            Real& force_mag_c, Real& pe_c) {
    Real energy_factor, force_factor;
    lookup_real_space_ewald(table, r2_min, inv_spacing, r2, energy_factor, force_factor);
    const Real qq = k_coulomb * qi * qj;
    force_mag_c = qq * force_factor;
    pe_c = qq * energy_factor;
}

#if defined(__AVX512F__)
inline __mmask8 compute_lj_avx512(
        __m512d r2, __mmask8 cutoff_mask, unsigned excl_bits,
        __m256i type_i_vec, __m256i vlj_num_types, int j_type,
        const Real* lj_c6, const Real* lj_c12,
        __m512d vone, __m512d v12, __m512d v6,
        Real cutoff_sq, bool lj_switching_enabled, Real switch_sq, Real inv_denom,
        __m512i& lj_cnt_vec, __m512i lj_cnt_one,
        __m512d& force_mag, __m512d& pe_pair) {
    const __mmask8 lj_mask = cutoff_mask & static_cast<__mmask8>(~excl_bits);
    if (!lj_mask) return lj_mask;

    const __m256i idx = _mm256_add_epi32(
        _mm256_mullo_epi32(type_i_vec, vlj_num_types), _mm256_set1_epi32(j_type));
    const __m512d c6 = _mm512_mask_i32gather_pd(_mm512_setzero_pd(), lj_mask, idx, lj_c6, 8);
    const __m512d c12 = _mm512_mask_i32gather_pd(_mm512_setzero_pd(), lj_mask, idx, lj_c12, 8);
    const __m512d r2inv = _mm512_mask_div_pd(_mm512_setzero_pd(), lj_mask, vone, r2);
    const __m512d r4inv = _mm512_mul_pd(r2inv, r2inv);
    const __m512d r6inv = _mm512_mul_pd(r4inv, r2inv);
    const __m512d attractive = _mm512_mul_pd(c6, r6inv);
    const __m512d repulsive = _mm512_mul_pd(c12, _mm512_mul_pd(r6inv, r6inv));

    force_mag = _mm512_mul_pd(r2inv,
        _mm512_sub_pd(_mm512_mul_pd(v12, repulsive), _mm512_mul_pd(v6, attractive)));
    pe_pair = _mm512_sub_pd(repulsive, attractive);

    if (lj_switching_enabled) {
        const __mmask8 sw = lj_mask & _mm512_cmp_pd_mask(r2, _mm512_set1_pd(switch_sq), _CMP_GT_OQ);
        if (sw) {
            const __m512d a = _mm512_sub_pd(_mm512_set1_pd(cutoff_sq), r2);
            const __m512d b = _mm512_sub_pd(
                _mm512_add_pd(_mm512_set1_pd(cutoff_sq), _mm512_mul_pd(_mm512_set1_pd(2.0), r2)),
                _mm512_set1_pd(3.0 * switch_sq));
            const __m512d switch_value = _mm512_mul_pd(
                _mm512_mul_pd(a, a), _mm512_mul_pd(b, _mm512_set1_pd(inv_denom)));
            const __m512d force_correction = _mm512_mul_pd(
                _mm512_set1_pd(12.0), _mm512_mul_pd(pe_pair,
                _mm512_mul_pd(a, _mm512_mul_pd(
                    _mm512_sub_pd(r2, _mm512_set1_pd(switch_sq)),
                    _mm512_set1_pd(inv_denom)))));
            force_mag = _mm512_mask_blend_pd(sw, force_mag,
                _mm512_add_pd(_mm512_mul_pd(switch_value, force_mag), force_correction));
            pe_pair = _mm512_mask_blend_pd(sw, pe_pair, _mm512_mul_pd(pe_pair, switch_value));
        }
    }

    lj_cnt_vec = _mm512_mask_add_epi64(lj_cnt_vec, lj_mask, lj_cnt_vec, lj_cnt_one);
    return lj_mask;
}

// AVX-512 erfc(alpha*r)-screened Ewald real-space Coulomb pair force/energy
// for 8 lanes at once -- the vectorized counterpart of
// compute_coulomb_pair_real_space() above, used only by
// compute_tiled_nonbonded()'s AVX-512 path. Table lookup vectorized
// with two disjoint masked gathers (non-excluded/excluded lanes, selected by
// excl_bits) merged into one register, mirroring the scalar linear
// interpolation in lookup_real_space_ewald().
inline void compute_pme_avx512(
        __m512d r2, __mmask8 cutoff_mask, unsigned excl_bits,
        const Real* rs_table, const Real* rs_table_excl,
        __m512d v_rs_r2_min, __m512d v_rs_inv_spacing,
        __m512d vk_coulomb, __m512d qi_vec, Real qj,
        __m512d& force_mag_c, __m512d& pe_c) {
    const __m512d idx_f = _mm512_max_pd(
        _mm512_mul_pd(_mm512_sub_pd(r2, v_rs_r2_min), v_rs_inv_spacing), _mm512_setzero_pd());
    const __m256i idx_i = _mm512_cvttpd_epi32(idx_f);
    const __m512d frac = _mm512_sub_pd(idx_f, _mm512_cvtepi32_pd(idx_i));
    const __m256i idx2 = _mm256_slli_epi32(idx_i, 1);
    const __m256i idx2p1 = _mm256_add_epi32(idx2, _mm256_set1_epi32(1));
    const __m256i idx2p2 = _mm256_add_epi32(idx2, _mm256_set1_epi32(2));
    const __m256i idx2p3 = _mm256_add_epi32(idx2, _mm256_set1_epi32(3));
    const __mmask8 non_excl_mask = cutoff_mask & static_cast<__mmask8>(~excl_bits);
    const __mmask8 excl_mask = cutoff_mask & static_cast<__mmask8>(excl_bits);

    __m512d e0 = _mm512_mask_i32gather_pd(_mm512_setzero_pd(), non_excl_mask, idx2, rs_table, 8);
    e0 = _mm512_mask_i32gather_pd(e0, excl_mask, idx2, rs_table_excl, 8);
    __m512d f0 = _mm512_mask_i32gather_pd(_mm512_setzero_pd(), non_excl_mask, idx2p1, rs_table, 8);
    f0 = _mm512_mask_i32gather_pd(f0, excl_mask, idx2p1, rs_table_excl, 8);
    __m512d e1 = _mm512_mask_i32gather_pd(_mm512_setzero_pd(), non_excl_mask, idx2p2, rs_table, 8);
    e1 = _mm512_mask_i32gather_pd(e1, excl_mask, idx2p2, rs_table_excl, 8);
    __m512d f1 = _mm512_mask_i32gather_pd(_mm512_setzero_pd(), non_excl_mask, idx2p3, rs_table, 8);
    f1 = _mm512_mask_i32gather_pd(f1, excl_mask, idx2p3, rs_table_excl, 8);

    const __m512d energy_factor = _mm512_add_pd(e0, _mm512_mul_pd(frac, _mm512_sub_pd(e1, e0)));
    const __m512d force_factor  = _mm512_add_pd(f0, _mm512_mul_pd(frac, _mm512_sub_pd(f1, f0)));
    const __m512d qq = _mm512_mul_pd(vk_coulomb, _mm512_mul_pd(qi_vec, _mm512_set1_pd(qj)));
    force_mag_c = _mm512_mul_pd(qq, force_factor);
    pe_c = _mm512_mul_pd(qq, energy_factor);
}
#elif defined(__AVX2__)
// Expands bit i (i=0..3) of `bits4` into lane i of a 4-wide "all-ones/
// all-zeros" double mask -- AVX2 has no mask-register type, so every
// mmask8-style predicate from the AVX-512 kernel above is represented here
// as a __m256d whose lanes are either 0 (false) or all-bits-set (true), and
// combined with _mm256_and_pd/_mm256_andnot_pd instead of bitwise mmask ops.
inline __m256d mask4_from_bits(unsigned bits4) {
    const __m128i shift_amt = _mm_set_epi32(3, 2, 1, 0);
    const __m128i bcast = _mm_set1_epi32(static_cast<int>(bits4));
    const __m128i bit = _mm_and_si128(_mm_srlv_epi32(bcast, shift_amt), _mm_set1_epi32(1));
    const __m128i mask32 = _mm_sub_epi32(_mm_setzero_si128(), bit);  // 0 or -1 per lane
    return _mm256_castsi256_pd(_mm256_cvtepi32_epi64(mask32));
}

// AVX2 (4-wide double) counterpart of compute_lj_avx512() above -- same
// algorithm, see its comment. Masked gathers use the AVX2 floating-point-mask
// gather variant (_mm256_mask_i32gather_pd): a lane is gathered where its
// mask double has the sign bit set (all-ones from mask4_from_bits/_mm256_cmp_pd),
// otherwise the lane keeps its `src` value -- the same zero-masking
// convention as the AVX-512 version's mmask8 gathers.
inline __m256d compute_lj_avx2(
        __m256d r2, __m256d cutoff_mask, __m256d excl_mask,
        __m128i type_i_vec, __m128i vlj_num_types, int j_type,
        const Real* lj_c6, const Real* lj_c12,
        __m256d vone, __m256d v12, __m256d v6,
        Real cutoff_sq, bool lj_switching_enabled, Real switch_sq, Real inv_denom,
        __m256i& lj_cnt_vec, __m256i lj_cnt_one,
        __m256d& force_mag, __m256d& pe_pair) {
    const __m256d lj_mask = _mm256_andnot_pd(excl_mask, cutoff_mask);
    if (_mm256_movemask_pd(lj_mask) == 0) return lj_mask;

    const __m128i idx = _mm_add_epi32(
        _mm_mullo_epi32(type_i_vec, vlj_num_types), _mm_set1_epi32(j_type));
    const __m256d c6 = _mm256_mask_i32gather_pd(_mm256_setzero_pd(), lj_c6, idx, lj_mask, 8);
    const __m256d c12 = _mm256_mask_i32gather_pd(_mm256_setzero_pd(), lj_c12, idx, lj_mask, 8);
    const __m256d r2inv = _mm256_div_pd(vone, r2);
    const __m256d r4inv = _mm256_mul_pd(r2inv, r2inv);
    const __m256d r6inv = _mm256_mul_pd(r4inv, r2inv);
    const __m256d attractive = _mm256_mul_pd(c6, r6inv);
    const __m256d repulsive = _mm256_mul_pd(c12, _mm256_mul_pd(r6inv, r6inv));

    force_mag = _mm256_mul_pd(r2inv,
        _mm256_sub_pd(_mm256_mul_pd(v12, repulsive), _mm256_mul_pd(v6, attractive)));
    pe_pair = _mm256_sub_pd(repulsive, attractive);

    if (lj_switching_enabled) {
        const __m256d sw = _mm256_and_pd(lj_mask,
            _mm256_cmp_pd(r2, _mm256_set1_pd(switch_sq), _CMP_GT_OQ));
        if (_mm256_movemask_pd(sw) != 0) {
            const __m256d a = _mm256_sub_pd(_mm256_set1_pd(cutoff_sq), r2);
            const __m256d b = _mm256_sub_pd(
                _mm256_add_pd(_mm256_set1_pd(cutoff_sq), _mm256_mul_pd(_mm256_set1_pd(2.0), r2)),
                _mm256_set1_pd(3.0 * switch_sq));
            const __m256d switch_value = _mm256_mul_pd(
                _mm256_mul_pd(a, a), _mm256_mul_pd(b, _mm256_set1_pd(inv_denom)));
            const __m256d force_correction = _mm256_mul_pd(
                _mm256_set1_pd(12.0), _mm256_mul_pd(pe_pair,
                _mm256_mul_pd(a, _mm256_mul_pd(
                    _mm256_sub_pd(r2, _mm256_set1_pd(switch_sq)),
                    _mm256_set1_pd(inv_denom)))));
            force_mag = _mm256_blendv_pd(force_mag,
                _mm256_add_pd(_mm256_mul_pd(switch_value, force_mag), force_correction), sw);
            pe_pair = _mm256_blendv_pd(pe_pair, _mm256_mul_pd(pe_pair, switch_value), sw);
        }
    }

    lj_cnt_vec = _mm256_add_epi64(lj_cnt_vec,
        _mm256_and_si256(_mm256_castpd_si256(lj_mask), lj_cnt_one));
    return lj_mask;
}

// AVX2 (4-wide double) counterpart of compute_pme_avx512() above -- see its
// comment for the shared table-lookup algorithm; masking/merging follows the
// same all-ones/all-zeros __m256d convention as compute_lj_avx2() above.
inline void compute_pme_avx2(
        __m256d r2, __m256d cutoff_mask, __m256d excl_mask,
        const Real* rs_table, const Real* rs_table_excl,
        __m256d v_rs_r2_min, __m256d v_rs_inv_spacing,
        __m256d vk_coulomb, __m256d qi_vec, Real qj,
        __m256d& force_mag_c, __m256d& pe_c) {
    const __m256d idx_f = _mm256_max_pd(
        _mm256_mul_pd(_mm256_sub_pd(r2, v_rs_r2_min), v_rs_inv_spacing), _mm256_setzero_pd());
    const __m128i idx_i = _mm256_cvttpd_epi32(idx_f);
    const __m256d frac = _mm256_sub_pd(idx_f, _mm256_cvtepi32_pd(idx_i));
    const __m128i idx2 = _mm_slli_epi32(idx_i, 1);
    const __m128i idx2p1 = _mm_add_epi32(idx2, _mm_set1_epi32(1));
    const __m128i idx2p2 = _mm_add_epi32(idx2, _mm_set1_epi32(2));
    const __m128i idx2p3 = _mm_add_epi32(idx2, _mm_set1_epi32(3));
    const __m256d non_excl_mask = _mm256_andnot_pd(excl_mask, cutoff_mask);
    const __m256d excl_only_mask = _mm256_and_pd(excl_mask, cutoff_mask);

    __m256d e0 = _mm256_mask_i32gather_pd(_mm256_setzero_pd(), rs_table, idx2, non_excl_mask, 8);
    e0 = _mm256_mask_i32gather_pd(e0, rs_table_excl, idx2, excl_only_mask, 8);
    __m256d f0 = _mm256_mask_i32gather_pd(_mm256_setzero_pd(), rs_table, idx2p1, non_excl_mask, 8);
    f0 = _mm256_mask_i32gather_pd(f0, rs_table_excl, idx2p1, excl_only_mask, 8);
    __m256d e1 = _mm256_mask_i32gather_pd(_mm256_setzero_pd(), rs_table, idx2p2, non_excl_mask, 8);
    e1 = _mm256_mask_i32gather_pd(e1, rs_table_excl, idx2p2, excl_only_mask, 8);
    __m256d f1 = _mm256_mask_i32gather_pd(_mm256_setzero_pd(), rs_table, idx2p3, non_excl_mask, 8);
    f1 = _mm256_mask_i32gather_pd(f1, rs_table_excl, idx2p3, excl_only_mask, 8);

    const __m256d energy_factor = _mm256_add_pd(e0, _mm256_mul_pd(frac, _mm256_sub_pd(e1, e0)));
    const __m256d force_factor  = _mm256_add_pd(f0, _mm256_mul_pd(frac, _mm256_sub_pd(f1, f0)));
    const __m256d qq = _mm256_mul_pd(vk_coulomb, _mm256_mul_pd(qi_vec, _mm256_set1_pd(qj)));
    force_mag_c = _mm256_mul_pd(qq, force_factor);
    pe_c = _mm256_mul_pd(qq, energy_factor);
}
#endif  // __AVX512F__ / __AVX2__

// Post-processing shared by both LJ and Coulomb terms inside
// compute_tiled_nonbonded()'s scalar fallback loop: turn a pair's
// scalar force magnitude into cartesian force components and, if this lane
// owns the pair's energy/virial (see the half-list ownership comment at its
// call site), fold it into the running per-tile accumulators. Called once
// for the LJ term and once for the Coulomb term of the same pair.
inline void accumulate_pair_contribution(Real force_mag, Real pe_pair, Real dx, Real dy, Real dz, Real r2,
                                         bool owner, Real& fxp, Real& fyp, Real& fzp,
                                         Real& pe_tile, Real& vir_tile) {
    fxp += force_mag * dx;
    fyp += force_mag * dy;
    fzp += force_mag * dz;
    if (owner) {
        pe_tile += pe_pair;
        vir_tile += r2 * force_mag;
    }
}

// Commits a tile-pair edge's j-reaction (Newton's 3rd law) directly into this
// thread's own private force slice. Safe unconditionally: every caller
// passes ITS OWN per-thread fx_out/fy_out/fz_out slice (see the
// lj_thread_fx_ comment in force_calculator_cpu.h), never the shared
// particles.fx/fy/fz array, so this can never race another thread no matter
// which tiles it touches. Identical in compute_tiled_nonbonded()'s AVX-512,
// AVX2 and scalar-fallback branches, so it lives here once instead of thrice.
inline void commit_reaction_force(int j_atom, Real jfx, Real jfy, Real jfz,
                                  Real* fx_out, Real* fy_out, Real* fz_out) {
    fx_out[j_atom] += jfx;
    fy_out[j_atom] += jfy;
    fz_out[j_atom] += jfz;
}


inline void commit_directed_pair_contribution(
        Real force_mag, Real pair_energy, Real dx, Real dy, Real dz, Real r2,
        bool j_is_local, bool i_owns_cross_rank_pair,
        Real& fix, Real& fiy, Real& fiz, Real& local_pe, Real& local_virial) {
    fix += force_mag * dx;
    fiy += force_mag * dy;
    fiz += force_mag * dz;
    if (j_is_local) {
        local_pe += 0.5 * pair_energy;
        local_virial += 0.5 * r2 * force_mag;
    } else if (i_owns_cross_rank_pair) {
        local_pe += pair_energy;
        local_virial += r2 * force_mag;
    }
}

struct ExclusionRow {
    const int* partners = nullptr;
    int count = 0;
    int lo = 0, hi = -1;  // min/max partner gid; hi < lo when count == 0
};

inline ExclusionRow exclusion_row_for(const int* offsets, const int* partners, int gid_i) {
    ExclusionRow row;
    const int begin = offsets[gid_i];
    const int end = offsets[gid_i + 1];
    row.partners = partners + begin;
    row.count = end - begin;
    if (row.count > 0) {
        row.lo = row.partners[0];                 // sorted by ensure_exclusion_cache_built
        row.hi = row.partners[row.count - 1];
    }
    return row;
}

inline bool exclusion_row_contains(const ExclusionRow& r, int gid_j) {
    if (gid_j < r.lo || gid_j > r.hi) return false;  // covers count == 0 too
    if (r.count <= 12) {
        for (int k = 0; k < r.count; ++k) {
            if (r.partners[k] == gid_j) return true;
        }
        return false;
    }
    return std::binary_search(r.partners, r.partners + r.count, gid_j);
}

#if defined(__AVX2__)
inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
    return _mm_cvtss_f32(s);
}

inline __m256 rcp_nr_ps(__m256 x) {
    const __m256 r = _mm256_rcp_ps(x);
    return _mm256_mul_ps(r, _mm256_sub_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(x, r)));
}

// Horizontal sum of a 4-wide double vector. Used ONCE per i atom to fold the
// per-i vector accumulators down to scalars -- see the fx_accv/pe_accv
// comment in compute_nonbonded_forces_fused().
inline double hsum256_pd(__m256d v) {
    const __m128d lo = _mm256_castpd256_pd128(v);
    const __m128d hi = _mm256_extractf128_pd(v, 1);
    const __m128d s = _mm_add_pd(lo, hi);
    return _mm_cvtsd_f64(_mm_add_sd(s, _mm_unpackhi_pd(s, s)));
}
#endif

struct LjSplitCacheD {
    int nt = -1;
    const Real* source = nullptr;
    size_t source_size = 0;
    std::vector<Real> c6;
    std::vector<Real> c12;
};

struct LjSplitCacheF {
    int nt = -1;
    const float* source = nullptr;
    size_t source_size = 0;
    std::vector<float> c6;
    std::vector<float> c12;
};

LjSplitCacheD g_lj_split_d;
LjSplitCacheF g_lj_split_f;

inline void ensure_lj_split_cache_d(
    const std::vector<Real>& interleaved, int nt, LjSplitCacheD& cache) {
    const size_t n = static_cast<size_t>(nt) * static_cast<size_t>(nt);
    const bool stale =
        (cache.nt != nt) ||
        (cache.source != interleaved.data()) ||
        (cache.source_size != interleaved.size()) ||
        (cache.c6.size() != n) ||
        (cache.c12.size() != n);
    if (!stale) return;
    cache.c6.assign(n, 0.0);
    cache.c12.assign(n, 0.0);
    for (size_t k = 0; k < n; ++k) {
        cache.c6[k] = interleaved[2 * k];
        cache.c12[k] = interleaved[2 * k + 1];
    }
    cache.nt = nt;
    cache.source = interleaved.data();
    cache.source_size = interleaved.size();
}

inline void ensure_lj_split_cache_f(
    const std::vector<float>& interleaved, int nt, LjSplitCacheF& cache) {
    const size_t n = static_cast<size_t>(nt) * static_cast<size_t>(nt);
    const bool stale =
        (cache.nt != nt) ||
        (cache.source != interleaved.data()) ||
        (cache.source_size != interleaved.size()) ||
        (cache.c6.size() != n) ||
        (cache.c12.size() != n);
    if (!stale) return;
    cache.c6.assign(n, 0.0f);
    cache.c12.assign(n, 0.0f);
    for (size_t k = 0; k < n; ++k) {
        cache.c6[k] = interleaved[2 * k];
        cache.c12[k] = interleaved[2 * k + 1];
    }
    cache.nt = nt;
    cache.source = interleaved.data();
    cache.source_size = interleaved.size();
}

} 

void ForceCalculatorCPU::ensure_lj_coeff_table_built(const ForceFieldParams& params) {
    const int nt = params.num_atom_types;
    if (lj_coeff_num_types_ == nt && !lj_c6c12_.empty()) return;
    lj_c6c12_.assign(static_cast<size_t>(2) * nt * nt, 0.0);
    for (int ti = 0; ti < nt; ++ti) {
        for (int tj = 0; tj < nt; ++tj) {
            const Real eps = params.epsilon_mixed[ti][tj];
            const Real sig = params.sigma_mixed[ti][tj];
            const Real sig2 = sig * sig;
            const Real sig6 = sig2 * sig2 * sig2;
            const size_t k = 2 * (static_cast<size_t>(ti) * nt + tj);
            lj_c6c12_[k]     = 4.0 * eps * sig6;          // c6
            lj_c6c12_[k + 1] = 4.0 * eps * sig6 * sig6;   // c12
        }
    }
    lj_coeff_num_types_ = nt;
}

// See the tile_px_ comment in force_calculator_cpu.h.
void ForceCalculatorCPU::repack_tile_soa(const ParticleData& particles,
                                         const CpuSpatialVerletList& nlist) {
    const int num_tiles = nlist.num_tiles();
    if (num_tiles <= 0 || particles.num_particles == 0) {
        tile_px_.clear(); tile_py_.clear(); tile_pz_.clear(); tile_q_.clear();
        tile_fx_.clear(); tile_fy_.clear(); tile_fz_.clear(); tile_fq_.clear();
        tile_type_.clear(); tile_gid_.clear();
        return;
    }
    const size_t n = static_cast<size_t>(num_tiles) * kCpuTileSize;
    // PERF FIX: process_tile_avx2_mixed() (the mixed-precision kernel) reads
    // ONLY the float mirror below, never tile_px_/py_/pz_/q_ -- so when
    // mixed precision is on, building the double SoA here was pure wasted
    // memory traffic every single step (a full extra n*4 doubles written
    // and never read), doubling this function's write bandwidth and the
    // per-tile working set for no benefit. Skip it (and free any previously
    // allocated double SoA) whenever nothing this step will read it; the
    // non-mixed kernels below still get it built exactly as before.
    const bool need_double_soa = !mixed_precision_nonbonded_;
    if (need_double_soa) {
        tile_px_.resize(n); tile_py_.resize(n); tile_pz_.resize(n); tile_q_.resize(n);
    } else {
        tile_px_.clear(); tile_py_.clear(); tile_pz_.clear(); tile_q_.clear();
        tile_px_.shrink_to_fit(); tile_py_.shrink_to_fit();
        tile_pz_.shrink_to_fit(); tile_q_.shrink_to_fit();
    }
    const bool refresh_static =
        (tile_type_.size() != n) ||
        (tile_gid_.size() != n) ||
        (tile_excl_generation_ != nlist.build_generation());
    if (refresh_static) {
        tile_type_.resize(n);
        tile_gid_.resize(n);
    }

    const int* __restrict__ tile_atoms = nlist.get_tile_atoms();
    const Real* __restrict__ px = particles.x.data();
    const Real* __restrict__ py = particles.y.data();
    const Real* __restrict__ pz = particles.z.data();
    const Real* __restrict__ pq = particles.charge.data();
    const int*  __restrict__ ptype = particles.type.data();
    const int*  __restrict__ pgid = particles.global_id.data();

    Real* __restrict__ ox = need_double_soa ? tile_px_.data() : nullptr;
    Real* __restrict__ oy = need_double_soa ? tile_py_.data() : nullptr;
    Real* __restrict__ oz = need_double_soa ? tile_pz_.data() : nullptr;
    Real* __restrict__ oq = need_double_soa ? tile_q_.data() : nullptr;
    int*  __restrict__ ot = refresh_static ? tile_type_.data() : nullptr;
    int*  __restrict__ og = refresh_static ? tile_gid_.data() : nullptr;

    // Single-precision mirror for the opt-in mixed kernel, produced in the
    // same pass so the gathered per-atom values are only read once.
    float *ofx = nullptr, *ofy = nullptr, *ofz = nullptr, *ofq = nullptr;
    if (mixed_precision_nonbonded_) {
        tile_fx_.resize(n); tile_fy_.resize(n); tile_fz_.resize(n); tile_fq_.resize(n);
        ofx = tile_fx_.data(); ofy = tile_fy_.data();
        ofz = tile_fz_.data(); ofq = tile_fq_.data();
    }

    // Gathered reads, sequential writes. Padding slots (-1) are clamped to
    // atom 0 so every packed value is in-bounds and finite; the kernels mask
    // them out from the RAW tile_atoms entry, never from the packed data.
    #pragma omp parallel for schedule(static)
    for (long k = 0; k < static_cast<long>(n); ++k) {
        const int a = tile_atoms[k];
        const size_t sa = static_cast<size_t>(a < 0 ? 0 : a);
        if (ox) {
            ox[k] = px[sa];
            oy[k] = py[sa];
            oz[k] = pz[sa];
            oq[k] = pq[sa];
        }
        if (ot) {
            ot[k] = ptype[sa];
            og[k] = pgid[sa];
        }
        if (ofx) {
            ofx[k] = static_cast<float>(px[sa]);
            ofy[k] = static_cast<float>(py[sa]);
            ofz[k] = static_cast<float>(pz[sa]);
            ofq[k] = static_cast<float>(pq[sa]);
        }
    }
}

// force_calculation.precision = mixed, called once from MDSimulation's
// initialize_impl() (before any compute_forces() ever runs). Marks
// mixed_precision_probed_ so ensure_mixed_precision_tables()'s lazy
// MD_NONBONDED_PRECISION env-var probe below never overrides an explicit
// config decision -- previously this config value was parsed into
// force_precision_ and then never read anywhere on the CPU backend, so
// "precision: mixed" in the YAML had literally no effect unless the env var
// was ALSO set by hand.
void ForceCalculatorCPU::set_mixed_precision(bool enabled) {
    mixed_precision_probed_ = true;
    if (!enabled) {
        mixed_precision_nonbonded_ = false;
        return;
    }
#if defined(__AVX2__)
    mixed_precision_nonbonded_ = true;
    if (is_rank_zero()) {
        std::cout << "[ForceCalculatorCPU] nonbonded kernel: MIXED precision "
                     "(float pair arithmetic, double accumulation) -- re-enabled by "
                     "request; an earlier measurement on this benchmark showed a "
                     "cache-miss regression from the doubled (double+float) packed-SoA "
                     "working set, so validate wall-clock/energy before trusting the run."
                  << std::endl;
    }
#else
    mixed_precision_nonbonded_ = false;
    if (is_rank_zero()) {
        std::cerr << "[ForceCalculatorCPU] WARNING: force_calculation.precision = mixed "
                     "needs an AVX2 build; falling back to the double kernel." << std::endl;
    }
#endif
}

void ForceCalculatorCPU::ensure_mixed_precision_tables() {
    if (!mixed_precision_probed_) {
        mixed_precision_probed_ = true;
        const char* v = std::getenv("MD_NONBONDED_PRECISION");
        if (v && *v) {
            std::string mode(v);
            std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            mixed_precision_nonbonded_ = (mode == "mixed" || mode == "single" || mode == "float");
            if (mixed_precision_nonbonded_ && is_rank_zero()) {
                std::cout << "[ForceCalculatorCPU] nonbonded kernel: MIXED precision "
                             "(float pair arithmetic, double accumulation)" << std::endl;
            }
        }
#if !defined(__AVX2__)
        if (mixed_precision_nonbonded_) {
            if (is_rank_zero()) {
                std::cerr << "[ForceCalculatorCPU] WARNING: MD_NONBONDED_PRECISION=mixed needs an "
                             "AVX2 build; falling back to the double kernel." << std::endl;
            }
            mixed_precision_nonbonded_ = false;
        }
#endif
    }
    if (!mixed_precision_nonbonded_) return;

    if (lj_c6c12_f_.size() != lj_c6c12_.size()) {
        lj_c6c12_f_.resize(lj_c6c12_.size());
        for (size_t i = 0; i < lj_c6c12_.size(); ++i)
            lj_c6c12_f_[i] = static_cast<float>(lj_c6c12_[i]);
    }
    if (ewald_rs_table_f_.size() != ewald_rs_table_.size()) {
        ewald_rs_table_f_.resize(ewald_rs_table_.size());
        for (size_t i = 0; i < ewald_rs_table_.size(); ++i)
            ewald_rs_table_f_[i] = static_cast<float>(ewald_rs_table_[i]);
    }
    if (ewald_rs_table_excl_f_.size() != ewald_rs_table_excl_.size()) {
        ewald_rs_table_excl_f_.resize(ewald_rs_table_excl_.size());
        for (size_t i = 0; i < ewald_rs_table_excl_.size(); ++i)
            ewald_rs_table_excl_f_[i] = static_cast<float>(ewald_rs_table_excl_[i]);
    }
}

// See the tile_pair_excl_ comment in force_calculator_cpu.h. Call after
// repack_tile_soa() and after ensure_exclusion_cache_built().
void ForceCalculatorCPU::ensure_tile_exclusion_masks(const CpuSpatialVerletList& nlist) {
    const int num_pairs = nlist.num_tile_pairs();
    if (tile_excl_generation_ == nlist.build_generation() &&
        tile_pair_excl_.size() == static_cast<size_t>(num_pairs)) {
        return;
    }
    tile_pair_excl_.assign(static_cast<size_t>(num_pairs), 0);
    const int num_tiles = nlist.num_tiles();
    const int* __restrict__ tile_atoms = nlist.get_tile_atoms();
    const int* __restrict__ tile_offsets = nlist.get_tile_pair_offsets();
    const int* __restrict__ tile_other = nlist.get_tile_pair_other();
    const int* __restrict__ excl_offsets = excl_cache_offsets_.data();
    const int* __restrict__ excl_partners = excl_cache_partners_.data();
    const int* __restrict__ tgid = tile_gid_.data();
    std::uint64_t* __restrict__ out = tile_pair_excl_.data();

    #pragma omp parallel for schedule(static)
    for (int T = 0; T < num_tiles; ++T) {
        // The home tile's exclusion CSR rows are resolved once for the whole
        // tile, then reused across every tile pair this tile takes part in.
        ExclusionRow rows[kCpuTileSize];
        bool row_used[kCpuTileSize];
        for (int lane = 0; lane < kCpuTileSize; ++lane) {
            const int ia = tile_atoms[static_cast<size_t>(T) * kCpuTileSize + lane];
            row_used[lane] = false;
            if (ia < 0) continue;
            rows[lane] = exclusion_row_for(excl_offsets, excl_partners,
                                           tgid[static_cast<size_t>(T) * kCpuTileSize + lane]);
            row_used[lane] = (rows[lane].count > 0);
        }
        for (int e = tile_offsets[T]; e < tile_offsets[T + 1]; ++e) {
            const int O = tile_other[e];
            std::uint64_t mask = 0;
            for (int lane = 0; lane < kCpuTileSize; ++lane) {
                if (!row_used[lane]) continue;
                const ExclusionRow& row = rows[lane];
                for (int slot = 0; slot < kCpuTileSize; ++slot) {
                    const size_t k = static_cast<size_t>(O) * kCpuTileSize + slot;
                    if (tile_atoms[k] < 0) continue;
                    if (exclusion_row_contains(row, tgid[k])) {
                        mask |= static_cast<std::uint64_t>(1) << (lane * kCpuTileSize + slot);
                    }
                }
            }
            out[e] = mask;
        }
    }
    tile_excl_generation_ = nlist.build_generation();
}

// See the lj_tile_valid_mask_/lj_tile_local_mask_/lj_same_tile_mask_
// comment in force_calculator_cpu.h. Call after repack_tile_soa().
void ForceCalculatorCPU::ensure_lj_tile_masks(const CpuSpatialVerletList& nlist, size_t local_particles) {
    const int num_tiles = nlist.num_tiles();
    if (lj_tile_mask_generation_ == nlist.build_generation() &&
        lj_tile_mask_local_particles_ == local_particles &&
        lj_tile_valid_mask_.size() == static_cast<size_t>(num_tiles)) {
        return;
    }
    lj_tile_valid_mask_.assign(static_cast<size_t>(num_tiles), 0);
    lj_tile_local_mask_.assign(static_cast<size_t>(num_tiles), 0);
    lj_same_tile_mask_.assign(static_cast<size_t>(num_tiles), 0);
    const int* __restrict__ tile_atoms = nlist.get_tile_atoms();
    uint8_t* __restrict__ out_valid = lj_tile_valid_mask_.data();
    uint8_t* __restrict__ out_local = lj_tile_local_mask_.data();
    std::uint64_t* __restrict__ out_same = lj_same_tile_mask_.data();

    #pragma omp parallel for schedule(static)
    for (int T = 0; T < num_tiles; ++T) {
        int atoms[kCpuTileSize];
        uint8_t valid = 0, local = 0;
        for (int s = 0; s < kCpuTileSize; ++s) {
            const int a = tile_atoms[static_cast<size_t>(T) * kCpuTileSize + s];
            atoms[s] = a;
            if (a < 0) continue;
            valid |= static_cast<uint8_t>(1u << s);
            if (static_cast<size_t>(a) < local_particles) local |= static_cast<uint8_t>(1u << s);
        }
        out_valid[T] = valid;
        out_local[T] = local;

        // Intra-tile (O==T) half-list tie-break, baked per (j=slot, i=lane):
        // a LOCAL slot atom keeps the pair only when it is the larger of
        // the two indices (each unordered local-local pair is then hit
        // exactly once, by its smaller-index atom acting as "i"); a ghost
        // slot atom carries no such restriction (see compute_lj_forces()'s
        // old scalar "same_tile && j_is_local && j_atom > i_atom" branch).
        std::uint64_t smask = 0;
        for (int slot = 0; slot < kCpuTileSize; ++slot) {
            if (!((valid >> slot) & 1u)) continue;
            const bool slot_local = (local >> slot) & 1u;
            for (int lane = 0; lane < kCpuTileSize; ++lane) {
                if (lane == slot || !((valid >> lane) & 1u)) continue;
                if (slot_local && !(atoms[slot] > atoms[lane])) continue;
                smask |= (static_cast<std::uint64_t>(1) << (slot * kCpuTileSize + lane));
            }
        }
        out_same[T] = smask;
    }
    lj_tile_mask_generation_ = nlist.build_generation();
    lj_tile_mask_local_particles_ = local_particles;
}

// Transpose of tile_pair_excl_: bit (slot*kCpuTileSize+lane) instead of
// (lane*kCpuTileSize+slot). Call after ensure_tile_exclusion_masks().
void ForceCalculatorCPU::ensure_lj_tile_excl_col(const CpuSpatialVerletList& nlist) {
    (void)nlist;
    if (lj_tile_excl_col_generation_ == tile_excl_generation_ &&
        lj_tile_excl_col_.size() == tile_pair_excl_.size()) {
        return;
    }
    const size_t n = tile_pair_excl_.size();
    lj_tile_excl_col_.resize(n);
    const std::uint64_t* __restrict__ in = tile_pair_excl_.data();
    std::uint64_t* __restrict__ out = lj_tile_excl_col_.data();

    #pragma omp parallel for schedule(static)
    for (long e = 0; e < static_cast<long>(n); ++e) {
        std::uint64_t m = in[static_cast<size_t>(e)];
        std::uint64_t t = 0;
        while (m) {
            const int bit = __builtin_ctzll(m);
            m &= (m - 1);
            const int lane = bit / kCpuTileSize;
            const int slot = bit % kCpuTileSize;
            t |= (static_cast<std::uint64_t>(1) << (slot * kCpuTileSize + lane));
        }
        out[static_cast<size_t>(e)] = t;
    }
    lj_tile_excl_col_generation_ = tile_excl_generation_;
}

// PERF FIX (false sharing): replaces the old ensure_lj_tile_plan() tile-
// ownership/remote-reaction-slot scheme. See the lj_thread_fx_ comment in
// force_calculator_cpu.h for why: static edge-balanced ownership avoided
// write RACES but not cache-line PING-PONG between threads, since tile
// membership follows spatial proximity, not original atom-array index.
// Ensures this rank's per-thread private force buffers are sized for the
// current thread count and local particle count; callers zero their own
// slice inside the parallel region right before use (see
// compute_tiled_nonbonded()), so no fill is needed here.
void ForceCalculatorCPU::ensure_lj_thread_buffers(int nthreads, size_t local_particles) {
    const size_t n = static_cast<size_t>(nthreads) * local_particles;
    if (lj_thread_fx_.size() != n) {
        lj_thread_fx_.resize(n);
        lj_thread_fy_.resize(n);
        lj_thread_fz_.resize(n);
    }
}

ForceCalculatorCPU::~ForceCalculatorCPU() {
    free_ewald_resources();
}

void ForceCalculatorCPU::initialize() {
    if (is_rank_zero()) {
        std::cout << "CPU Force Calculator initialized" << std::endl;
    }
}

void ForceCalculatorCPU::finalize() {
    // Nothing to cleanup for CPU version
}

void ForceCalculatorCPU::compute_forces(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    
    // Clear forces
    std::fill(particles.fx.begin(), particles.fx.end(), 0.0);
    std::fill(particles.fy.begin(), particles.fy.end(), 0.0);
    std::fill(particles.fz.begin(), particles.fz.end(), 0.0);
    
    // Reset energies (last_virial is accumulated by LJ + electrostatics below)
    last_potential_energy = 0.0;
    last_kinetic_energy = 0.0;
    last_virial = 0.0;
    
    // Get force field configuration from force manager
    ForceManager& force_manager = system.get_force_manager();
    const auto& force_config = force_manager.get_config();

    // The CPU LJ kernels (compute_lj_forces/compute_nonbonded_forces_fused)
    // hardcode the standard Mie(12,6) exponents for performance (no more
    // per-pair n/m branch) -- SPICA's Mie(9,6) is not supported on this
    // backend. Fail fast instead of silently computing wrong LJ forces.
    if (force_config.enable_lennard_jones &&
        system.get_ff_params().mie_repulsive_exponent != 12) {
        throw std::runtime_error(
            "ForceCalculatorCPU: only Mie(12,6) Lennard-Jones is supported "
            "(got exponents " +
            std::to_string(system.get_ff_params().mie_repulsive_exponent) + "," +
            std::to_string(system.get_ff_params().mie_attractive_exponent) +
            "). Mie(9,6)/SPICA is not implemented on the CPU backend.");
    }

    // Diagnostic per-force-component timing (MD_TIMING_TRACE_FORCE=1). Wall-
    // clock (not cudaEvent, there's no device queue here) around each
    // component; near-zero overhead when disabled. compute_graph_nonbonded()/
    // compute_tiled_nonbonded() push their own component timing directly
    // (see their bodies) instead of going through the time_component()
    // wrapper here, since they also need to time the PME reciprocal-space
    // overlap around the same call.
    if (timing_breakdown_enabled_) {
        last_force_component_timings_.clear();
    }
    using clock = std::chrono::high_resolution_clock;
    auto time_component = [&](const char* name, auto&& fn) {
        if (!timing_breakdown_enabled_) {
            fn();
            return;
        }
        auto t0 = clock::now();
        fn();
        auto t1 = clock::now();
        last_force_component_timings_.push_back(
            {name, std::chrono::duration<double, std::milli>(t1 - t0).count()});
    };

    // Compute different force contributions based on configuration
    if (force_config.enable_bonds) {
        time_component("bond", [&]{ compute_bond_forces(system); });         // Bonded interactions
    }
    if (force_config.enable_angles) {
        time_component("angle", [&]{ compute_angle_forces(system); });       // Angle bending forces
    }
    if (force_config.enable_dihedrals) {
        time_component("dihedral", [&]{
            compute_dihedral_forces(system);     // Torsional forces
            compute_improper_forces(system);     // Improper torsions (same functional form)
        });
    }
    // LJ and/or Coulomb over the neighbor list. compute_forces() only picks
    // which KERNEL BACKEND walks the neighbor list (graph vs tiled); each of
    // compute_graph_nonbonded()/compute_tiled_nonbonded() owns everything
    // about which interaction(s) it computes -- LJ-only vs Coulomb-only vs
    // fused, the PME reciprocal-space overlap, and its own component timing.
    if (force_config.enable_lennard_jones || force_config.enable_coulomb) {
        if (nonbonded_kernel_ == "graph") {
            compute_graph_nonbonded(system, force_config.enable_lennard_jones,
                                    force_config.enable_coulomb);
        } else {
            compute_tiled_nonbonded(system, force_config.enable_lennard_jones,
                                    force_config.enable_coulomb);
        }
    }
    if (force_config.enable_lennard_jones || force_config.enable_coulomb) {
        time_component("special_pairs", [&]{ compute_special_pair_forces(system); }); // Scaled 1-4 nonbonded pairs
    }

    // MPI-AWARE: Sum potential energy across all ranks
    // Each rank computes PE for pairs it "owns" (global_id[i] < global_id[j]),
    // so we need to sum these partial energies to get the total system PE
    // OPTIMIZATION: Batch the two MPI_Allreduce calls (PE reduction + particle count)
    // into a single call to reduce communication overhead.
    size_t local_owned = system.get_local_particles();
    Real batch_in[2]  = { last_potential_energy, static_cast<Real>(local_owned) };
    Real batch_out[2] = { 0.0, 0.0 };
    MPI_Allreduce(batch_in, batch_out, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    last_potential_energy = batch_out[0];
    (void)batch_out[1];

    // NOTE: Long-range tail correction is NOT applied here.
    // LAMMPS default: truncated LJ at cutoff with NO tail correction unless
    // "pair_modify tail yes" is explicitly specified.  Applying tail correction
    // would add ~-14.6 kcal/mol to PE (for 32K Ar @ cutoff=12Å), making DS CPU
    // diverge from LAMMPS and DS GPU by that amount.  Tail correction can be
    // enabled separately if needed, but must also be added to the virial/pressure
    // calculation to remain thermodynamically consistent.
}

// Compact directed-CSR nonbonded path. Each row is owned by one source atom,
// so threads only update F[i]; the reverse force is produced independently by
// j's row (or by the owning rank for a ghost target). This deliberately gives
// up Newton half-list arithmetic to remove reaction atomics and tile padding.
void ForceCalculatorCPU::compute_graph_nonbonded(MDSystem& system,
                                                 bool do_lj,
                                                 bool do_coulomb) {
    const char* label = do_lj ? (do_coulomb ? "LJ+PME" : "LJ") : "PME";
    run_nonbonded_with_pme_overlap(system, do_coulomb, label, [&] {
    ParticleData& p = system.get_particles();
    NeighborList& graph = system.get_neighbor_list();
    ForceFieldParams& params = system.get_ff_params();
    SimulationBox& box = system.get_box();
    Topology& topology = system.get_topology();
    ensure_exclusion_cache_built(topology);
    const size_t nlocal = system.get_local_particles();
    const size_t ntotal = p.num_particles;
    const int* offsets = graph.get_active_offsets();
    const int* col = graph.get_active_neighbors();
    if (!offsets || (graph.num_active_edges() > 0 && !col)) {
        throw std::runtime_error("graph nonbonded kernel requires an active directed CSR");
    }
    if (!graph_kernel_logged_) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::cout << "[ForceCalculatorCPU][rank " << rank
                  << "] executing nonbonded_kernel=graph"
                  << " owned_atoms=" << nlocal
                  << " active_edges=" << graph.num_active_edges()
                  << " lj=" << (do_lj ? "on" : "off")
                  << " coulomb=" << (do_coulomb ? "on" : "off")
                  << std::endl;
        graph_kernel_logged_ = true;
    }

    if (do_lj) ensure_lj_coeff_table_built(params);
    const Real cutoff = system.get_physical_cutoff();
    const Real cutoff_sq = cutoff * cutoff;
    const bool switching = do_lj && params.lj_switch_distance > 0.0;
    const Real switch_sq = switching
        ? params.lj_switch_distance * params.lj_switch_distance : 0.0;
    const Real inv_denom = switching
        ? 1.0 / std::pow(cutoff_sq - switch_sq, 3) : 0.0;
    const bool mpi_box = box.is_mpi_subdomain();
    constexpr Real k_coulomb = 332.0636;

    if (lj_pair_counts_.size() != nlocal) lj_pair_counts_.resize(nlocal);
    Real local_pe = 0.0;
    Real local_virial = 0.0;

    #pragma omp parallel for schedule(dynamic, 64) reduction(+:local_pe,local_virial)
    for (long ii = 0; ii < static_cast<long>(nlocal); ++ii) {
        const size_t i = static_cast<size_t>(ii);
        Real fix = 0.0, fiy = 0.0, fiz = 0.0;
        int lj_count = 0;
        const int gid_i = p.global_id[i];
        const ExclusionRow exclusion_row = exclusion_row_for(
            excl_cache_offsets_.data(), excl_cache_partners_.data(), gid_i);
        for (int e = offsets[i]; e < offsets[i + 1]; ++e) {
            const int j_raw = col[static_cast<size_t>(e)];
            if (j_raw < 0 || static_cast<size_t>(j_raw) >= ntotal) continue;
            const size_t j = static_cast<size_t>(j_raw);
            Real dx = p.x[i] - p.x[j];
            Real dy = p.y[i] - p.y[j];
            Real dz = p.z[i] - p.z[j];
            if (!mpi_box || j < nlocal) box.apply_minimum_image(dx, dy, dz);
            const Real r2 = dx * dx + dy * dy + dz * dz;
            if (r2 >= cutoff_sq || r2 < 1e-10) continue;

            const int gid_j = p.global_id[j];
            const bool excluded = exclusion_row_contains(exclusion_row, gid_j);
            Real force_mag = 0.0;
            Real pair_energy = 0.0;

            if (do_lj && !excluded) {
                const size_t tij = static_cast<size_t>(p.type[i]) *
                    static_cast<size_t>(lj_coeff_num_types_) + static_cast<size_t>(p.type[j]);
                Real lj_force, lj_energy;
                compute_lj_pair(r2, lj_c6c12_[2 * tij], lj_c6c12_[2 * tij + 1], cutoff_sq,
                                switching, switch_sq, inv_denom, lj_force, lj_energy);
                force_mag += lj_force;
                pair_energy += lj_energy;
                ++lj_count;
            }

            if (do_coulomb) {
                Real coulomb_force, coulomb_energy;
                compute_coulomb_pair_real_space(
                    excluded ? ewald_rs_table_excl_.data() : ewald_rs_table_.data(),
                    ewald_rs_table_r2_min_, ewald_rs_table_inv_spacing_, r2,
                    p.charge[i], p.charge[j], k_coulomb, coulomb_force, coulomb_energy);
                pair_energy += coulomb_energy;
                force_mag += coulomb_force;
            }

            commit_directed_pair_contribution(force_mag, pair_energy, dx, dy, dz, r2,
                                              j < nlocal, gid_i < gid_j,
                                              fix, fiy, fiz, local_pe, local_virial);
        }
        p.fx[i] += fix;
        p.fy[i] += fiy;
        p.fz[i] += fiz;
        lj_pair_counts_[i] = lj_count;
    }

    last_potential_energy += local_pe;
    last_virial += local_virial;
    }); // run_nonbonded_with_pme_overlap
}

struct TiledNonbondedTileCtx {
    bool do_lj = false;
    bool do_coulomb = false;

    int lj_num_types = 0;
    const Real* lj_c6 = nullptr;
    const Real* lj_c12 = nullptr;
    bool lj_switching_enabled = false;
    Real switch_sq = 0.0;
    Real inv_denom = 0.0;
    Real cutoff_sq = 0.0;

    const Real* rs_table = nullptr;
    const Real* rs_table_excl = nullptr;
    Real rs_r2_min = 0.0;
    Real rs_inv_spacing = 0.0;
    Real k_coulomb = 0.0;

    const int* tile_atoms = nullptr;
    const int* tile_offsets = nullptr;
    const int* tile_other = nullptr;
    const std::uint64_t* verlet_mask = nullptr;
    const Real* tile_shift_x = nullptr;
    const Real* tile_shift_y = nullptr;
    const Real* tile_shift_z = nullptr;

    const Real* tpx = nullptr;
    const Real* tpy = nullptr;
    const Real* tpz = nullptr;
    const Real* tpq = nullptr;
    const int* ttype = nullptr;
    const int* tgid = nullptr;
    const std::uint64_t* excl_col = nullptr;
    const std::uint64_t* same_tile_mask_of = nullptr;
    const uint8_t* valid_mask_of = nullptr;
    const uint8_t* local_mask_of = nullptr;

    // Base pointer of the nthreads*thread_stride private force buffer (see
    // the lj_thread_fx_ comment in force_calculator_cpu.h); each caller
    // offsets by its own tid*thread_stride before touching it, never the
    // shared particles.fx/fy/fz array directly.
    Real* fx_out = nullptr;
    Real* fy_out = nullptr;
    Real* fz_out = nullptr;
    size_t thread_stride = 0;
    int* lj_count_out = nullptr;

#if defined(__AVX512F__)
    __m512d vcutoff2{}, vr2min{}, vone{}, v12{}, v6{};
    __m256i vlj_num_types{};
    __m512d vk_coulomb{}, v_rs_r2_min{}, v_rs_inv_spacing{};
#elif defined(__AVX2__)
    __m256d vcutoff2{}, vr2min{}, vone{}, v12{}, v6{};
    __m128i vlj_num_types{};
    __m256d vk_coulomb{}, v_rs_r2_min{}, v_rs_inv_spacing{};
#endif

    // ===== MIXED PRECISION (float pair arithmetic, double accumulation) =====
    // See ForceCalculatorCPU::ensure_mixed_precision_tables()'s header
    // comment. Populated only when use_mixed_precision is true; independent
    // of the AVX-512/AVX2 double-precision fields above (a separate,
    // non-exclusive #if) since AVX2 -- everything process_tile_avx2_mixed()
    // needs -- is available whenever AVX-512F is (see the dispatch site in
    // compute_tiled_nonbonded()).
    bool use_mixed_precision = false;
    const float* lj_c6_f = nullptr;
    const float* lj_c12_f = nullptr;
    const float* rs_table_f = nullptr;
    const float* rs_table_excl_f = nullptr;
    float rs_r2_min_f = 0.0f;
    float rs_inv_spacing_f = 0.0f;
    float k_coulomb_f = 0.0f;
    float cutoff_sq_f = 0.0f;
    float switch_sq_f = 0.0f;
    float inv_denom_f = 0.0f;
    const float* tpx_f = nullptr;
    const float* tpy_f = nullptr;
    const float* tpz_f = nullptr;
    const float* tpq_f = nullptr;
#if defined(__AVX2__)
    __m256i vlj_num_types_f{};
    __m256 vcutoff2_f{}, vr2min_f{}, vone_f{}, v12_f{}, v6_f{};
    __m256 vk_coulomb_f{}, v_rs_r2_min_f{}, v_rs_inv_spacing_f{};
#endif
};

#if defined(__AVX2__)
// ===== MIXED PRECISION (float, 8-wide) tile kernel =====
// Independent of the AVX-512/AVX2 double-precision kernels below: this
// compiles whenever AVX2 is available, including on an AVX-512 build (AVX2
// is always available there too), and is dispatched by
// compute_tiled_nonbonded() ahead of process_tile_avx512()/process_tile_
// avx2() whenever ctx.use_mixed_precision is set (see the dispatch site).
//
// One tile's 8 home (i) lanes fit in ONE 8-wide __m256 float vector --
// unlike process_tile_avx2()'s double kernel below, which needs two 4-wide
// halves for the same 8 lanes -- so a tile-pair edge's LJ/Coulomb pair math
// runs as a SINGLE 8-wide float pass here instead of two 4-wide double
// passes: half the gather/divide instructions, half the load traffic, for
// every pair. Each pair's resulting force/energy/virial contribution is
// widened to double immediately (a couple of cheap _mm256_cvtps_pd calls,
// well off the gather/divide critical path) before folding into the running
// per-tile accumulators, so -- as documented in force_calculator_cpu.h's
// mixed-precision section -- only the PAIR PHYSICS below runs in float;
// every SUM that could otherwise compound rounding error across a busy
// atom's dozens of neighbours (forces, energy, virial) still accumulates in
// double, exactly like the double-precision kernels below.

// 8-lane analogue of mask4_from_bits() above: expands bit i (i=0..7) of
// `bits8` into lane i of an 8-wide "all-ones/all-zeros" float mask. Float
// lanes are already 32-bit, so (unlike the double kernels' epi32->epi64
// widen) this is a single epi32 compare-against-zero.
inline __m256 mask8_from_bits_ps(unsigned bits8) {
    const __m256i shift_amt = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i bcast = _mm256_set1_epi32(static_cast<int>(bits8));
    const __m256i bit = _mm256_and_si256(_mm256_srlv_epi32(bcast, shift_amt), _mm256_set1_epi32(1));
    const __m256i mask32 = _mm256_sub_epi32(_mm256_setzero_si256(), bit);  // 0 or -1 per lane
    return _mm256_castsi256_ps(mask32);
}

// Float, 8-wide analogue of compute_lj_avx2() above -- see its comment for
// the shared algorithm. lj_cnt_vec here holds 8 lanes of int32 (not 4 lanes
// of int64 like the double kernel): a per-pair count only ever needs to
// reach a few hundred, well within int32, and 8 lanes is the natural width
// for a 256-bit register of 32-bit ints.
inline __m256 compute_lj_avx2_f(
        __m256 r2, __m256 cutoff_mask, __m256 excl_mask,
        __m256i type_i_vec, __m256i vlj_num_types, int j_type,
        const float* lj_c6, const float* lj_c12,
        __m256 vone, __m256 v12, __m256 v6,
        float cutoff_sq, bool lj_switching_enabled, float switch_sq, float inv_denom,
        __m256i& lj_cnt_vec, __m256i lj_cnt_one,
        __m256& force_mag, __m256& pe_pair) {
    const __m256 lj_mask = _mm256_andnot_ps(excl_mask, cutoff_mask);
    if (_mm256_movemask_ps(lj_mask) == 0) {
        force_mag = _mm256_setzero_ps();
        pe_pair = _mm256_setzero_ps();
        return lj_mask;
    }

    const __m256i idx = _mm256_add_epi32(
        _mm256_mullo_epi32(type_i_vec, vlj_num_types), _mm256_set1_epi32(j_type));
    const __m256 c6 = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), lj_c6, idx, lj_mask, 4);
    const __m256 c12 = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), lj_c12, idx, lj_mask, 4);
    // NOTE: tried rcp_nr_ps() (Newton-Raphson reciprocal) here instead of an
    // exact divide -- measured a net REGRESSION together with the
    // compute_pme_avx2_f() exclusion-skip branch below, reverted pending
    // isolated re-measurement of each change separately.
    const __m256 r2inv = _mm256_div_ps(vone, r2);
    const __m256 r4inv = _mm256_mul_ps(r2inv, r2inv);
    const __m256 r6inv = _mm256_mul_ps(r4inv, r2inv);
    const __m256 attractive = _mm256_mul_ps(c6, r6inv);
    const __m256 repulsive = _mm256_mul_ps(c12, _mm256_mul_ps(r6inv, r6inv));

    force_mag = _mm256_mul_ps(r2inv,
        _mm256_sub_ps(_mm256_mul_ps(v12, repulsive), _mm256_mul_ps(v6, attractive)));
    pe_pair = _mm256_sub_ps(repulsive, attractive);

    if (lj_switching_enabled) {
        const __m256 sw = _mm256_and_ps(lj_mask, _mm256_cmp_ps(r2, _mm256_set1_ps(switch_sq), _CMP_GT_OQ));
        if (_mm256_movemask_ps(sw) != 0) {
            const __m256 a = _mm256_sub_ps(_mm256_set1_ps(cutoff_sq), r2);
            const __m256 b = _mm256_sub_ps(
                _mm256_add_ps(_mm256_set1_ps(cutoff_sq), _mm256_mul_ps(_mm256_set1_ps(2.0f), r2)),
                _mm256_set1_ps(3.0f * switch_sq));
            const __m256 switch_value = _mm256_mul_ps(
                _mm256_mul_ps(a, a), _mm256_mul_ps(b, _mm256_set1_ps(inv_denom)));
            const __m256 force_correction = _mm256_mul_ps(
                _mm256_set1_ps(12.0f), _mm256_mul_ps(pe_pair,
                _mm256_mul_ps(a, _mm256_mul_ps(
                    _mm256_sub_ps(r2, _mm256_set1_ps(switch_sq)),
                    _mm256_set1_ps(inv_denom)))));
            force_mag = _mm256_blendv_ps(force_mag,
                _mm256_add_ps(_mm256_mul_ps(switch_value, force_mag), force_correction), sw);
            pe_pair = _mm256_blendv_ps(pe_pair, _mm256_mul_ps(pe_pair, switch_value), sw);
        }
    }

    lj_cnt_vec = _mm256_add_epi32(lj_cnt_vec,
        _mm256_and_si256(_mm256_castps_si256(lj_mask), lj_cnt_one));
    return lj_mask;
}

// Float, 8-wide analogue of compute_pme_avx2() above -- see its comment for
// the shared table-lookup algorithm. The float mirror tables
// (ForceCalculatorCPU::ewald_rs_table_f_/_excl_f_) use the SAME interleaved
// [e0,f0,e1,f1,...] layout as the double tables, just as float instead of
// double, so the index arithmetic is identical -- only the gather scale (4
// bytes, not 8) and the arithmetic type change.
inline void compute_pme_avx2_f(
        __m256 r2, __m256 cutoff_mask, __m256 excl_mask,
        const float* rs_table, const float* rs_table_excl,
        __m256 v_rs_r2_min, __m256 v_rs_inv_spacing,
        __m256 vk_coulomb, __m256 qi_vec, float qj,
        __m256& force_mag_c, __m256& pe_c) {
    const __m256 idx_f = _mm256_max_ps(
        _mm256_mul_ps(_mm256_sub_ps(r2, v_rs_r2_min), v_rs_inv_spacing), _mm256_setzero_ps());
    const __m256i idx_i = _mm256_cvttps_epi32(idx_f);
    const __m256 frac = _mm256_sub_ps(idx_f, _mm256_cvtepi32_ps(idx_i));
    const __m256i idx2 = _mm256_slli_epi32(idx_i, 1);
    const __m256i idx2p1 = _mm256_add_epi32(idx2, _mm256_set1_epi32(1));
    const __m256i idx2p2 = _mm256_add_epi32(idx2, _mm256_set1_epi32(2));
    const __m256i idx2p3 = _mm256_add_epi32(idx2, _mm256_set1_epi32(3));
    const __m256 non_excl_mask = _mm256_andnot_ps(excl_mask, cutoff_mask);
    const __m256 excl_only_mask = _mm256_and_ps(excl_mask, cutoff_mask);

    __m256 e0 = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), rs_table, idx2, non_excl_mask, 4);
    __m256 f0 = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), rs_table, idx2p1, non_excl_mask, 4);
    __m256 e1 = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), rs_table, idx2p2, non_excl_mask, 4);
    __m256 f1 = _mm256_mask_i32gather_ps(_mm256_setzero_ps(), rs_table, idx2p3, non_excl_mask, 4);
    // NOTE: tried branching on excl_only_mask here to skip these four
    // gathers when this 8-lane group has no excluded pair -- measured a net
    // REGRESSION (branch misprediction cost apparently outweighing the
    // skipped-gather savings), reverted to the unconditional form pending
    // isolated re-measurement.
    e0 = _mm256_mask_i32gather_ps(e0, rs_table_excl, idx2, excl_only_mask, 4);
    f0 = _mm256_mask_i32gather_ps(f0, rs_table_excl, idx2p1, excl_only_mask, 4);
    e1 = _mm256_mask_i32gather_ps(e1, rs_table_excl, idx2p2, excl_only_mask, 4);
    f1 = _mm256_mask_i32gather_ps(f1, rs_table_excl, idx2p3, excl_only_mask, 4);

    const __m256 energy_factor = _mm256_add_ps(e0, _mm256_mul_ps(frac, _mm256_sub_ps(e1, e0)));
    const __m256 force_factor  = _mm256_add_ps(f0, _mm256_mul_ps(frac, _mm256_sub_ps(f1, f0)));
    const __m256 qq = _mm256_mul_ps(vk_coulomb, _mm256_mul_ps(qi_vec, _mm256_set1_ps(qj)));
    force_mag_c = _mm256_mul_ps(qq, force_factor);
    pe_c = _mm256_mul_ps(qq, energy_factor);
}

// Widens an 8-wide float vector's low/high 4 lanes into two double halves,
// matching process_tile_avx2()'s (fix[0]=lanes 0-3, fix[1]=lanes 4-7) split.
inline void widen_ps_to_pd2(__m256 v, __m256d& lo, __m256d& hi) {
    lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));
    hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
}

// Caller (compute_tiled_nonbonded()) has already checked that T has local
// atoms, exactly like process_tile_avx512()/process_tile_avx2() below.
inline void process_tile_avx2_mixed(int T, int tid, const TiledNonbondedTileCtx& ctx,
                                    Real& local_pe, Real& local_virial) {
    constexpr int TILE_SIZE = kCpuTileSize;
    const size_t tbase = static_cast<size_t>(T) * TILE_SIZE;
    const uint8_t i_local8 = ctx.local_mask_of[T];
    Real* __restrict__ fx_out = ctx.fx_out + static_cast<size_t>(tid) * ctx.thread_stride;
    Real* __restrict__ fy_out = ctx.fy_out + static_cast<size_t>(tid) * ctx.thread_stride;
    Real* __restrict__ fz_out = ctx.fz_out + static_cast<size_t>(tid) * ctx.thread_stride;

    const float* __restrict__ ix = ctx.tpx_f + tbase;
    const float* __restrict__ iy = ctx.tpy_f + tbase;
    const float* __restrict__ iz = ctx.tpz_f + tbase;
    const float* __restrict__ iq = ctx.do_coulomb ? (ctx.tpq_f + tbase) : nullptr;
    const int*   __restrict__ i_type_s = ctx.ttype + tbase;
    const int*   __restrict__ i_gid_s  = ctx.tgid + tbase;

    const __m256 xi = _mm256_loadu_ps(ix);
    const __m256 yi = _mm256_loadu_ps(iy);
    const __m256 zi = _mm256_loadu_ps(iz);
    const __m256 qi_vec = ctx.do_coulomb ? _mm256_loadu_ps(iq) : _mm256_setzero_ps();
    const __m256i type_i_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(i_type_s));
    const __m256i gid_i_vec  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(i_gid_s));

    // Double accumulators (two 4-wide halves, like process_tile_avx2()) --
    // see the file comment above: pair PHYSICS runs in float, every SUM
    // stays double.
    __m256d fix[2] = {_mm256_setzero_pd(), _mm256_setzero_pd()};
    __m256d fiy[2] = {_mm256_setzero_pd(), _mm256_setzero_pd()};
    __m256d fiz[2] = {_mm256_setzero_pd(), _mm256_setzero_pd()};
    __m256d pe_acc[2] = {_mm256_setzero_pd(), _mm256_setzero_pd()};
    __m256d vir_acc[2] = {_mm256_setzero_pd(), _mm256_setzero_pd()};
    __m256i lj_cnt_vec = _mm256_setzero_si256();
    const __m256i lj_cnt_one = _mm256_set1_epi32(1);

    for (int e = ctx.tile_offsets[T]; e < ctx.tile_offsets[T + 1]; ++e) {
        const std::uint64_t vmask_full = ctx.verlet_mask[e];
        if (vmask_full == 0) continue;
        const int O = ctx.tile_other[e];
        const bool same_tile = (O == T);
        const size_t obase = static_cast<size_t>(O) * TILE_SIZE;
        const uint8_t j_valid8 = ctx.valid_mask_of[O];
        if (j_valid8 == 0) continue;
        const uint8_t j_local8 = ctx.local_mask_of[O];
        const std::uint64_t ecol = ctx.excl_col[e];
        const std::uint64_t smask_full = same_tile ? ctx.same_tile_mask_of[T] : 0;
        const float edge_sx = static_cast<float>(ctx.tile_shift_x[e]);
        const float edge_sy = static_cast<float>(ctx.tile_shift_y[e]);
        const float edge_sz = static_cast<float>(ctx.tile_shift_z[e]);

        const int*   __restrict__ j_atom_s = ctx.tile_atoms + obase;
        const float* __restrict__ jx = ctx.tpx_f + obase;
        const float* __restrict__ jy = ctx.tpy_f + obase;
        const float* __restrict__ jz = ctx.tpz_f + obase;
        const float* __restrict__ jq = ctx.do_coulomb ? (ctx.tpq_f + obase) : nullptr;
        const int*   __restrict__ j_type_s = ctx.ttype + obase;
        const int*   __restrict__ j_gid_s  = ctx.tgid + obase;

        for (int slot = 0; slot < TILE_SIZE; ++slot) {
            if (!((j_valid8 >> slot) & 1u)) continue;
            const bool slot_local = (j_local8 >> slot) & 1u;
            const int j_atom = j_atom_s[slot];

            __m256 base_mask = mask8_from_bits_ps(
                static_cast<unsigned>(i_local8) &
                static_cast<unsigned>((vmask_full >> (slot * TILE_SIZE)) & 0xFFu));
            if (same_tile) {
                base_mask = _mm256_and_ps(base_mask, mask8_from_bits_ps(
                    static_cast<unsigned>((smask_full >> (slot * TILE_SIZE)) & 0xFFu)));
            }
            if (_mm256_movemask_ps(base_mask) == 0) continue;

            const __m256 dx = _mm256_sub_ps(xi, _mm256_set1_ps(jx[slot] + edge_sx));
            const __m256 dy = _mm256_sub_ps(yi, _mm256_set1_ps(jy[slot] + edge_sy));
            const __m256 dz = _mm256_sub_ps(zi, _mm256_set1_ps(jz[slot] + edge_sz));
            const __m256 r2 = _mm256_fmadd_ps(dx, dx, _mm256_fmadd_ps(dy, dy, _mm256_mul_ps(dz, dz)));

            __m256 cutoff_mask = _mm256_and_ps(base_mask, _mm256_cmp_ps(r2, ctx.vcutoff2_f, _CMP_LT_OQ));
            cutoff_mask = _mm256_and_ps(cutoff_mask, _mm256_cmp_ps(r2, ctx.vr2min_f, _CMP_GE_OQ));
            if (_mm256_movemask_ps(cutoff_mask) == 0) continue;

            const unsigned excl_bits = static_cast<unsigned>((ecol >> (slot * TILE_SIZE)) & 0xFFu);
            const __m256 excl_mask = mask8_from_bits_ps(excl_bits);

            __m256 owner_bits;
            if (slot_local) {
                owner_bits = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
            } else {
                const __m256i gt = _mm256_cmpgt_epi32(_mm256_set1_epi32(j_gid_s[slot]), gid_i_vec);
                owner_bits = _mm256_castsi256_ps(gt);
            }

            __m256 force_mag = _mm256_setzero_ps(), pe_pair = _mm256_setzero_ps();
            __m256 lj_mask = _mm256_setzero_ps();
            if (ctx.do_lj) {
                lj_mask = compute_lj_avx2_f(r2, cutoff_mask, excl_mask,
                    type_i_vec, ctx.vlj_num_types_f, j_type_s[slot], ctx.lj_c6_f, ctx.lj_c12_f,
                    ctx.vone_f, ctx.v12_f, ctx.v6_f, ctx.cutoff_sq_f, ctx.lj_switching_enabled,
                    ctx.switch_sq_f, ctx.inv_denom_f, lj_cnt_vec, lj_cnt_one, force_mag, pe_pair);
            }

            __m256 force_mag_c = _mm256_setzero_ps(), pe_c = _mm256_setzero_ps();
            if (ctx.do_coulomb) {
                compute_pme_avx2_f(r2, cutoff_mask, excl_mask, ctx.rs_table_f, ctx.rs_table_excl_f,
                    ctx.v_rs_r2_min_f, ctx.v_rs_inv_spacing_f, ctx.vk_coulomb_f, qi_vec, jq[slot],
                    force_mag_c, pe_c);
            }

            __m256 fxv, fyv, fzv;
            if (ctx.do_lj && ctx.do_coulomb) {
                fxv = _mm256_add_ps(_mm256_and_ps(lj_mask, _mm256_mul_ps(force_mag, dx)),
                                    _mm256_and_ps(cutoff_mask, _mm256_mul_ps(force_mag_c, dx)));
                fyv = _mm256_add_ps(_mm256_and_ps(lj_mask, _mm256_mul_ps(force_mag, dy)),
                                    _mm256_and_ps(cutoff_mask, _mm256_mul_ps(force_mag_c, dy)));
                fzv = _mm256_add_ps(_mm256_and_ps(lj_mask, _mm256_mul_ps(force_mag, dz)),
                                    _mm256_and_ps(cutoff_mask, _mm256_mul_ps(force_mag_c, dz)));
            } else if (ctx.do_lj) {
                fxv = _mm256_and_ps(lj_mask, _mm256_mul_ps(force_mag, dx));
                fyv = _mm256_and_ps(lj_mask, _mm256_mul_ps(force_mag, dy));
                fzv = _mm256_and_ps(lj_mask, _mm256_mul_ps(force_mag, dz));
            } else {
                fxv = _mm256_and_ps(cutoff_mask, _mm256_mul_ps(force_mag_c, dx));
                fyv = _mm256_and_ps(cutoff_mask, _mm256_mul_ps(force_mag_c, dy));
                fzv = _mm256_and_ps(cutoff_mask, _mm256_mul_ps(force_mag_c, dz));
            }

            __m256d fxv_lo, fxv_hi, fyv_lo, fyv_hi, fzv_lo, fzv_hi;
            widen_ps_to_pd2(fxv, fxv_lo, fxv_hi);
            widen_ps_to_pd2(fyv, fyv_lo, fyv_hi);
            widen_ps_to_pd2(fzv, fzv_lo, fzv_hi);
            fix[0] = _mm256_add_pd(fix[0], fxv_lo); fix[1] = _mm256_add_pd(fix[1], fxv_hi);
            fiy[0] = _mm256_add_pd(fiy[0], fyv_lo); fiy[1] = _mm256_add_pd(fiy[1], fyv_hi);
            fiz[0] = _mm256_add_pd(fiz[0], fzv_lo); fiz[1] = _mm256_add_pd(fiz[1], fzv_hi);

            if (ctx.do_lj) {
                const __m256 energy_mask_lj = _mm256_and_ps(lj_mask, owner_bits);
                const __m256 pe_lj = _mm256_and_ps(energy_mask_lj, pe_pair);
                const __m256 vir_lj = _mm256_and_ps(energy_mask_lj, _mm256_mul_ps(r2, force_mag));
                __m256d pe_lo, pe_hi, vir_lo, vir_hi;
                widen_ps_to_pd2(pe_lj, pe_lo, pe_hi);
                widen_ps_to_pd2(vir_lj, vir_lo, vir_hi);
                pe_acc[0] = _mm256_add_pd(pe_acc[0], pe_lo); pe_acc[1] = _mm256_add_pd(pe_acc[1], pe_hi);
                vir_acc[0] = _mm256_add_pd(vir_acc[0], vir_lo); vir_acc[1] = _mm256_add_pd(vir_acc[1], vir_hi);
            }
            if (ctx.do_coulomb) {
                const __m256 energy_mask_c = _mm256_and_ps(cutoff_mask, owner_bits);
                const __m256 pe_c_m = _mm256_and_ps(energy_mask_c, pe_c);
                const __m256 vir_c_m = _mm256_and_ps(energy_mask_c, _mm256_mul_ps(r2, force_mag_c));
                __m256d pe_lo, pe_hi, vir_lo, vir_hi;
                widen_ps_to_pd2(pe_c_m, pe_lo, pe_hi);
                widen_ps_to_pd2(vir_c_m, vir_lo, vir_hi);
                pe_acc[0] = _mm256_add_pd(pe_acc[0], pe_lo); pe_acc[1] = _mm256_add_pd(pe_acc[1], pe_hi);
                vir_acc[0] = _mm256_add_pd(vir_acc[0], vir_lo); vir_acc[1] = _mm256_add_pd(vir_acc[1], vir_hi);
            }

            // j reaction (Newton's 3rd law): widen the already-computed
            // float fxv/fyv/fzv once more and horizontally sum both double
            // halves -- same double-accumulation guarantee as the home (i)
            // side above.
            if (slot_local) {
                __m256d jlo, jhi;
                widen_ps_to_pd2(fxv, jlo, jhi);
                const Real jfx = -(hsum256_pd(jlo) + hsum256_pd(jhi));
                widen_ps_to_pd2(fyv, jlo, jhi);
                const Real jfy = -(hsum256_pd(jlo) + hsum256_pd(jhi));
                widen_ps_to_pd2(fzv, jlo, jhi);
                const Real jfz = -(hsum256_pd(jlo) + hsum256_pd(jhi));
                commit_reaction_force(j_atom, jfx, jfy, jfz, fx_out, fy_out, fz_out);
            }
        }
    }

    Real fi_x[TILE_SIZE], fi_y[TILE_SIZE], fi_z[TILE_SIZE];
    Real pe_tile = 0.0, vir_tile = 0.0;
    int32_t lj_cnt_arr[TILE_SIZE];
    for (int h = 0; h < 2; ++h) {
        _mm256_storeu_pd(fi_x + h * 4, fix[h]);
        _mm256_storeu_pd(fi_y + h * 4, fiy[h]);
        _mm256_storeu_pd(fi_z + h * 4, fiz[h]);
        pe_tile += hsum256_pd(pe_acc[h]);
        vir_tile += hsum256_pd(vir_acc[h]);
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(lj_cnt_arr), lj_cnt_vec);

    for (int lane = 0; lane < TILE_SIZE; ++lane) {
        if (!((i_local8 >> lane) & 1u)) continue;
        const int a = ctx.tile_atoms[tbase + lane];
        fx_out[a] += fi_x[lane];
        fy_out[a] += fi_y[lane];
        fz_out[a] += fi_z[lane];
        if (ctx.do_lj) ctx.lj_count_out[a] = static_cast<int>(lj_cnt_arr[lane]);
    }
    local_pe += pe_tile;
    local_virial += vir_tile;
}
#endif  // __AVX2__ (mixed precision)

#if defined(__AVX512F__)
// AVX-512 per-tile kernel: walks tile T's edges 8 lanes (one home tile) at a
// time, delegating the vectorized pair physics to compute_lj_avx512()/
// compute_pme_avx512() and reducing to scalars only once, at the end of the
// tile (see the fix/fiy/fiz and pe_acc/vir_acc comments below). Caller
// (compute_tiled_nonbonded()) has already checked that T has local atoms.
inline void process_tile_avx512(int T, int tid, const TiledNonbondedTileCtx& ctx,
                                Real& local_pe, Real& local_virial) {
    constexpr int TILE_SIZE = kCpuTileSize;
    const size_t tbase = static_cast<size_t>(T) * TILE_SIZE;
    const uint8_t i_local8 = ctx.local_mask_of[T];
    // This thread's own private force slice -- see the lj_thread_fx_ comment
    // in force_calculator_cpu.h. No other thread ever touches these bytes.
    Real* __restrict__ fx_out = ctx.fx_out + static_cast<size_t>(tid) * ctx.thread_stride;
    Real* __restrict__ fy_out = ctx.fy_out + static_cast<size_t>(tid) * ctx.thread_stride;
    Real* __restrict__ fz_out = ctx.fz_out + static_cast<size_t>(tid) * ctx.thread_stride;

    const Real* __restrict__ ix = ctx.tpx + tbase;
    const Real* __restrict__ iy = ctx.tpy + tbase;
    const Real* __restrict__ iz = ctx.tpz + tbase;
    const Real* __restrict__ iq = ctx.do_coulomb ? (ctx.tpq + tbase) : nullptr;
    const int*  __restrict__ i_type_s = ctx.ttype + tbase;
    const int*  __restrict__ i_gid_s  = ctx.tgid + tbase;

    const __m512d xi = _mm512_loadu_pd(ix);
    const __m512d yi = _mm512_loadu_pd(iy);
    const __m512d zi = _mm512_loadu_pd(iz);
    const __m512d qi_vec = ctx.do_coulomb ? _mm512_loadu_pd(iq) : _mm512_setzero_pd();
    const __m256i type_i_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(i_type_s));
    const __m256i gid_i_vec  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(i_gid_s));
    const __mmask8 i_local = static_cast<__mmask8>(i_local8);
    __m512d fix = _mm512_setzero_pd(), fiy = _mm512_setzero_pd(), fiz = _mm512_setzero_pd();
    __m512d pe_acc = _mm512_setzero_pd(), vir_acc = _mm512_setzero_pd();
    // Per-lane pair count, accumulated with one masked vector add per
    // pair-mask instead of a per-pair ctz/while scan (see its extraction to
    // the scalar array after the edge loop below).
    __m512i lj_cnt_vec = _mm512_setzero_si512();
    const __m512i lj_cnt_one = _mm512_set1_epi64(1);

    for (int e = ctx.tile_offsets[T]; e < ctx.tile_offsets[T + 1]; ++e) {
        const std::uint64_t vmask_full = ctx.verlet_mask[e];
        if (vmask_full == 0) continue;
        const int O = ctx.tile_other[e];
        const bool same_tile = (O == T);
        const size_t obase = static_cast<size_t>(O) * TILE_SIZE;
        const uint8_t j_valid8 = ctx.valid_mask_of[O];
        if (j_valid8 == 0) continue;
        const uint8_t j_local8 = ctx.local_mask_of[O];
        const std::uint64_t ecol = ctx.excl_col[e];
        const std::uint64_t smask_full = same_tile ? ctx.same_tile_mask_of[T] : 0;
        // Periodic-image shift baked in at neighbor-list build time (see
        // neighbor_list.h) -- constant for every lane/slot in this edge, so
        // it's added once here instead of a per-pair minimum-image
        // compare/branch below.
        const Real edge_sx = ctx.tile_shift_x[e], edge_sy = ctx.tile_shift_y[e], edge_sz = ctx.tile_shift_z[e];

        const int*  __restrict__ j_atom_s = ctx.tile_atoms + obase;
        const Real* __restrict__ jx = ctx.tpx + obase;
        const Real* __restrict__ jy = ctx.tpy + obase;
        const Real* __restrict__ jz = ctx.tpz + obase;
        const Real* __restrict__ jq = ctx.do_coulomb ? (ctx.tpq + obase) : nullptr;
        const int*  __restrict__ j_type_s = ctx.ttype + obase;
        const int*  __restrict__ j_gid_s  = ctx.tgid + obase;

        for (int slot = 0; slot < TILE_SIZE; ++slot) {
            if (!((j_valid8 >> slot) & 1u)) continue;
            const bool slot_local = (j_local8 >> slot) & 1u;
            const int j_atom = j_atom_s[slot];

            __mmask8 base_mask = i_local & static_cast<__mmask8>(
                (vmask_full >> (slot * TILE_SIZE)) & 0xFFu);
            if (same_tile) {
                base_mask &= static_cast<__mmask8>((smask_full >> (slot * TILE_SIZE)) & 0xFFu);
            }
            if (!base_mask) continue;

            const __m512d dx = _mm512_sub_pd(xi, _mm512_set1_pd(jx[slot] + edge_sx));
            const __m512d dy = _mm512_sub_pd(yi, _mm512_set1_pd(jy[slot] + edge_sy));
            const __m512d dz = _mm512_sub_pd(zi, _mm512_set1_pd(jz[slot] + edge_sz));

            const __m512d r2 = _mm512_fmadd_pd(dx, dx,
                _mm512_fmadd_pd(dy, dy, _mm512_mul_pd(dz, dz)));
            __mmask8 cutoff_mask = base_mask;
            cutoff_mask &= _mm512_cmp_pd_mask(r2, ctx.vcutoff2, _CMP_LT_OQ);
            cutoff_mask &= _mm512_cmp_pd_mask(r2, ctx.vr2min, _CMP_GE_OQ);
            if (!cutoff_mask) continue;

            const unsigned excl_bits = static_cast<unsigned>((ecol >> (slot * TILE_SIZE)) & 0xFFu);

            // Energy/virial ownership: local j always co-owns (both ranks
            // would otherwise double count); remote j uses the gid
            // tie-break so exactly one rank credits the pair.
            __mmask8 owner_bits;
            if (slot_local) {
                owner_bits = static_cast<__mmask8>(0xFFu);
            } else {
                const __m256i gt = _mm256_cmpgt_epi32(_mm256_set1_epi32(j_gid_s[slot]), gid_i_vec);
                owner_bits = static_cast<__mmask8>(_mm256_movemask_ps(_mm256_castsi256_ps(gt)) & 0xFF);
            }

            __m512d force_mag = _mm512_setzero_pd();
            __m512d pe_pair = _mm512_setzero_pd();
            __mmask8 lj_mask = 0;
            if (ctx.do_lj) {
                lj_mask = compute_lj_avx512(r2, cutoff_mask, excl_bits,
                    type_i_vec, ctx.vlj_num_types, j_type_s[slot], ctx.lj_c6, ctx.lj_c12,
                    ctx.vone, ctx.v12, ctx.v6, ctx.cutoff_sq, ctx.lj_switching_enabled,
                    ctx.switch_sq, ctx.inv_denom, lj_cnt_vec, lj_cnt_one, force_mag, pe_pair);
            }

            __m512d force_mag_c = _mm512_setzero_pd();
            __m512d pe_c = _mm512_setzero_pd();
            if (ctx.do_coulomb) {
                compute_pme_avx512(r2, cutoff_mask, excl_bits, ctx.rs_table, ctx.rs_table_excl,
                    ctx.v_rs_r2_min, ctx.v_rs_inv_spacing, ctx.vk_coulomb, qi_vec, jq[slot],
                    force_mag_c, pe_c);
            }

            __m512d fxv, fyv, fzv;
            if (ctx.do_lj && ctx.do_coulomb) {
                fxv = _mm512_add_pd(_mm512_maskz_mul_pd(lj_mask, force_mag, dx), _mm512_maskz_mul_pd(cutoff_mask, force_mag_c, dx));
                fyv = _mm512_add_pd(_mm512_maskz_mul_pd(lj_mask, force_mag, dy), _mm512_maskz_mul_pd(cutoff_mask, force_mag_c, dy));
                fzv = _mm512_add_pd(_mm512_maskz_mul_pd(lj_mask, force_mag, dz), _mm512_maskz_mul_pd(cutoff_mask, force_mag_c, dz));
            } else if (ctx.do_lj) {
                fxv = _mm512_maskz_mul_pd(lj_mask, force_mag, dx);
                fyv = _mm512_maskz_mul_pd(lj_mask, force_mag, dy);
                fzv = _mm512_maskz_mul_pd(lj_mask, force_mag, dz);
            } else {
                fxv = _mm512_maskz_mul_pd(cutoff_mask, force_mag_c, dx);
                fyv = _mm512_maskz_mul_pd(cutoff_mask, force_mag_c, dy);
                fzv = _mm512_maskz_mul_pd(cutoff_mask, force_mag_c, dz);
            }

            // Home i forces stay in vector registers -- no reduction until
            // this tile's entire neighbour list is done (see fix/fiy/fiz
            // below the edge loop).
            fix = _mm512_add_pd(fix, fxv);
            fiy = _mm512_add_pd(fiy, fyv);
            fiz = _mm512_add_pd(fiz, fzv);

            if (ctx.do_lj) {
                const __mmask8 energy_mask_lj = lj_mask & owner_bits;
                pe_acc = _mm512_add_pd(pe_acc, _mm512_maskz_mov_pd(energy_mask_lj, pe_pair));
                vir_acc = _mm512_add_pd(vir_acc, _mm512_maskz_mov_pd(energy_mask_lj, _mm512_mul_pd(r2, force_mag)));
            }
            if (ctx.do_coulomb) {
                const __mmask8 energy_mask_c = cutoff_mask & owner_bits;
                pe_acc = _mm512_add_pd(pe_acc, _mm512_maskz_mov_pd(energy_mask_c, pe_c));
                vir_acc = _mm512_add_pd(vir_acc, _mm512_maskz_mov_pd(energy_mask_c, _mm512_mul_pd(r2, force_mag_c)));
            }

            // j reaction (Newton's 3rd law): the only thing this kernel ever
            // horizontally reduces, and it already combines the LJ +
            // Coulomb contribution to this pair.
            if (slot_local) {
                const Real jfx = -_mm512_reduce_add_pd(fxv);
                const Real jfy = -_mm512_reduce_add_pd(fyv);
                const Real jfz = -_mm512_reduce_add_pd(fzv);
                commit_reaction_force(j_atom, jfx, jfy, jfz, fx_out, fy_out, fz_out);
            }
        }
    }

    Real fi_x[TILE_SIZE], fi_y[TILE_SIZE], fi_z[TILE_SIZE];
    _mm512_storeu_pd(fi_x, fix);
    _mm512_storeu_pd(fi_y, fiy);
    _mm512_storeu_pd(fi_z, fiz);
    const Real pe_tile = _mm512_reduce_add_pd(pe_acc);
    const Real vir_tile = _mm512_reduce_add_pd(vir_acc);
    // lj_cnt_vec accumulated one masked vector add per pair-mask instead of
    // a per-pair ctz/while scan (see its declaration above the edge loop) --
    // extracted to the scalar array once per tile, not once per pair.
    int64_t lj_cnt_arr[TILE_SIZE];
    _mm512_storeu_si512(reinterpret_cast<__m512i*>(lj_cnt_arr), lj_cnt_vec);

    // Home force: ONE scatter per tile, into this thread's own private
    // slice -- never races another thread regardless of which tiles it
    // touches (see the lj_thread_fx_ comment in force_calculator_cpu.h).
    for (int lane = 0; lane < TILE_SIZE; ++lane) {
        if (!((i_local8 >> lane) & 1u)) continue;
        const int a = ctx.tile_atoms[tbase + lane];
        fx_out[a] += fi_x[lane];
        fy_out[a] += fi_y[lane];
        fz_out[a] += fi_z[lane];
        if (ctx.do_lj) ctx.lj_count_out[a] = static_cast<int>(lj_cnt_arr[lane]);
    }
    local_pe += pe_tile;
    local_virial += vir_tile;
}
#elif defined(__AVX2__)
// AVX2 per-tile kernel: same per-thread-private-buffer design as
// process_tile_avx512(), but with no 8-wide double vector available, the
// tile's 8 home (i) lanes are split into two 4-wide __m256d halves (h=0:
// lanes 0-3, h=1: lanes 4-7), each fed through compute_lj_avx2()/
// compute_pme_avx2() independently. A pair-edge's j-reaction is the sum of
// both halves' contributions, so it's accumulated across both before the
// single commit_reaction_force() call per (edge, slot) -- unlike
// process_tile_avx512(), which reduces one 8-wide register directly. Caller
// (compute_tiled_nonbonded()) has already checked that T has local atoms.
inline void process_tile_avx2(int T, int tid, const TiledNonbondedTileCtx& ctx,
                              Real& local_pe, Real& local_virial) {
    constexpr int TILE_SIZE = kCpuTileSize;  // 8, processed as two 4-wide halves
    const size_t tbase = static_cast<size_t>(T) * TILE_SIZE;
    const uint8_t i_local8 = ctx.local_mask_of[T];
    // This thread's own private force slice -- see the lj_thread_fx_ comment
    // in force_calculator_cpu.h. No other thread ever touches these bytes.
    Real* __restrict__ fx_out = ctx.fx_out + static_cast<size_t>(tid) * ctx.thread_stride;
    Real* __restrict__ fy_out = ctx.fy_out + static_cast<size_t>(tid) * ctx.thread_stride;
    Real* __restrict__ fz_out = ctx.fz_out + static_cast<size_t>(tid) * ctx.thread_stride;

    const Real* __restrict__ ix = ctx.tpx + tbase;
    const Real* __restrict__ iy = ctx.tpy + tbase;
    const Real* __restrict__ iz = ctx.tpz + tbase;
    const Real* __restrict__ iq = ctx.do_coulomb ? (ctx.tpq + tbase) : nullptr;
    const int*  __restrict__ i_type_s = ctx.ttype + tbase;
    const int*  __restrict__ i_gid_s  = ctx.tgid + tbase;

    __m256d xi[2], yi[2], zi[2], qi_vec[2];
    __m128i type_i_vec[2], gid_i_vec[2];
    __m256d fix[2], fiy[2], fiz[2], pe_acc[2], vir_acc[2];
    __m256i lj_cnt_vec[2];
    const __m256i lj_cnt_one = _mm256_set1_epi64x(1);
    unsigned i_local4[2];
    for (int h = 0; h < 2; ++h) {
        const size_t off = static_cast<size_t>(h) * 4;
        xi[h] = _mm256_loadu_pd(ix + off);
        yi[h] = _mm256_loadu_pd(iy + off);
        zi[h] = _mm256_loadu_pd(iz + off);
        qi_vec[h] = ctx.do_coulomb ? _mm256_loadu_pd(iq + off) : _mm256_setzero_pd();
        type_i_vec[h] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(i_type_s + off));
        gid_i_vec[h]  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(i_gid_s + off));
        fix[h] = fiy[h] = fiz[h] = _mm256_setzero_pd();
        pe_acc[h] = vir_acc[h] = _mm256_setzero_pd();
        lj_cnt_vec[h] = _mm256_setzero_si256();
        i_local4[h] = (static_cast<unsigned>(i_local8) >> (h * 4)) & 0x0Fu;
    }

    for (int e = ctx.tile_offsets[T]; e < ctx.tile_offsets[T + 1]; ++e) {
        const std::uint64_t vmask_full = ctx.verlet_mask[e];
        if (vmask_full == 0) continue;
        const int O = ctx.tile_other[e];
        const bool same_tile = (O == T);
        const size_t obase = static_cast<size_t>(O) * TILE_SIZE;
        const uint8_t j_valid8 = ctx.valid_mask_of[O];
        if (j_valid8 == 0) continue;
        const uint8_t j_local8 = ctx.local_mask_of[O];
        const std::uint64_t ecol = ctx.excl_col[e];
        const std::uint64_t smask_full = same_tile ? ctx.same_tile_mask_of[T] : 0;
        const Real edge_sx = ctx.tile_shift_x[e], edge_sy = ctx.tile_shift_y[e], edge_sz = ctx.tile_shift_z[e];

        const int*  __restrict__ j_atom_s = ctx.tile_atoms + obase;
        const Real* __restrict__ jx = ctx.tpx + obase;
        const Real* __restrict__ jy = ctx.tpy + obase;
        const Real* __restrict__ jz = ctx.tpz + obase;
        const Real* __restrict__ jq = ctx.do_coulomb ? (ctx.tpq + obase) : nullptr;
        const int*  __restrict__ j_type_s = ctx.ttype + obase;
        const int*  __restrict__ j_gid_s  = ctx.tgid + obase;

        for (int slot = 0; slot < TILE_SIZE; ++slot) {
            if (!((j_valid8 >> slot) & 1u)) continue;
            const bool slot_local = (j_local8 >> slot) & 1u;
            const int j_atom = j_atom_s[slot];

            const unsigned vbits8 = static_cast<unsigned>((vmask_full >> (slot * TILE_SIZE)) & 0xFFu);
            const unsigned sbits8 = same_tile
                ? static_cast<unsigned>((smask_full >> (slot * TILE_SIZE)) & 0xFFu)
                : 0xFFu;
            const unsigned ebits8 = static_cast<unsigned>((ecol >> (slot * TILE_SIZE)) & 0xFFu);

            const __m256d xj = _mm256_set1_pd(jx[slot] + edge_sx);
            const __m256d yj = _mm256_set1_pd(jy[slot] + edge_sy);
            const __m256d zj = _mm256_set1_pd(jz[slot] + edge_sz);
            const int type_j = j_type_s[slot];
            const int gid_j = j_gid_s[slot];
            const Real qj = ctx.do_coulomb ? jq[slot] : 0.0;

            Real jfx = 0.0, jfy = 0.0, jfz = 0.0;

            for (int h = 0; h < 2; ++h) {
                const unsigned base4 = i_local4[h] & (vbits8 >> (h * 4)) & (sbits8 >> (h * 4)) & 0x0Fu;
                if (base4 == 0) continue;
                const __m256d base_mask = mask4_from_bits(base4);

                const __m256d dx = _mm256_sub_pd(xi[h], xj);
                const __m256d dy = _mm256_sub_pd(yi[h], yj);
                const __m256d dz = _mm256_sub_pd(zi[h], zj);
                const __m256d r2 = _mm256_fmadd_pd(dx, dx,
                    _mm256_fmadd_pd(dy, dy, _mm256_mul_pd(dz, dz)));

                const __m256d cutoff_mask = _mm256_and_pd(base_mask,
                    _mm256_and_pd(_mm256_cmp_pd(r2, ctx.vcutoff2, _CMP_LT_OQ),
                                  _mm256_cmp_pd(r2, ctx.vr2min, _CMP_GE_OQ)));
                if (_mm256_movemask_pd(cutoff_mask) == 0) continue;

                const unsigned excl4 = (ebits8 >> (h * 4)) & 0x0Fu;
                const __m256d excl_mask = mask4_from_bits(excl4);

                __m256d owner_bits;
                if (slot_local) {
                    owner_bits = _mm256_castsi256_pd(_mm256_set1_epi64x(-1LL));
                } else {
                    const __m128i gt = _mm_cmpgt_epi32(_mm_set1_epi32(gid_j), gid_i_vec[h]);
                    owner_bits = _mm256_castsi256_pd(_mm256_cvtepi32_epi64(gt));
                }

                __m256d force_mag = _mm256_setzero_pd(), pe_pair = _mm256_setzero_pd();
                __m256d lj_mask = _mm256_setzero_pd();
                if (ctx.do_lj) {
                    lj_mask = compute_lj_avx2(r2, cutoff_mask, excl_mask,
                        type_i_vec[h], ctx.vlj_num_types, type_j, ctx.lj_c6, ctx.lj_c12,
                        ctx.vone, ctx.v12, ctx.v6, ctx.cutoff_sq, ctx.lj_switching_enabled,
                        ctx.switch_sq, ctx.inv_denom, lj_cnt_vec[h], lj_cnt_one, force_mag, pe_pair);
                }

                __m256d force_mag_c = _mm256_setzero_pd(), pe_c = _mm256_setzero_pd();
                if (ctx.do_coulomb) {
                    compute_pme_avx2(r2, cutoff_mask, excl_mask, ctx.rs_table, ctx.rs_table_excl,
                        ctx.v_rs_r2_min, ctx.v_rs_inv_spacing, ctx.vk_coulomb, qi_vec[h], qj,
                        force_mag_c, pe_c);
                }

                __m256d fxv, fyv, fzv;
                if (ctx.do_lj && ctx.do_coulomb) {
                    fxv = _mm256_add_pd(_mm256_and_pd(lj_mask, _mm256_mul_pd(force_mag, dx)),
                                        _mm256_and_pd(cutoff_mask, _mm256_mul_pd(force_mag_c, dx)));
                    fyv = _mm256_add_pd(_mm256_and_pd(lj_mask, _mm256_mul_pd(force_mag, dy)),
                                        _mm256_and_pd(cutoff_mask, _mm256_mul_pd(force_mag_c, dy)));
                    fzv = _mm256_add_pd(_mm256_and_pd(lj_mask, _mm256_mul_pd(force_mag, dz)),
                                        _mm256_and_pd(cutoff_mask, _mm256_mul_pd(force_mag_c, dz)));
                } else if (ctx.do_lj) {
                    fxv = _mm256_and_pd(lj_mask, _mm256_mul_pd(force_mag, dx));
                    fyv = _mm256_and_pd(lj_mask, _mm256_mul_pd(force_mag, dy));
                    fzv = _mm256_and_pd(lj_mask, _mm256_mul_pd(force_mag, dz));
                } else {
                    fxv = _mm256_and_pd(cutoff_mask, _mm256_mul_pd(force_mag_c, dx));
                    fyv = _mm256_and_pd(cutoff_mask, _mm256_mul_pd(force_mag_c, dy));
                    fzv = _mm256_and_pd(cutoff_mask, _mm256_mul_pd(force_mag_c, dz));
                }

                fix[h] = _mm256_add_pd(fix[h], fxv);
                fiy[h] = _mm256_add_pd(fiy[h], fyv);
                fiz[h] = _mm256_add_pd(fiz[h], fzv);

                if (ctx.do_lj) {
                    const __m256d energy_mask_lj = _mm256_and_pd(lj_mask, owner_bits);
                    pe_acc[h] = _mm256_add_pd(pe_acc[h], _mm256_and_pd(energy_mask_lj, pe_pair));
                    vir_acc[h] = _mm256_add_pd(vir_acc[h],
                        _mm256_and_pd(energy_mask_lj, _mm256_mul_pd(r2, force_mag)));
                }
                if (ctx.do_coulomb) {
                    const __m256d energy_mask_c = _mm256_and_pd(cutoff_mask, owner_bits);
                    pe_acc[h] = _mm256_add_pd(pe_acc[h], _mm256_and_pd(energy_mask_c, pe_c));
                    vir_acc[h] = _mm256_add_pd(vir_acc[h],
                        _mm256_and_pd(energy_mask_c, _mm256_mul_pd(r2, force_mag_c)));
                }

                // j reaction (Newton's 3rd law): unlike process_tile_avx512()'s
                // single 8-wide reduce, this pair's reaction is split across
                // both halves' 4-wide registers -- summed into jfx/jfy/jfz here
                // and committed once after both halves are done (see below).
                if (slot_local) {
                    jfx -= hsum256_pd(fxv);
                    jfy -= hsum256_pd(fyv);
                    jfz -= hsum256_pd(fzv);
                }
            }

            if (slot_local) {
                commit_reaction_force(j_atom, jfx, jfy, jfz, fx_out, fy_out, fz_out);
            }
        }
    }

    Real fi_x[TILE_SIZE], fi_y[TILE_SIZE], fi_z[TILE_SIZE];
    Real pe_tile = 0.0, vir_tile = 0.0;
    int64_t lj_cnt_arr[TILE_SIZE];
    for (int h = 0; h < 2; ++h) {
        _mm256_storeu_pd(fi_x + h * 4, fix[h]);
        _mm256_storeu_pd(fi_y + h * 4, fiy[h]);
        _mm256_storeu_pd(fi_z + h * 4, fiz[h]);
        pe_tile += hsum256_pd(pe_acc[h]);
        vir_tile += hsum256_pd(vir_acc[h]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(lj_cnt_arr + h * 4), lj_cnt_vec[h]);
    }

    // Home force: ONE scatter per tile, into this thread's own private
    // slice -- same no-race guarantee as process_tile_avx512() (see its
    // comment).
    for (int lane = 0; lane < TILE_SIZE; ++lane) {
        if (!((i_local8 >> lane) & 1u)) continue;
        const int a = ctx.tile_atoms[tbase + lane];
        fx_out[a] += fi_x[lane];
        fy_out[a] += fi_y[lane];
        fz_out[a] += fi_z[lane];
        if (ctx.do_lj) ctx.lj_count_out[a] = static_cast<int>(lj_cnt_arr[lane]);
    }
    local_pe += pe_tile;
    local_virial += vir_tile;
}
#else
// Portable scalar per-tile kernel (no AVX-512/AVX2): same per-thread-private-
// buffer design as process_tile_avx512(), but LJ and Coulomb are
// evaluated lane-by-lane via the shared compute_lj_pair()/
// compute_coulomb_pair_real_space() helpers (the same ones
// compute_graph_nonbonded() uses) and summed per pair before the (optional)
// Newton reaction. Caller (compute_tiled_nonbonded()) has already checked
// that T has local atoms.
inline void process_tile_scalar(int T, int tid, const TiledNonbondedTileCtx& ctx,
                                Real& local_pe, Real& local_virial) {
    constexpr int TILE_SIZE = kCpuTileSize;
    const size_t tbase = static_cast<size_t>(T) * TILE_SIZE;
    const uint8_t i_local8 = ctx.local_mask_of[T];
    // This thread's own private force slice -- see the lj_thread_fx_ comment
    // in force_calculator_cpu.h. No other thread ever touches these bytes.
    Real* __restrict__ fx_out = ctx.fx_out + static_cast<size_t>(tid) * ctx.thread_stride;
    Real* __restrict__ fy_out = ctx.fy_out + static_cast<size_t>(tid) * ctx.thread_stride;
    Real* __restrict__ fz_out = ctx.fz_out + static_cast<size_t>(tid) * ctx.thread_stride;

    const Real* __restrict__ ix = ctx.tpx + tbase;
    const Real* __restrict__ iy = ctx.tpy + tbase;
    const Real* __restrict__ iz = ctx.tpz + tbase;
    const Real* __restrict__ iq = ctx.do_coulomb ? (ctx.tpq + tbase) : nullptr;
    const int*  __restrict__ i_type_s = ctx.ttype + tbase;
    const int*  __restrict__ i_gid_s  = ctx.tgid + tbase;

    Real fi_x[TILE_SIZE], fi_y[TILE_SIZE], fi_z[TILE_SIZE];
    int lj_cnt[TILE_SIZE];
    for (int lane = 0; lane < TILE_SIZE; ++lane) { fi_x[lane] = fi_y[lane] = fi_z[lane] = 0.0; lj_cnt[lane] = 0; }
    Real pe_tile = 0.0, vir_tile = 0.0;

    for (int e = ctx.tile_offsets[T]; e < ctx.tile_offsets[T + 1]; ++e) {
        const std::uint64_t vmask_full = ctx.verlet_mask[e];
        if (vmask_full == 0) continue;
        const int O = ctx.tile_other[e];
        const bool same_tile = (O == T);
        const size_t obase = static_cast<size_t>(O) * TILE_SIZE;
        const uint8_t j_valid8 = ctx.valid_mask_of[O];
        if (j_valid8 == 0) continue;
        const uint8_t j_local8 = ctx.local_mask_of[O];
        const std::uint64_t ecol = ctx.excl_col[e];
        const std::uint64_t smask_full = same_tile ? ctx.same_tile_mask_of[T] : 0;
        const Real edge_sx = ctx.tile_shift_x[e], edge_sy = ctx.tile_shift_y[e], edge_sz = ctx.tile_shift_z[e];

        const int*  __restrict__ j_atom_s = ctx.tile_atoms + obase;
        const Real* __restrict__ jx = ctx.tpx + obase;
        const Real* __restrict__ jy = ctx.tpy + obase;
        const Real* __restrict__ jz = ctx.tpz + obase;
        const Real* __restrict__ jq = ctx.do_coulomb ? (ctx.tpq + obase) : nullptr;
        const int*  __restrict__ j_type_s = ctx.ttype + obase;
        const int*  __restrict__ j_gid_s  = ctx.tgid + obase;

        for (int slot = 0; slot < TILE_SIZE; ++slot) {
            if (!((j_valid8 >> slot) & 1u)) continue;
            const bool slot_local = (j_local8 >> slot) & 1u;
            const int j_atom = j_atom_s[slot];

            uint8_t mask8 = i_local8 & static_cast<uint8_t>(
                (vmask_full >> (slot * TILE_SIZE)) & 0xFFu);
            if (same_tile) mask8 &= static_cast<uint8_t>((smask_full >> (slot * TILE_SIZE)) & 0xFFu);
            if (!mask8) continue;

            const Real xj = jx[slot] + edge_sx, yj = jy[slot] + edge_sy, zj = jz[slot] + edge_sz;
            const int type_j = ctx.do_lj ? j_type_s[slot] : 0;
            const int gid_j = j_gid_s[slot];
            const Real qj = ctx.do_coulomb ? jq[slot] : 0.0;
            const unsigned excl_bits = static_cast<unsigned>((ecol >> (slot * TILE_SIZE)) & 0xFFu);
            Real jfx = 0.0, jfy = 0.0, jfz = 0.0;

            for (int lane = 0; lane < TILE_SIZE; ++lane) {
                if (!((mask8 >> lane) & 1u)) continue;
                const Real dx = ix[lane] - xj;
                const Real dy = iy[lane] - yj;
                const Real dz = iz[lane] - zj;
                const Real r2 = dx * dx + dy * dy + dz * dz;
                if (r2 >= ctx.cutoff_sq || r2 < 1e-10) continue;

                const bool excluded = (excl_bits >> lane) & 1u;
                const bool owner = slot_local || (i_gid_s[lane] < gid_j);
                Real fxp = 0.0, fyp = 0.0, fzp = 0.0;

                if (ctx.do_lj && !excluded) {
                    ++lj_cnt[lane];
                    const size_t tij = static_cast<size_t>(i_type_s[lane]) *
                        static_cast<size_t>(ctx.lj_num_types) + static_cast<size_t>(type_j);
                    Real force_mag, pe_pair;
                    compute_lj_pair(r2, ctx.lj_c6[tij], ctx.lj_c12[tij], ctx.cutoff_sq,
                                    ctx.lj_switching_enabled, ctx.switch_sq, ctx.inv_denom, force_mag, pe_pair);
                    accumulate_pair_contribution(force_mag, pe_pair, dx, dy, dz, r2, owner,
                                                 fxp, fyp, fzp, pe_tile, vir_tile);
                }

                if (ctx.do_coulomb) {
                    Real force_mag_c, pe_c;
                    compute_coulomb_pair_real_space(excluded ? ctx.rs_table_excl : ctx.rs_table,
                                                    ctx.rs_r2_min, ctx.rs_inv_spacing, r2,
                                                    iq[lane], qj, ctx.k_coulomb, force_mag_c, pe_c);
                    accumulate_pair_contribution(force_mag_c, pe_c, dx, dy, dz, r2, owner,
                                                 fxp, fyp, fzp, pe_tile, vir_tile);
                }

                fi_x[lane] += fxp; fi_y[lane] += fyp; fi_z[lane] += fzp;
                if (slot_local) { jfx -= fxp; jfy -= fyp; jfz -= fzp; }
            }

            if (slot_local) {
                commit_reaction_force(j_atom, jfx, jfy, jfz, fx_out, fy_out, fz_out);
            }
        }
    }

    for (int lane = 0; lane < TILE_SIZE; ++lane) {
        if (!((i_local8 >> lane) & 1u)) continue;
        const int a = ctx.tile_atoms[tbase + lane];
        fx_out[a] += fi_x[lane];
        fy_out[a] += fi_y[lane];
        fz_out[a] += fi_z[lane];
        if (ctx.do_lj) ctx.lj_count_out[a] = lj_cnt[lane];
    }
    local_pe += pe_tile;
    local_virial += vir_tile;
}
#endif  // __AVX512F__ / __AVX2__

void ForceCalculatorCPU::compute_tiled_nonbonded(MDSystem& system, bool do_lj, bool do_coulomb) {
    const char* label = do_lj ? (do_coulomb ? "LJ+PME" : "LJ") : "PME";
    run_nonbonded_with_pme_overlap(system, do_coulomb, label, [&] {
    ParticleData& particles = system.get_particles();
    NeighborList& nlist_base = system.get_neighbor_list();
    // CPU_SPATIAL_VERLET is the only NeighborList backend (see
    // neighbor_list.h); the tile-graph accessors are specific to it, the
    // same way the CUDA tile graph lives entirely outside the NeighborList
    // interface.
    auto& nlist = static_cast<CpuSpatialVerletList&>(nlist_base);
    ForceFieldParams& params = system.get_ff_params();
    Topology& topology = system.get_topology();
    ensure_exclusion_cache_built(topology);

    int lj_num_types = 0;
    const Real* __restrict__ lj_c6 = nullptr;
    const Real* __restrict__ lj_c12 = nullptr;
    bool lj_switching_enabled = false;
    Real switch_sq = 0.0, inv_denom = 0.0;
    if (do_lj) {
        ensure_lj_coeff_table_built(params);
        lj_num_types = lj_coeff_num_types_;
        ensure_lj_split_cache_d(lj_c6c12_, lj_num_types, g_lj_split_d);
        lj_c6 = g_lj_split_d.c6.data();
        lj_c12 = g_lj_split_d.c12.data();
        lj_switching_enabled = (params.lj_switch_distance > 0.0);
    }

    const Real* __restrict__ rs_table = nullptr;
    const Real* __restrict__ rs_table_excl = nullptr;
    Real rs_r2_min = 0.0, rs_inv_spacing = 0.0;
    const Real k_coulomb = 332.0636;
    if (do_coulomb) {
        rs_table = ewald_rs_table_.data();
        rs_table_excl = ewald_rs_table_excl_.data();
        rs_r2_min = ewald_rs_table_r2_min_;
        rs_inv_spacing = ewald_rs_table_inv_spacing_;
    }

    // force_calculation.precision = mixed (see set_mixed_precision()):
    // builds/refreshes the float mirror tables (no-op once already current)
    // and, if do_lj, the float LJ c6/c12 split cache. The packed-SoA float
    // position/charge mirrors (tile_fx_/fy_/fz_/fq_) are produced by
    // repack_tile_soa() below whenever mixed_precision_nonbonded_ is set,
    // which it already is by this point.
    ensure_mixed_precision_tables();
    const bool use_mixed_precision = mixed_precision_nonbonded_;
    const float* __restrict__ lj_c6_f = nullptr;
    const float* __restrict__ lj_c12_f = nullptr;
    if (use_mixed_precision && do_lj) {
        ensure_lj_split_cache_f(lj_c6c12_f_, lj_num_types, g_lj_split_f);
        lj_c6_f = g_lj_split_f.c6.data();
        lj_c12_f = g_lj_split_f.c12.data();
    }
    const float* __restrict__ rs_table_f =
        (use_mixed_precision && do_coulomb) ? ewald_rs_table_f_.data() : nullptr;
    const float* __restrict__ rs_table_excl_f =
        (use_mixed_precision && do_coulomb) ? ewald_rs_table_excl_f_.data() : nullptr;

    size_t local_particles = system.get_local_particles();

    // Use physical cutoff, not neighbor list cutoff (which includes skin)
    const Real physical_cutoff = system.get_physical_cutoff();
    const Real cutoff_sq = physical_cutoff * physical_cutoff;

    if (do_lj) {
        if (lj_switching_enabled) {
            switch_sq = params.lj_switch_distance * params.lj_switch_distance;
            const Real cutoff_delta = cutoff_sq - switch_sq;
            inv_denom = 1.0 / (cutoff_delta * cutoff_delta * cutoff_delta);
        }
    }

    Real local_pe = 0.0;
    Real local_virial = 0.0;

    // Every owned atom occurs in exactly one home-tile lane, and every such
    // lane overwrites its count below (when do_lj). Avoid an otherwise
    // redundant O(N) zero-fill on every step.
    if (lj_pair_counts_.size() != local_particles) {
        lj_pair_counts_.resize(local_particles);
    }

    const int* __restrict__ tile_atoms = nlist.get_tile_atoms();
    const int* __restrict__ tile_offsets = nlist.get_tile_pair_offsets();
    const int* __restrict__ tile_other = nlist.get_tile_pair_other();
    const std::uint64_t* __restrict__ verlet_mask = nlist.get_tile_pair_verlet_mask();
    const uint8_t* __restrict__ tile_has_local = nlist.get_tile_has_local();
    // Rebuild-time periodic-image shift per tile-pair edge (see its comment
    // in neighbor_list.h): folded straight into the j broadcast below
    // instead of re-deriving minimum-image with a per-pair compare/branch.
    const Real* __restrict__ tile_shift_x = nlist.get_tile_pair_shift_x();
    const Real* __restrict__ tile_shift_y = nlist.get_tile_pair_shift_y();
    const Real* __restrict__ tile_shift_z = nlist.get_tile_pair_shift_z();

    // Cluster-packed tile SoA (rebuilt every step) + the per-rebuild caches:
    // lane-major exclusion bits, its slot-major transpose, and per-tile
    // valid/local/same-tile masks. See each cache's member comment in
    // force_calculator_cpu.h.
    repack_tile_soa(particles, nlist);
    ensure_tile_exclusion_masks(nlist);
    ensure_lj_tile_excl_col(nlist);
    ensure_lj_tile_masks(nlist, local_particles);
    const Real* __restrict__ tpx = tile_px_.data();
    const Real* __restrict__ tpy = tile_py_.data();
    const Real* __restrict__ tpz = tile_pz_.data();
    const Real* __restrict__ tpq = do_coulomb ? tile_q_.data() : nullptr;
    const int*  __restrict__ ttype = tile_type_.data();
    const int*  __restrict__ tgid  = tile_gid_.data();
    const std::uint64_t* __restrict__ excl_col = lj_tile_excl_col_.data();
    const std::uint64_t* __restrict__ same_tile_mask_of = lj_same_tile_mask_.data();
    const uint8_t* __restrict__ valid_mask_of = lj_tile_valid_mask_.data();
    const uint8_t* __restrict__ local_mask_of = lj_tile_local_mask_.data();

    // The REAL, final destination arrays -- only ever touched by the merge
    // pass at the very end, after every thread's private slice is complete.
    Real* __restrict__ real_fx = particles.fx.data();
    Real* __restrict__ real_fy = particles.fy.data();
    Real* __restrict__ real_fz = particles.fz.data();
    int*  __restrict__ lj_count_out = lj_pair_counts_.data();

#ifdef _OPENMP
    const int nthreads = omp_get_max_threads();
#else
    const int nthreads = 1;
#endif
    // Per-thread private force accumulation -- see the lj_thread_fx_ comment
    // in force_calculator_cpu.h. Every thread writes ONLY its own
    // [tid*local_particles, (tid+1)*local_particles) slice, so tiles need no
    // ownership assignment: any tile can go to any thread.
    ensure_lj_thread_buffers(nthreads, local_particles);

    TiledNonbondedTileCtx ctx;
    ctx.do_lj = do_lj;
    ctx.do_coulomb = do_coulomb;
    ctx.lj_num_types = lj_num_types;
    ctx.lj_c6 = lj_c6;
    ctx.lj_c12 = lj_c12;
    ctx.lj_switching_enabled = lj_switching_enabled;
    ctx.switch_sq = switch_sq;
    ctx.inv_denom = inv_denom;
    ctx.cutoff_sq = cutoff_sq;
    ctx.rs_table = rs_table;
    ctx.rs_table_excl = rs_table_excl;
    ctx.rs_r2_min = rs_r2_min;
    ctx.rs_inv_spacing = rs_inv_spacing;
    ctx.k_coulomb = k_coulomb;
    ctx.tile_atoms = tile_atoms;
    ctx.tile_offsets = tile_offsets;
    ctx.tile_other = tile_other;
    ctx.verlet_mask = verlet_mask;
    ctx.tile_shift_x = tile_shift_x;
    ctx.tile_shift_y = tile_shift_y;
    ctx.tile_shift_z = tile_shift_z;
    ctx.tpx = tpx;
    ctx.tpy = tpy;
    ctx.tpz = tpz;
    ctx.tpq = tpq;
    ctx.ttype = ttype;
    ctx.tgid = tgid;
    ctx.excl_col = excl_col;
    ctx.same_tile_mask_of = same_tile_mask_of;
    ctx.valid_mask_of = valid_mask_of;
    ctx.local_mask_of = local_mask_of;
    // Base of the private per-thread buffers; each process_tile_* callee
    // offsets by its own tid*thread_stride (see their bodies).
    ctx.fx_out = lj_thread_fx_.data();
    ctx.fy_out = lj_thread_fy_.data();
    ctx.fz_out = lj_thread_fz_.data();
    ctx.thread_stride = local_particles;
    ctx.lj_count_out = lj_count_out;
    #if defined(__AVX512F__)
        ctx.vcutoff2 = _mm512_set1_pd(cutoff_sq);
        ctx.vr2min = _mm512_set1_pd(1e-10);
        ctx.vone = _mm512_set1_pd(1.0);
        ctx.v12 = _mm512_set1_pd(12.0);
        ctx.v6 = _mm512_set1_pd(6.0);
        ctx.vlj_num_types = _mm256_set1_epi32(lj_num_types);
        ctx.vk_coulomb = _mm512_set1_pd(k_coulomb);
        ctx.v_rs_r2_min = _mm512_set1_pd(rs_r2_min);
        ctx.v_rs_inv_spacing = _mm512_set1_pd(rs_inv_spacing);
    #elif defined(__AVX2__)
        ctx.vcutoff2 = _mm256_set1_pd(cutoff_sq);
        ctx.vr2min = _mm256_set1_pd(1e-10);
        ctx.vone = _mm256_set1_pd(1.0);
        ctx.v12 = _mm256_set1_pd(12.0);
        ctx.v6 = _mm256_set1_pd(6.0);
        ctx.vlj_num_types = _mm_set1_epi32(lj_num_types);
        ctx.vk_coulomb = _mm256_set1_pd(k_coulomb);
        ctx.v_rs_r2_min = _mm256_set1_pd(rs_r2_min);
        ctx.v_rs_inv_spacing = _mm256_set1_pd(rs_inv_spacing);
    #endif

    ctx.use_mixed_precision = use_mixed_precision;
    if (use_mixed_precision) {
        ctx.lj_c6_f = lj_c6_f;
        ctx.lj_c12_f = lj_c12_f;
        ctx.rs_table_f = rs_table_f;
        ctx.rs_table_excl_f = rs_table_excl_f;
        ctx.rs_r2_min_f = static_cast<float>(rs_r2_min);
        ctx.rs_inv_spacing_f = static_cast<float>(rs_inv_spacing);
        ctx.k_coulomb_f = static_cast<float>(k_coulomb);
        ctx.cutoff_sq_f = static_cast<float>(cutoff_sq);
        ctx.switch_sq_f = static_cast<float>(switch_sq);
        ctx.inv_denom_f = static_cast<float>(inv_denom);
        ctx.tpx_f = tile_fx_.data();
        ctx.tpy_f = tile_fy_.data();
        ctx.tpz_f = tile_fz_.data();
        ctx.tpq_f = do_coulomb ? tile_fq_.data() : nullptr;
    #if defined(__AVX2__)
        ctx.vlj_num_types_f = _mm256_set1_epi32(lj_num_types);
        ctx.vcutoff2_f = _mm256_set1_ps(ctx.cutoff_sq_f);
        ctx.vr2min_f = _mm256_set1_ps(1e-10f);
        ctx.vone_f = _mm256_set1_ps(1.0f);
        ctx.v12_f = _mm256_set1_ps(12.0f);
        ctx.v6_f = _mm256_set1_ps(6.0f);
        ctx.vk_coulomb_f = _mm256_set1_ps(ctx.k_coulomb_f);
        ctx.v_rs_r2_min_f = _mm256_set1_ps(ctx.rs_r2_min_f);
        ctx.v_rs_inv_spacing_f = _mm256_set1_ps(ctx.rs_inv_spacing_f);
    #endif
    }

    const int num_tiles = nlist.num_tiles();

    #pragma omp parallel reduction(+:local_pe, local_virial)
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        // Zero this thread's own private slice -- touches only memory this
        // thread will exclusively write to below, so this can never be a
        // source of false sharing either (each thread's slice is a
        // contiguous, disjoint range of lj_thread_fx_/fy_/fz_).
        Real* __restrict__ my_fx = ctx.fx_out + static_cast<size_t>(tid) * ctx.thread_stride;
        Real* __restrict__ my_fy = ctx.fy_out + static_cast<size_t>(tid) * ctx.thread_stride;
        Real* __restrict__ my_fz = ctx.fz_out + static_cast<size_t>(tid) * ctx.thread_stride;
        std::fill(my_fx, my_fx + local_particles, 0.0);
        std::fill(my_fy, my_fy + local_particles, 0.0);
        std::fill(my_fz, my_fz + local_particles, 0.0);

        // No tile ownership needed any more: every write process_tile_*
        // makes lands in THIS thread's own private slice regardless of
        // which tile T it is handed, so a plain dynamic work-sharing loop
        // is both race-free and load-balanced (unlike the old static
        // edge-count partition, which could not react to actual per-tile
        // cutoff-cull variance).
        #pragma omp for schedule(dynamic, 64)
        for (int T = 0; T < num_tiles; ++T) {
            if (tile_has_local && tile_has_local[T] == 0) continue;
            if (local_mask_of[T] == 0) continue;
            // Mixed precision is a RUNTIME choice (force_calculation.
            // precision = mixed / MD_NONBONDED_PRECISION), checked ahead of
            // the AVX-512/AVX2/scalar dispatch below: this check compiles on
            // any AVX2-capable build, including an AVX-512 one (AVX2 is
            // always available wherever AVX-512F is), so mixed precision is
            // no longer silently unreachable on an AVX-512 build the way the
            // old compile-time-only dispatch left it.
#if defined(__AVX2__)
            if (ctx.use_mixed_precision) {
                process_tile_avx2_mixed(T, tid, ctx, local_pe, local_virial);
            } else
#endif
#if defined(__AVX512F__)
            {
                process_tile_avx512(T, tid, ctx, local_pe, local_virial);
            }
#elif defined(__AVX2__)
            {
                process_tile_avx2(T, tid, ctx, local_pe, local_virial);
            }
#else
            {
                process_tile_scalar(T, tid, ctx, local_pe, local_virial);
            }
#endif
        }
    } // end omp parallel

    // Merge the nthreads private slices into the real output arrays, once.
    // Chunked so each (chunk, thread-slice) pair is read/written as one
    // contiguous run -- same cache-friendly layout as
    // ewald_spread_charges_cpu's thread-scratch merge above, and (unlike
    // process_tile_*'s writes) this is the ONLY place the real
    // particles.fx/fy/fz array is touched by this kernel, so there is
    // nothing for two threads to false-share here either: each chunk is
    // handled by exactly one thread, start to finish.
    constexpr long kForceMergeChunk = 4096;
    const long n_local = static_cast<long>(local_particles);
    const long n_merge_chunks = (n_local + kForceMergeChunk - 1) / kForceMergeChunk;
    #pragma omp parallel for schedule(static)
    for (long c = 0; c < n_merge_chunks; ++c) {
        const long begin = c * kForceMergeChunk;
        const long end = std::min(begin + kForceMergeChunk, n_local);
        for (int t = 0; t < nthreads; ++t) {
            const Real* __restrict__ sx = lj_thread_fx_.data() + static_cast<size_t>(t) * local_particles;
            const Real* __restrict__ sy = lj_thread_fy_.data() + static_cast<size_t>(t) * local_particles;
            const Real* __restrict__ sz = lj_thread_fz_.data() + static_cast<size_t>(t) * local_particles;
            for (long idx = begin; idx < end; ++idx) {
                real_fx[idx] += sx[idx];
                real_fy[idx] += sy[idx];
                real_fz[idx] += sz[idx];
            }
        }
    }

    last_potential_energy += local_pe;
    last_virial += local_virial;
    }); // run_nonbonded_with_pme_overlap
}

// ============================================================================
// EWALD / PME RECIPROCAL-SPACE ELECTROSTATICS (CPU, FFTW)
//
// Pipeline, all in place on ONE real-to-complex grid (see pme_real_grid_ in
// force_calculator_cpu.h):
//
//   spread  rho(r)  -- cardinal B-spline scatter, thread-private slices
//   R2C     rho(k)
//   green   Phi(k) = g(k) * rho(k), with the reciprocal energy and virial
//           accumulated in the same pass
//   C2R     Phi(r)
//   gather  F_i = -q_i * grad(spline interpolant of Phi)
//
// The forces come from the analytic gradient of the interpolant against the
// potential grid, so no separate Ex/Ey/Ez field grids and no extra inverse
// transforms are needed (the CUDA backend's pme_interpolate_forces_kernel
// takes the same route).
//
// TODO(ExaFMM): see the CUDA implementation's note -- an FMM-based solve
// would replace this spread/FFT/solve/interpolate sequence for multi-node
// scaling.
// ============================================================================

namespace {

Real ewald_solve_alpha_cpu(Real cutoff, Real tolerance) {
    Real lo = 1e-4, hi = 10.0 / cutoff;
    for (int iter = 0; iter < 100; ++iter) {
        Real mid = 0.5 * (lo + hi);
        if (std::erfc(mid * cutoff) > tolerance) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

// require_even is used for the z (contiguous) axis: the R2C/C2R transform
// stores nz/2+1 complex values there, and an even nz keeps the Nyquist plane
// a clean self-conjugate special case in the Hermitian weighting (see
// ensure_pme_green_cached) as well as giving FFTW a better-behaved split.
int ewald_next_good_fft_size_cpu(int minimum, bool require_even = false) {
    if (minimum < 8) minimum = 8;
    for (int n = minimum; ; ++n) {
        if (require_even && (n % 2 != 0)) continue;
        int m = n;
        while (m % 2 == 0) m /= 2;
        while (m % 3 == 0) m /= 3;
        while (m % 5 == 0) m /= 5;
        if (m == 1) return n;
    }
}

std::vector<Real> ewald_compute_bspline_moduli_cpu(int grid_size_1d, int order) {
    std::vector<Real> b_mod(grid_size_1d);
    for (int k = 0; k < grid_size_1d; ++k) {
        Real sum_real = 0.0, sum_imag = 0.0;
        for (int l = 0; l < order; ++l) {
            Real angle = 2.0 * M_PI * k * l / grid_size_1d;
            Real weight = 0.0;
            if (order == 4) {
                if (l == 0) weight = 1.0/6.0;
                else if (l == 1) weight = 2.0/3.0;
                else if (l == 2) weight = 1.0/6.0;
            } else if (order == 6) {
                if (l == 0) weight = 1.0/120.0;
                else if (l == 1) weight = 26.0/120.0;
                else if (l == 2) weight = 66.0/120.0;
                else if (l == 3) weight = 26.0/120.0;
                else if (l == 4) weight = 1.0/120.0;
            }
            sum_real += weight * std::cos(angle);
            sum_imag += weight * std::sin(angle);
        }
        Real m = std::sqrt(sum_real*sum_real + sum_imag*sum_imag);
        b_mod[k] = (m < 1e-10) ? 1.0 : m;
    }
    return b_mod;
}

// Cardinal cubic B-spline weights + grid offsets (matches the CUDA device
// function pme_compute_bspline_weights exactly, order 4 only -- the only
// spline order this code path uses).
void ewald_bspline_weights_cpu(Real u, int order, Real* weights, int* grid_indices) {
    int base_index = static_cast<int>(std::floor(u));
    Real w = u - static_cast<Real>(base_index);
    if (order == 4) {
        weights[0] = (1.0 - w) * (1.0 - w) * (1.0 - w) / 6.0;
        weights[1] = (3.0 * w * w * w - 6.0 * w * w + 4.0) / 6.0;
        weights[2] = (-3.0 * w * w * w + 3.0 * w * w + 3.0 * w + 1.0) / 6.0;
        weights[3] = w * w * w / 6.0;
        grid_indices[0] = base_index - 1;
        grid_indices[1] = base_index;
        grid_indices[2] = base_index + 1;
        grid_indices[3] = base_index + 2;
    }
}

inline int ewald_pbc_index_cpu(int idx, int grid_size) {
    idx %= grid_size;
    if (idx < 0) idx += grid_size;
    return idx;
}

void ewald_spread_charges_cpu(
    const Real* x, const Real* y, const Real* z, const Real* charge,
    Real* charge_grid,
    int grid_nx, int grid_ny, int grid_nz, int grid_nz_pad,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    int spline_order, size_t num_particles,
    std::vector<Real>& thread_scratch)
{
    // Padded real-grid size: the last (contiguous) axis is grid_nz_pad
    // reals, not grid_nz -- see the pme_real_grid_ comment in the header.
    // The pad slots are never touched here, which is what lets the merge
    // pass below write the whole buffer unconditionally.
    const size_t grid_size = static_cast<size_t>(grid_nx) * grid_ny * grid_nz_pad;
#ifdef _OPENMP
    const int nthreads = omp_get_max_threads();
#else
    const int nthreads = 1;
#endif
    const size_t scratch_size = static_cast<size_t>(nthreads) * grid_size;

    if (thread_scratch.size() != scratch_size) {
        thread_scratch.assign(scratch_size, 0.0);
    }

    #pragma omp parallel
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        Real* grid = thread_scratch.data() + static_cast<size_t>(tid) * grid_size;
        Real wx[8] = {}, wy[8] = {}, wz[8] = {};
        int ix[8] = {}, iy[8] = {}, iz[8] = {};
        int gx[8] = {}, gy[8] = {}, gz[8] = {};

        #pragma omp for schedule(static)
        for (long ii = 0; ii < static_cast<long>(num_particles); ++ii) {
            size_t i = static_cast<size_t>(ii);
            Real qi = charge[i];
            if (std::fabs(qi) < 1e-12) continue;

            Real u_x = (x[i] * inv_box_x + 0.5) * static_cast<Real>(grid_nx);
            Real u_y = (y[i] * inv_box_y + 0.5) * static_cast<Real>(grid_ny);
            Real u_z = (z[i] * inv_box_z + 0.5) * static_cast<Real>(grid_nz);
            u_x -= std::floor(u_x / grid_nx) * grid_nx;
            u_y -= std::floor(u_y / grid_ny) * grid_ny;
            u_z -= std::floor(u_z / grid_nz) * grid_nz;

            ewald_bspline_weights_cpu(u_x, spline_order, wx, ix);
            ewald_bspline_weights_cpu(u_y, spline_order, wy, iy);
            ewald_bspline_weights_cpu(u_z, spline_order, wz, iz);

            // PERF FIX: wrap the spline support indices ONCE per atom per
            // axis (order*3 = 12 modulo-fixups) instead of re-deriving the
            // same gz inside the innermost loop, where order^3 = 64 grid
            // points each paid an ewald_pbc_index_cpu plus two index
            // multiplies. The x/y row offsets are hoisted out of the inner
            // loops for the same reason, leaving the hot loop as one add
            // and one FMA per grid point.
            for (int a = 0; a < spline_order; ++a) gx[a] = ewald_pbc_index_cpu(ix[a], grid_nx);
            for (int b = 0; b < spline_order; ++b) gy[b] = ewald_pbc_index_cpu(iy[b], grid_ny);
            for (int c = 0; c < spline_order; ++c) gz[c] = ewald_pbc_index_cpu(iz[c], grid_nz);

            for (int a = 0; a < spline_order; ++a) {
                const size_t off_x = static_cast<size_t>(gx[a]) * grid_ny * grid_nz_pad;
                const Real wax = wx[a] * qi;
                for (int b = 0; b < spline_order; ++b) {
                    Real* __restrict__ row = grid + off_x + static_cast<size_t>(gy[b]) * grid_nz_pad;
                    const Real waxy = wy[b] * wax;
                    for (int c = 0; c < spline_order; ++c) {
                        row[gz[c]] += wz[c] * waxy;
                    }
                }
            }
        }
    }


    constexpr long kChunk = 4096;
    const long nchunks = static_cast<long>((grid_size + kChunk - 1) / kChunk);
    #pragma omp parallel for schedule(static)
    for (long c = 0; c < nchunks; ++c) {
        const long begin = c * kChunk;
        const long end = std::min(begin + kChunk, static_cast<long>(grid_size));
        for (int t = 0; t < nthreads; ++t) {
            Real* __restrict__ slice = thread_scratch.data() + static_cast<size_t>(t) * grid_size;
            if (t == 0) {
                #pragma omp simd
                for (long idx = begin; idx < end; ++idx) {
                    charge_grid[idx] = slice[idx];
                    slice[idx] = 0.0;
                }
            } else {
                #pragma omp simd
                for (long idx = begin; idx < end; ++idx) {
                    charge_grid[idx] += slice[idx];
                    slice[idx] = 0.0;
                }
            }
        }
    }
}

// Apply the precomputed Green's function in place (rho(k) -> Phi(k)) over
// the R2C half-spectrum, and accumulate the reciprocal-space energy AND
// virial in the SAME pass.
//
// PERF FIX: this replaces two separate full-grid passes that each recomputed
// the k vector, k^2, the B-spline modulus product, a std::exp and a division
// at every grid point, every step. Those quantities depend only on the grid
// and the box, so they now come out of three precomputed arrays
// (ensure_pme_green_cached) and this loop is a pure streaming
// multiply-accumulate. Fusing the energy and virial sums into one pass also
// halves the grid traffic, since the virial pass re-read the whole grid only
// to form |rho|^2 a second time.
//
// The Hermitian multiplicity (the conjugate half the R2C transform does not
// store) is already baked into green_energy/green_virial, and the k=0 bin is
// zero in all three arrays, which reproduces the old explicit DC special
// case without a branch.
Real ewald_apply_green_cpu(
    fftw_complex* fft_grid,
    const Real* __restrict__ green,
    const Real* __restrict__ green_energy,
    const Real* __restrict__ green_virial,
    size_t num_k, Real& virial_out)
{
    Real energy = 0.0;
    Real virial = 0.0;
    #pragma omp parallel for reduction(+:energy, virial) schedule(static)
    for (long idx = 0; idx < static_cast<long>(num_k); ++idx) {
        const Real re = fft_grid[idx][0];
        const Real im = fft_grid[idx][1];
        const Real rho_sq = re * re + im * im;
        energy += green_energy[idx] * rho_sq;
        virial += green_virial[idx] * rho_sq;
        const Real g = green[idx];
        fft_grid[idx][0] = g * re;
        fft_grid[idx][1] = g * im;
    }
    virial_out = virial;
    return energy;
}

// PERF FIX: this used to compute E(k) = -i*k*
// Phi(k) into three separate complex grids and inverse-FFT each of them --
// 3 extra backward FFTs on top of the 1 forward, i.e. 4 FFTs total per PME
// evaluation, measured at ~72ms of a ~110-120ms rank-0 reciprocal-space
// pass (mostly those 3 backward transforms; the CUDA backend never had
// this cost -- see force_calculator_cuda.cu's pme_interpolate_forces_kernel
// and its "avoid three field grids entirely" comment). Mirrors that CUDA
// approach now: inverse-FFT the POTENTIAL grid once (Phi(k) -> Phi(r), done
// by the caller reusing pme_fft_plan_bwd_ in place on the PME grid),
// then get each atom's force from the analytic derivative of the cardinal
// B-spline interpolant against Phi(r) directly -- no separate field grids,
// no extra FFTs. ewald_bspline_weights_cpu's grid indices are reused as-is;
// only the weights need a derivative counterpart.
void ewald_bspline_weights_and_derivatives_cpu(
    Real u, int order, Real* weights, Real* derivatives, int* grid_indices) {
    ewald_bspline_weights_cpu(u, order, weights, grid_indices);
    const Real w = u - std::floor(u);
    if (order == 4) {
        const Real omw = 1.0 - w;
        const Real w2 = w * w;
        derivatives[0] = -0.5 * omw * omw;
        derivatives[1] =  1.5 * w2 - 2.0 * w;
        derivatives[2] = -1.5 * w2 + w + 0.5;
        derivatives[3] =  0.5 * w2;
    }
}

// Reads the REAL part of `potential_grid` (Phi(r), UNNORMALIZED -- FFTW's
// inverse transform doesn't divide by grid_size, folded into inv_grid_size
// below instead of a separate full-grid normalization pass) and accumulates
// each local atom's PME force directly, via the interpolant's analytic
// gradient. Safe to parallelize directly (unlike ewald_spread_charges_cpu
// above): each atom i only ever reads the shared, read-only potential grid
// and writes its own fx[i]/fy[i]/fz[i] slot.
void ewald_interpolate_forces_from_potential_cpu(
    const Real* x, const Real* y, const Real* z, const Real* charge,
    const Real* potential_grid,
    Real* fx, Real* fy, Real* fz,
    int grid_nx, int grid_ny, int grid_nz, int grid_nz_pad,
    Real inv_box_x, Real inv_box_y, Real inv_box_z,
    int spline_order, size_t num_particles)
{
    const Real inv_grid_size =
        1.0 / (static_cast<Real>(grid_nx) * grid_ny * grid_nz);
    #pragma omp parallel for schedule(static)
    for (long ii = 0; ii < static_cast<long>(num_particles); ++ii) {
        size_t i = static_cast<size_t>(ii);
        Real wx[8] = {}, wy[8] = {}, wz[8] = {};
        Real dwx[8] = {}, dwy[8] = {}, dwz[8] = {};
        int ix[8] = {}, iy[8] = {}, iz[8] = {};
        int gx[8] = {}, gy[8] = {}, gz[8] = {};
        Real qi = charge[i];
        if (std::fabs(qi) < 1e-12) continue;

        Real u_x = (x[i] * inv_box_x + 0.5) * static_cast<Real>(grid_nx);
        Real u_y = (y[i] * inv_box_y + 0.5) * static_cast<Real>(grid_ny);
        Real u_z = (z[i] * inv_box_z + 0.5) * static_cast<Real>(grid_nz);
        u_x -= std::floor(u_x / grid_nx) * grid_nx;
        u_y -= std::floor(u_y / grid_ny) * grid_ny;
        u_z -= std::floor(u_z / grid_nz) * grid_nz;

        ewald_bspline_weights_and_derivatives_cpu(u_x, spline_order, wx, dwx, ix);
        ewald_bspline_weights_and_derivatives_cpu(u_y, spline_order, wy, dwy, iy);
        ewald_bspline_weights_and_derivatives_cpu(u_z, spline_order, wz, dwz, iz);

        // PERF FIX: index wrapping hoisted out of the order^3 inner loop --
        // see ewald_spread_charges_cpu.
        for (int a = 0; a < spline_order; ++a) gx[a] = ewald_pbc_index_cpu(ix[a], grid_nx);
        for (int b = 0; b < spline_order; ++b) gy[b] = ewald_pbc_index_cpu(iy[b], grid_ny);
        for (int c = 0; c < spline_order; ++c) gz[c] = ewald_pbc_index_cpu(iz[c], grid_nz);

        Real dphi_du_x = 0.0, dphi_du_y = 0.0, dphi_du_z = 0.0;
        for (int a = 0; a < spline_order; ++a) {
            const size_t off_x = static_cast<size_t>(gx[a]) * grid_ny * grid_nz_pad;
            for (int b = 0; b < spline_order; ++b) {
                const Real* __restrict__ row = potential_grid + off_x
                                             + static_cast<size_t>(gy[b]) * grid_nz_pad;
                const Real wab   = wx[a]  * wy[b];
                const Real dwa_b = dwx[a] * wy[b];
                const Real wa_db = wx[a]  * dwy[b];
                for (int c = 0; c < spline_order; ++c) {
                    const Real phi = row[gz[c]];
                    dphi_du_x += dwa_b * wz[c]  * phi;
                    dphi_du_y += wa_db * wz[c]  * phi;
                    dphi_du_z += wab   * dwz[c] * phi;
                }
            }
        }
        fx[i] += -qi * dphi_du_x * (grid_nx * inv_box_x) * inv_grid_size;
        fy[i] += -qi * dphi_du_y * (grid_ny * inv_box_y) * inv_grid_size;
        fz[i] += -qi * dphi_du_z * (grid_nz * inv_box_z) * inv_grid_size;
    }
}

}  // namespace

void ForceCalculatorCPU::ensure_ewald_initialized(MDSystem& system) {
    SimulationBox& box = system.get_box();
    const Real k_coulomb = 332.0636;

    if (ewald_initialized_) {
        // Grid size and alpha stay fixed after first init; only the
        // volume-dependent Green's-function prefactor is refreshed (NPT).
        ewald_prefactor_ = k_coulomb * 4.0 * M_PI / (box.lx * box.ly * box.lz);
        return;
    }

    Real cutoff = system.get_physical_cutoff();
    ewald_alpha_ = ewald_solve_alpha_cpu(cutoff, 1e-5);
    ewald_spline_order_ = 4;

    // PERF FIX: see the ewald_rs_*_table_ member comment in force_calculator_
    // cpu.h -- replaces the per-pair sqrt+exp+erfc/erf approximation in the
    // real-space Ewald hot loops with a table lookup. Fine, uniform spacing
    // in r^2 (0.005 A^2 -- e.g. ~28.8k points for a 12 A cutoff, ~230KB/
    // table) keeps interpolation error far below the fast_erfc_from_exp
    // approximation's own ~1.5e-7 error it replaces, while entirely
    // eliminating sqrt/exp/the rational approximation from the pair loop.
    ewald_rs_table_r2_min_ = 1e-4;
    const Real ewald_rs_table_spacing = 0.005;
    ewald_rs_table_inv_spacing_ = 1.0 / ewald_rs_table_spacing;
    ewald_rs_table_size_ = build_real_space_ewald_table(
        cutoff, ewald_alpha_, ewald_rs_table_r2_min_,
        ewald_rs_table_spacing, /*use_erf=*/false, ewald_rs_table_);
    build_real_space_ewald_table(
        cutoff, ewald_alpha_, ewald_rs_table_r2_min_,
        ewald_rs_table_spacing, /*use_erf=*/true, ewald_rs_table_excl_);

    // Grid spacing must resolve the reciprocal-space Gaussian width, which
    // narrows as alpha grows.
    //
    // PERF FIX: this used to target alpha*h <= 0.25 capped at 1.0 Angstrom,
    // which for a typical 12 A cutoff / 1e-5 tolerance (alpha ~ 0.26) gave
    // h ~ 0.96 A. That is considerably finer than the accepted standard --
    // GROMACS ships 0.12 nm = 1.2 A at spline order 4 -- and the cost is
    // cubic: a 1.25x finer spacing on each axis is ~2x the grid points, and
    // every grid point is paid for again in the FFT, the Poisson pass, the
    // spread scratch and the MPI reduce/broadcast. Relaxed to alpha*h <= 0.4
    // capped at 1.2 A, matching the mainstream accuracy/speed tradeoff.
    //
    // MD_PME_GRID_SPACING overrides it (in Angstrom) for anyone who needs
    // the old, tighter grid or wants to sweep the tradeoff without a
    // rebuild; the reciprocal-space energy is the quantity to watch when
    // changing it.
    Real target_spacing = std::min(static_cast<Real>(1.2), static_cast<Real>(0.4) / ewald_alpha_);
    if (const char* spacing_env = std::getenv("MD_PME_GRID_SPACING")) {
        const Real v = static_cast<Real>(std::atof(spacing_env));
        if (v > 0.0) target_spacing = v;
    }
    ewald_grid_x_ = ewald_next_good_fft_size_cpu(static_cast<int>(std::ceil(box.lx / target_spacing)));
    ewald_grid_y_ = ewald_next_good_fft_size_cpu(static_cast<int>(std::ceil(box.ly / target_spacing)));
    // z is the R2C axis -- keep it even (see ewald_next_good_fft_size_cpu).
    ewald_grid_z_ = ewald_next_good_fft_size_cpu(static_cast<int>(std::ceil(box.lz / target_spacing)), true);

    pme_grid_z_half_ = ewald_grid_z_ / 2 + 1;
    pme_grid_z_pad_ = 2 * pme_grid_z_half_;

    // One allocation, two views (see the pme_real_grid_ comment in the
    // header). fftw_alloc_real gives the SIMD alignment the plans want.
    const size_t real_grid_size =
        static_cast<size_t>(ewald_grid_x_) * ewald_grid_y_ * pme_grid_z_pad_;
    pme_real_grid_ = fftw_alloc_real(real_grid_size);
    pme_complex_grid_ = reinterpret_cast<fftw_complex*>(pme_real_grid_);
    std::memset(pme_real_grid_, 0, real_grid_size * sizeof(Real));

    // STAGE 1 PARALLELIZATION: if FFTW was built with its OpenMP-threaded
    // backend (libfftw3_omp; see CMakeLists.txt's optional
    // DEEPSCALEMD_HAVE_FFTW_OMP detection), have every plan created below
    // (the forward transform and the 3 reused backward transforms) actually
    // use the same thread count as everything else in this process. With
    // OMP_NUM_THREADS unset or 1, omp_get_max_threads() is 1 and this is a
    // harmless no-op -- fftw_plan_with_nthreads(1) behaves identically to
    // never having called it. fftw_init_threads() must be called exactly
    // once, before any threaded plan is created; ensure_ewald_initialized
    // only reaches this point once (see the ewald_initialized_ early-return
    // above), so no guard against repeat calls is needed here.
#ifdef DEEPSCALEMD_HAVE_FFTW_OMP
    {
        int threads_ok = fftw_init_threads();
        if (threads_ok) {
#ifdef _OPENMP
            fftw_plan_with_nthreads(omp_get_max_threads());
#else
            fftw_plan_with_nthreads(1);
#endif
        } else if (is_rank_zero()) {
            std::cerr << "[Ewald/PME][CPU] WARNING: fftw_init_threads() failed; "
                         "PME FFTs will run single-threaded." << std::endl;
        }
    }
#endif

    // One forward plan and one backward plan, BOTH in place on the same
    // buffer: the R2C forward turns rho(r) into rho(k) (then the Green's
    // function pass turns that into Phi(k) in place), the C2R backward turns
    // Phi(k) back into Phi(r) for ewald_interpolate_forces_from_potential_cpu
    // to read directly -- no separate field grids (see that function's
    // comment for why this replaced a 3-extra-backward-FFT scheme).
    // PERF FIX: R2C/C2R rather than the old complex-to-complex pair -- the
    // density is real, so half the transform was computing a spectrum half
    // that Hermitian symmetry already determines. See pme_real_grid_ in the
    // header. C2R destroys its input, which is fine here: the Green's
    // function pass and the virial sum both run before the backward
    // transform.
    // PERF FIX: FFTW_MEASURE (not FFTW_ESTIMATE) -- these plans are created
    // once here and reused every single MD step for the rest of the run, so
    // paying a one-time planning cost (FFTW actually times a few candidate
    // algorithms against these buffers) to get a faster steady-state
    // execution plan is a clear net win. FFTW_MEASURE overwrites its input/
    // output buffers while timing candidates, which is fine here: the buffer
    // is fully repopulated (memset + spread) before every real use, so the
    // garbage left over from planning is never read.
    pme_fft_plan_fwd_ = fftw_plan_dft_r2c_3d(
        ewald_grid_x_, ewald_grid_y_, ewald_grid_z_,
        pme_real_grid_, pme_complex_grid_, FFTW_MEASURE);
    pme_fft_plan_bwd_ = fftw_plan_dft_c2r_3d(
        ewald_grid_x_, ewald_grid_y_, ewald_grid_z_,
        pme_complex_grid_, pme_real_grid_, FFTW_MEASURE);

    pme_b_spline_mod_x_ = ewald_compute_bspline_moduli_cpu(ewald_grid_x_, ewald_spline_order_);
    pme_b_spline_mod_y_ = ewald_compute_bspline_moduli_cpu(ewald_grid_y_, ewald_spline_order_);
    pme_b_spline_mod_z_ = ewald_compute_bspline_moduli_cpu(ewald_grid_z_, ewald_spline_order_);

    ewald_prefactor_ = k_coulomb * 4.0 * M_PI / (box.lx * box.ly * box.lz);
    ewald_initialized_ = true;

    if (is_rank_zero()) {
        std::cout << "[Ewald/PME][CPU] grid " << ewald_grid_x_ << "x" << ewald_grid_y_ << "x" << ewald_grid_z_
                  << " (r2c), alpha=" << ewald_alpha_ << " 1/A, spline_order=" << ewald_spline_order_
                  << ", spacing<=" << target_spacing << " A, cutoff=" << cutoff << " A" << std::endl;
    }
}

void ForceCalculatorCPU::free_ewald_resources() {
    if (pme_fft_plan_fwd_) { fftw_destroy_plan(pme_fft_plan_fwd_); pme_fft_plan_fwd_ = nullptr; }
    if (pme_fft_plan_bwd_) { fftw_destroy_plan(pme_fft_plan_bwd_); pme_fft_plan_bwd_ = nullptr; }
    if (pme_real_grid_) { fftw_free(pme_real_grid_); pme_real_grid_ = nullptr; }
    pme_complex_grid_ = nullptr;
    // The cached Green's function is sized to the grid that just went away,
    // so drop it too -- a re-initialization would otherwise find a non-empty
    // cache of the wrong length.
    pme_green_.clear();
    pme_green_energy_.clear();
    pme_green_virial_.clear();
    pme_green_cached_lx_ = pme_green_cached_ly_ = pme_green_cached_lz_ = 0.0;
    ewald_initialized_ = false;
}

// See the pme_green_ comment in force_calculator_cpu.h. Everything here is a
// function of the grid dimensions, alpha and the box, so it is rebuilt only
// when the box actually changes (i.e. never, outside NPT) rather than being
// recomputed at every grid point on every step in two separate passes.
void ForceCalculatorCPU::ensure_pme_green_cached(const SimulationBox& box) {
    if (!pme_green_.empty() &&
        pme_green_cached_lx_ == box.lx &&
        pme_green_cached_ly_ == box.ly &&
        pme_green_cached_lz_ == box.lz) {
        return;
    }

    const int nx = ewald_grid_x_, ny = ewald_grid_y_, nz = ewald_grid_z_;
    const int nzh = pme_grid_z_half_;
    const size_t num_k = static_cast<size_t>(nx) * ny * nzh;
    pme_green_.assign(num_k, 0.0);
    pme_green_energy_.assign(num_k, 0.0);
    pme_green_virial_.assign(num_k, 0.0);

    const Real inv_box_x = 1.0 / box.lx, inv_box_y = 1.0 / box.ly, inv_box_z = 1.0 / box.lz;
    const Real alpha_sq = ewald_alpha_ * ewald_alpha_;
    const Real inv_4alpha_sq = 1.0 / (4.0 * alpha_sq);
    const Real inv_2alpha_sq = 1.0 / (2.0 * alpha_sq);
    const Real two_pi = 2.0 * M_PI;

    #pragma omp parallel for schedule(static)
    for (int kx = 0; kx < nx; ++kx) {
        const Real kxw = (kx <= nx / 2 ? kx : kx - nx) * two_pi * inv_box_x;
        const Real bx = pme_b_spline_mod_x_[kx];
        for (int ky = 0; ky < ny; ++ky) {
            const Real kyw = (ky <= ny / 2 ? ky : ky - ny) * two_pi * inv_box_y;
            const Real by = pme_b_spline_mod_y_[ky];
            const Real bxy_sq = (bx * bx) * (by * by);
            for (int kz = 0; kz < nzh; ++kz) {
                const size_t idx = (static_cast<size_t>(kx) * ny + ky) * nzh + kz;
                if (kx == 0 && ky == 0 && kz == 0) continue;  // arrays already 0 at DC
                const Real kzw = kz * two_pi * inv_box_z;     // kz <= nz/2 by construction
                const Real k_sq = kxw * kxw + kyw * kyw + kzw * kzw;
                const Real bz = pme_b_spline_mod_z_[kz];
                const Real m_sq = bxy_sq * (bz * bz);
                const Real green = ewald_prefactor_ * std::exp(-k_sq * inv_4alpha_sq)
                                 / (k_sq * m_sq + 1e-12);

                // Hermitian multiplicity: the R2C transform stores only
                // kz in [0, nz/2]. Every other kz plane has an unstored
                // conjugate partner with identical |rho|^2 and identical
                // green (both depend on k only through k^2 and |b|, which
                // are symmetric under k -> -k), so it counts twice. The
                // kz == 0 and kz == nz/2 planes are their own conjugates
                // within the stored data and count once.
                const Real w = (kz == 0 || 2 * kz == nz) ? 1.0 : 2.0;

                pme_green_[idx] = green;
                pme_green_energy_[idx] = 0.5 * w * green;
                pme_green_virial_[idx] = w * green * (1.0 + k_sq * inv_2alpha_sq);
            }
        }
    }

    pme_green_cached_lx_ = box.lx;
    pme_green_cached_ly_ = box.ly;
    pme_green_cached_lz_ = box.lz;
}

// See the declaration comment in force_calculator_cpu.h. Bonded topology
// (and therefore the exclusion set) never changes during a run, so this only
// ever does real work once -- decodes Topology::get_exclusion_keys() (sorted
// by pair_key = (min<<32)|max) into a per-global-atom-id CSR via a two-pass
// counting sort (degree count, then scatter), then sorts each atom's small
// partner sub-range so exclusion_row_contains' binary_search (and its
// [lo,hi] range reject) is valid.
void ForceCalculatorCPU::ensure_exclusion_cache_built(const Topology& topology) {
    if (excl_cache_built_) return;

    const size_t num_atoms = static_cast<size_t>(topology.get_num_atoms());
    const auto& keys = topology.get_exclusion_keys();

    std::vector<int> degree(num_atoms + 1, 0);
    for (std::uint64_t key : keys) {
        int lo = static_cast<int>(key >> 32);
        int hi = static_cast<int>(key & 0xffffffffu);
        ++degree[static_cast<size_t>(lo)];
        ++degree[static_cast<size_t>(hi)];
    }

    excl_cache_offsets_.assign(num_atoms + 1, 0);
    for (size_t a = 0; a < num_atoms; ++a) {
        excl_cache_offsets_[a + 1] = excl_cache_offsets_[a] + degree[a];
    }

    excl_cache_partners_.assign(static_cast<size_t>(excl_cache_offsets_[num_atoms]), 0);
    std::vector<int> cursor(excl_cache_offsets_);
    for (std::uint64_t key : keys) {
        int lo = static_cast<int>(key >> 32);
        int hi = static_cast<int>(key & 0xffffffffu);
        excl_cache_partners_[static_cast<size_t>(cursor[static_cast<size_t>(lo)]++)] = hi;
        excl_cache_partners_[static_cast<size_t>(cursor[static_cast<size_t>(hi)]++)] = lo;
    }

    for (size_t a = 0; a < num_atoms; ++a) {
        std::sort(excl_cache_partners_.begin() + excl_cache_offsets_[a],
                  excl_cache_partners_.begin() + excl_cache_offsets_[a + 1]);
    }

    excl_cache_built_ = true;
}

// STAGE 3 PLACEHOLDER -- see the doc comment on the declaration in
// force_calculator_cpu.h. MDSimulation's config parser rejects
// electrostatics_solver = fmm before this is ever reachable; the throw here
// is a defense-in-depth backstop, not the primary guard.
void ForceCalculatorCPU::compute_electrostatic_forces_fmm(MDSystem& /*system*/) {
    throw std::runtime_error(
        "ForceCalculatorCPU::compute_electrostatic_forces_fmm: FMM long-range "
        "electrostatics is not implemented yet (placeholder for future work). "
        "Use force_calculation.electrostatics_solver = pme instead.");
}

// Reciprocal-space (long-range, FFTW PME grid) part of the Ewald sum, plus
// the self-energy correction. Split out so the real-space pass
// (compute_tiled_nonbonded()/compute_graph_nonbonded()) can run
// underneath the non-blocking grid reduce instead of serializing in front
// of it -- see
// pme_reciprocal_begin/end's declaration comment in force_calculator_cpu.h.
// This wrapper (begin+end back-to-back) is for callers with nothing to
// overlap.
void ForceCalculatorCPU::compute_electrostatic_forces_reciprocal(MDSystem& system) {
    pme_reciprocal_begin(system);
    pme_reciprocal_end(system);
}

// Phase 1: spread the local charges onto the PME grid and start the
// inter-rank reduction. Touches only the PME grid; the caller is free to run
// the real-space pair kernel before calling pme_reciprocal_end().
void ForceCalculatorCPU::pme_reciprocal_begin(MDSystem& system) {
    ensure_ewald_initialized(system);

    using clock = std::chrono::high_resolution_clock;
    const auto t0 = clock::now();

    ParticleData& particles = system.get_particles();
    SimulationBox& box = system.get_box();
    // Local particles only: ghosts are periodic copies of atoms owned by
    // other MPI ranks and must not be double-counted in the charge density.
    const size_t local_particles = system.get_local_particles();

    // Green's function: built once, reused until the box changes (see
    // ensure_pme_green_cached). ensure_ewald_initialized() has already
    // refreshed ewald_prefactor_ for the current volume.
    ensure_pme_green_cached(box);

    const Real inv_box_x = 1.0/box.lx, inv_box_y = 1.0/box.ly, inv_box_z = 1.0/box.lz;
    const size_t real_grid_size =
        static_cast<size_t>(ewald_grid_x_) * ewald_grid_y_ * pme_grid_z_pad_;

    // No memset here: ewald_spread_charges_cpu's merge pass writes (not
    // accumulates) the first thread slice over the whole buffer, padding
    // included, so the grid is fully defined without a separate zeroing sweep.
    ewald_spread_charges_cpu(
        particles.x.data(), particles.y.data(), particles.z.data(), particles.charge.data(),
        pme_real_grid_, ewald_grid_x_, ewald_grid_y_, ewald_grid_z_, pme_grid_z_pad_,
        inv_box_x, inv_box_y, inv_box_z, ewald_spline_order_, local_particles,
        pme_charge_spread_priv_);

    // MPI: each rank has so far spread only its own locally-owned charges.
    // The FFT/solve/interpolate pipeline needs the FULL global charge
    // density -- otherwise the reciprocal-space force/energy would reflect
    // only this rank's ~1/mpi_size slice of the system's charges, which is
    // wrong and, via the exclusion correction in the real-space pass (which
    // assumes the reciprocal grid already contains every charged pair's
    // erf(alpha*r)/r term), injects large spurious forces on bonded pairs
    // that straddle rank boundaries.
    //
    // Reduce (not allreduce) onto rank 0, which alone runs the transform;
    // the compact real-space RESULT is broadcast back in phase 2. An
    // Allreduce would instead have every rank redundantly repeat the
    // identical full-grid transform.
    //
    // PERF FIX: non-blocking, so this moves underneath the real-space kernel
    // (see the declaration comment). The R2C layout already halved the bytes.
    int mpi_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    pme_reduce_pending_ = false;
    if (mpi_size > 1) {
        int mpi_rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
        MPI_Ireduce(mpi_rank == 0 ? MPI_IN_PLACE : pme_real_grid_,
                    pme_real_grid_, static_cast<int>(real_grid_size),
                    MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, &pme_reduce_req_);
        pme_reduce_pending_ = true;
    }

    pme_time_ms_ = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

// Phase 2: finish the reduction, transform and solve on rank 0, broadcast
// Phi(r), and interpolate each rank's own local forces out of it.
void ForceCalculatorCPU::pme_reciprocal_end(MDSystem& system) {
    using clock = std::chrono::high_resolution_clock;
    const auto t0 = clock::now();

    ParticleData& particles = system.get_particles();
    SimulationBox& box = system.get_box();
    const Real k_coulomb = 332.0636;
    const size_t local_particles = system.get_local_particles();

    const Real inv_box_x = 1.0/box.lx, inv_box_y = 1.0/box.ly, inv_box_z = 1.0/box.lz;
    const size_t real_grid_size =
        static_cast<size_t>(ewald_grid_x_) * ewald_grid_y_ * pme_grid_z_pad_;
    const size_t num_k =
        static_cast<size_t>(ewald_grid_x_) * ewald_grid_y_ * pme_grid_z_half_;

    int mpi_size = 1, mpi_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    if (mpi_size > 1) MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    if (pme_reduce_pending_) {
        MPI_Wait(&pme_reduce_req_, MPI_STATUS_IGNORE);
        pme_reduce_pending_ = false;
    }

    // recip_energy/recip_virial default to 0 on non-root ranks: the later
    // cross-rank sums (last_potential_energy via MPI_Allreduce in
    // compute_forces(), last_virial via comm_backend_->sum_reduce() in
    // compute_output_thermodynamics()) add every rank's contribution, so
    // "root contributes the true total, everyone else contributes 0" gives
    // the correct grand total without needing to broadcast these scalars
    // back out separately.
    Real recip_energy = 0.0;
    Real recip_virial = 0.0;
    if (mpi_rank == 0) {
        fftw_execute(pme_fft_plan_fwd_);   // rho(r) -> rho(k), R2C

        // Energy, virial and the in-place Phi(k) = green * rho(k) scaling all
        // come out of ONE streaming pass over the half-spectrum (see
        // ewald_apply_green_cpu), replacing two full-grid passes that each
        // recomputed the Green's function from scratch.
        recip_energy = ewald_apply_green_cpu(
            pme_complex_grid_, pme_green_.data(), pme_green_energy_.data(),
            pme_green_virial_.data(), num_k, recip_virial);

        // Single in-place backward transform (Phi(k) -> Phi(r)) instead of
        // building 3 separate field grids and inverse-FFTing each -- see
        // ewald_interpolate_forces_from_potential_cpu's comment. No explicit
        // normalization pass either: FFTW's missing 1/grid_size factor is
        // folded directly into that function's force expression.
        fftw_execute(pme_fft_plan_bwd_);   // Phi(k) -> Phi(r), C2R
    }
    last_potential_energy += recip_energy;
    last_virial += recip_virial;

    if (mpi_size > 1) {
        MPI_Bcast(pme_real_grid_, static_cast<int>(real_grid_size), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    ewald_interpolate_forces_from_potential_cpu(
        particles.x.data(), particles.y.data(), particles.z.data(), particles.charge.data(),
        pme_real_grid_,
        particles.fx.data(), particles.fy.data(), particles.fz.data(),
        ewald_grid_x_, ewald_grid_y_, ewald_grid_z_, pme_grid_z_pad_,
        inv_box_x, inv_box_y, inv_box_z, ewald_spline_order_, local_particles);

    // ---- Self-energy correction (local particles only) ----
    // Not cached: atoms migrate between ranks, so the local charge set is
    // not invariant across steps. It is a single vectorizable O(N) pass.
    Real self_energy_sum_q2 = 0.0;
    const Real* __restrict__ pq = particles.charge.data();
    #pragma omp simd reduction(+:self_energy_sum_q2)
    for (long i = 0; i < static_cast<long>(local_particles); ++i) {
        self_energy_sum_q2 += pq[i] * pq[i];
    }
    last_potential_energy -= k_coulomb * ewald_alpha_ / 1.7724538509055159 * self_energy_sum_q2;  // sqrt(pi)

    if (timing_breakdown_enabled_) {
        // Charge spread, the reduce/broadcast, rank 0's forward FFT, the
        // Green's-function pass and backward FFT, force interpolation and the
        // self-energy sum -- but NOT the real-space kernel that ran between
        // the two phases, which is timed by its own component.
        pme_time_ms_ += std::chrono::duration<double, std::milli>(clock::now() - t0).count();
        last_force_component_timings_.push_back({"coulomb_recip", pme_time_ms_});
    }
}

void ForceCalculatorCPU::compute_bond_forces(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    Topology& topology = system.get_topology();
    SimulationBox& box = system.get_box();

    // Check if we have bonds
    const auto& bonds = system.get_local_bonds();
    if (bonds.empty()) {
        return; // No bonds defined
    }

    // PERF FIX: was a fully serial loop regardless of OMP_NUM_THREADS -- cheap
    // per iteration but O(num_bonds) every single step, so at high thread
    // counts (where the nonbonded kernel below scales down by ~8x) this and
    // the other bonded terms became a growing FIXED (non-scaling) fraction of
    // the step. Parallelized with atomics on the scatter-add (two atoms only
    // ever share a bond rarely enough that atomic RMW cost is negligible next
    // to the win of using every thread here) and a reduction for the energy
    // sum -- no change to the physics, purely how the same per-bond work is
    // scheduled across threads.
    Real local_pe = 0.0;
    const long num_bonds = static_cast<long>(bonds.size());
    #pragma omp parallel for reduction(+:local_pe) schedule(static)
    for (long bi = 0; bi < num_bonds; ++bi) {
        const auto& bond = bonds[static_cast<size_t>(bi)];
        size_t i = bond.atom1;
        size_t j = bond.atom2;

        if (i >= particles.x.size() || j >= particles.x.size()) continue;

        Real dx = particles.x[i] - particles.x[j];
        Real dy = particles.y[i] - particles.y[j];
        Real dz = particles.z[i] - particles.z[j];

        // Apply periodic boundary conditions
        box.apply_minimum_image(dx, dy, dz);

        Real r = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (r < 1e-10) continue; // Avoid division by zero

        // Get bond parameters from topology
        const BondType& bond_type = topology.get_bond_type(bond.type);
        Real k_bond = bond_type.k;    // Force constant
        Real r0 = bond_type.r0;       // Equilibrium distance

        // Harmonic bond potential: V = 0.5 * k * (r - r0)^2
        // Force: F = -k * (r - r0) * r_hat
        Real dr = r - r0;
        local_pe += 0.5 * k_bond * dr * dr;
        Real force_mag = -k_bond * dr / r;

        Real fx = force_mag * dx;
        Real fy = force_mag * dy;
        Real fz = force_mag * dz;

        #pragma omp atomic
        particles.fx[i] += fx;
        #pragma omp atomic
        particles.fy[i] += fy;
        #pragma omp atomic
        particles.fz[i] += fz;
        #pragma omp atomic
        particles.fx[j] -= fx;
        #pragma omp atomic
        particles.fy[j] -= fy;
        #pragma omp atomic
        particles.fz[j] -= fz;
    }
    last_potential_energy += local_pe;
}

void ForceCalculatorCPU::compute_angle_forces(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    Topology& topology = system.get_topology();
    SimulationBox& box = system.get_box();

    // Check if we have angles
    const auto& angles = system.get_local_angles();
    if (angles.empty()) {
        return; // No angles defined
    }

    // PERF FIX: see compute_bond_forces()'s comment -- same parallelization,
    // same atomics-on-scatter-add + energy-reduction pattern, no change to
    // the physics.
    Real local_pe = 0.0;
    const long num_angles = static_cast<long>(angles.size());
    #pragma omp parallel for reduction(+:local_pe) schedule(static)
    for (long ai = 0; ai < num_angles; ++ai) {
        const auto& angle = angles[static_cast<size_t>(ai)];
        size_t i = angle.atom1;  // First atom
        size_t j = angle.atom2;  // Central atom
        size_t k = angle.atom3;  // Third atom

        if (i >= particles.x.size() || j >= particles.x.size() || k >= particles.x.size()) continue;

        // Vectors from central atom j to atoms i and k
        Real dx_ij = particles.x[i] - particles.x[j];
        Real dy_ij = particles.y[i] - particles.y[j];
        Real dz_ij = particles.z[i] - particles.z[j];

        Real dx_kj = particles.x[k] - particles.x[j];
        Real dy_kj = particles.y[k] - particles.y[j];
        Real dz_kj = particles.z[k] - particles.z[j];

        // Apply periodic boundary conditions
        box.apply_minimum_image(dx_ij, dy_ij, dz_ij);
        box.apply_minimum_image(dx_kj, dy_kj, dz_kj);
        
        Real r_ij = std::sqrt(dx_ij*dx_ij + dy_ij*dy_ij + dz_ij*dz_ij);
        Real r_kj = std::sqrt(dx_kj*dx_kj + dy_kj*dy_kj + dz_kj*dz_kj);
        
        if (r_ij < 1e-10 || r_kj < 1e-10) continue; // Avoid division by zero
        
        // Calculate angle
        Real dot_product = dx_ij*dx_kj + dy_ij*dy_kj + dz_ij*dz_kj;
        Real cos_theta = dot_product / (r_ij * r_kj);
        
        // Clamp to avoid numerical issues
        if (cos_theta > 1.0) cos_theta = 1.0;
        if (cos_theta < -1.0) cos_theta = -1.0;
        
        Real theta = std::acos(cos_theta);
        
        // Get angle parameters from topology
        const AngleType& angle_type = topology.get_angle_type(angle.type);
        Real k_angle = angle_type.k;         // Force constant
        Real theta0 = angle_type.theta0;     // Equilibrium angle
        Real dtheta = theta - theta0;
        // Harmonic angle potential: V = 0.5 * k * (theta - theta0)^2
        local_pe += 0.5 * k_angle * dtheta * dtheta;

        if (std::abs(dtheta) < 1e-10) continue; // No force if at equilibrium
        
        // Use more numerically stable force calculation
        Real sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);
        if (sin_theta < 1e-10) continue; // Avoid division by zero
        
        // -dV/dtheta times dtheta/dr.  The vector expressions below are
        // (v_hat - cos(theta) u_hat)/r, i.e. the negative angular gradient,
        // so the scalar coefficient must be positive here.
        Real force_constant = k_angle * dtheta / sin_theta;
        
        // Force calculation (simplified - this is a complex derivation)
        Real r_ij_inv = 1.0 / r_ij;
        Real r_kj_inv = 1.0 / r_kj;
        
        Real cos_theta_ij = cos_theta / r_ij;
        Real cos_theta_kj = cos_theta / r_kj;
        
        Real fx_i = force_constant * (dx_kj * r_ij_inv * r_kj_inv - dx_ij * cos_theta_ij * r_ij_inv);
        Real fy_i = force_constant * (dy_kj * r_ij_inv * r_kj_inv - dy_ij * cos_theta_ij * r_ij_inv);
        Real fz_i = force_constant * (dz_kj * r_ij_inv * r_kj_inv - dz_ij * cos_theta_ij * r_ij_inv);
        
        Real fx_k = force_constant * (dx_ij * r_ij_inv * r_kj_inv - dx_kj * cos_theta_kj * r_kj_inv);
        Real fy_k = force_constant * (dy_ij * r_ij_inv * r_kj_inv - dy_kj * cos_theta_kj * r_kj_inv);
        Real fz_k = force_constant * (dz_ij * r_ij_inv * r_kj_inv - dz_kj * cos_theta_kj * r_kj_inv);
        
        // Forces on central atom j (Newton's third law)
        Real fx_j = -(fx_i + fx_k);
        Real fy_j = -(fy_i + fy_k);
        Real fz_j = -(fz_i + fz_k);
        
        #pragma omp atomic
        particles.fx[i] += fx_i;
        #pragma omp atomic
        particles.fy[i] += fy_i;
        #pragma omp atomic
        particles.fz[i] += fz_i;
        #pragma omp atomic
        particles.fx[j] += fx_j;
        #pragma omp atomic
        particles.fy[j] += fy_j;
        #pragma omp atomic
        particles.fz[j] += fz_j;
        #pragma omp atomic
        particles.fx[k] += fx_k;
        #pragma omp atomic
        particles.fy[k] += fy_k;
        #pragma omp atomic
        particles.fz[k] += fz_k;
    }
    last_potential_energy += local_pe;
}

void ForceCalculatorCPU::compute_dihedral_forces(MDSystem& system) {
    compute_torsion_forces(system, system.get_local_dihedrals());
}

void ForceCalculatorCPU::compute_improper_forces(MDSystem& system) {
    compute_torsion_forces(system, system.get_local_impropers());
}

void ForceCalculatorCPU::compute_torsion_forces(MDSystem& system,
                                                 const std::vector<MDSystem::LocalDihedral>& dihedrals) {
    ParticleData& particles = system.get_particles();
    Topology& topology = system.get_topology();

    if (dihedrals.empty()) {
        return; // No torsions of this kind defined
    }

    // PERF FIX: see compute_bond_forces()'s comment -- same parallelization,
    // same atomics-on-scatter-add + energy-reduction pattern, no change to
    // the physics. Shared by both proper dihedrals and impropers, so both
    // benefit.
    SimulationBox& box = system.get_box();
    Real local_pe = 0.0;
    const long num_dihedrals = static_cast<long>(dihedrals.size());
    #pragma omp parallel for reduction(+:local_pe) schedule(static)
    for (long di = 0; di < num_dihedrals; ++di) {
        const auto& dihedral = dihedrals[static_cast<size_t>(di)];
        size_t i = dihedral.atom1;
        size_t j = dihedral.atom2;
        size_t k = dihedral.atom3;
        size_t l = dihedral.atom4;

        if (i >= particles.x.size() || j >= particles.x.size() ||
            k >= particles.x.size() || l >= particles.x.size()) continue;

        // Vectors for dihedral calculation
        Real dx_ij = particles.x[i] - particles.x[j];
        Real dy_ij = particles.y[i] - particles.y[j];
        Real dz_ij = particles.z[i] - particles.z[j];

        Real dx_jk = particles.x[j] - particles.x[k];
        Real dy_jk = particles.y[j] - particles.y[k];
        Real dz_jk = particles.z[j] - particles.z[k];

        Real dx_kl = particles.x[k] - particles.x[l];
        Real dy_kl = particles.y[k] - particles.y[l];
        Real dz_kl = particles.z[k] - particles.z[l];

        // Apply periodic boundary conditions
        box.apply_minimum_image(dx_ij, dy_ij, dz_ij);
        box.apply_minimum_image(dx_jk, dy_jk, dz_jk);
        box.apply_minimum_image(dx_kl, dy_kl, dz_kl);

        // Cross products for plane normals -- kept UNNORMALIZED, the exact
        // analytical force formula below needs the raw vectors and |.|^2.
        Real n1_x = dy_ij * dz_jk - dz_ij * dy_jk;  // ij x jk
        Real n1_y = dz_ij * dx_jk - dx_ij * dz_jk;
        Real n1_z = dx_ij * dy_jk - dy_ij * dx_jk;

        Real n2_x = dy_jk * dz_kl - dz_jk * dy_kl;  // jk x kl
        Real n2_y = dz_jk * dx_kl - dx_jk * dz_kl;
        Real n2_z = dx_jk * dy_kl - dy_jk * dx_kl;

        Real n1_sq = n1_x*n1_x + n1_y*n1_y + n1_z*n1_z;
        Real n2_sq = n2_x*n2_x + n2_y*n2_y + n2_z*n2_z;
        Real jk_sq = dx_jk*dx_jk + dy_jk*dy_jk + dz_jk*dz_jk;

        if (n1_sq < 1e-20 || n2_sq < 1e-20 || jk_sq < 1e-20) continue; // Degenerate geometry

        Real jk_mag = std::sqrt(jk_sq);

        // Dihedral angle via atan2(sin, cos) computed directly from the
        // unnormalized cross products (equivalent to normalizing first, but
        // avoids the extra sqrt/division and keeps n1/n2 intact for the
        // force formula below).
        Real cos_phi_num = n1_x*n2_x + n1_y*n2_y + n1_z*n2_z;
        Real sin_phi_num = (n1_y*n2_z - n1_z*n2_y) * dx_jk +
                            (n1_z*n2_x - n1_x*n2_z) * dy_jk +
                            (n1_x*n2_y - n1_y*n2_x) * dz_jk;
        Real phi = std::atan2(sin_phi_num, cos_phi_num * jk_mag);

        // Get dihedral parameters from topology
        const DihedralType& dihedral_type = topology.get_dihedral_type(dihedral.type);
        Real k_dihedral = dihedral_type.k;
        Real phi0 = dihedral_type.phi0;  // Phase shift
        int n = dihedral_type.n;         // Multiplicity

        Real dVdphi;
        if (n == 0) {
            // CHARMM harmonic improper: V = Kpsi*(phi-phi0)^2. Keep the
            // angular displacement on the principal branch.
            Real dphi = std::remainder(phi - phi0, 2.0 * M_PI);
            local_pe += k_dihedral * dphi * dphi;
            dVdphi = 2.0 * k_dihedral * dphi;
        } else {
            // CHARMM-style periodic torsion: V = k*(1+cos(n*phi-phi0)).
            Real dphi = n * phi - phi0;
            local_pe += k_dihedral * (1.0 + std::cos(dphi));
            dVdphi = -k_dihedral * n * std::sin(dphi);
        }

        // Exact analytical gradient (Blondel-Karplus / Bekker formula),
        // momentum- and torque-conserving by construction. Using the
        // standard notation r_ij, r_kj = -dx_jk, r_kl, m = r_ij x r_kj,
        // n_std = r_kj x r_kl (note m = -n1, n_std = -n2 for the ij x jk /
        // jk x kl cross products already computed above), the textbook
        // With the signed angle convention used by atan2 above, dphi/dr_i
        // points along +n1 and dphi/dr_l along -n2. Force is -dV/dphi times
        // that gradient.
        Real fi_scale = -dVdphi * jk_mag / n1_sq;
        Real fl_scale =  dVdphi * jk_mag / n2_sq;

        Real fix = fi_scale * n1_x, fiy = fi_scale * n1_y, fiz = fi_scale * n1_z;
        Real flx = fl_scale * n2_x, fly = fl_scale * n2_y, flz = fl_scale * n2_z;

        // p, q are the fractional projections of r_ij and r_kl onto r_kj
        // (= -dx_jk here, but the sign cancels in the dot-product ratio).
        Real p = -(dx_ij*dx_jk + dy_ij*dy_jk + dz_ij*dz_jk) / jk_sq;
        Real q = -(dx_kl*dx_jk + dy_kl*dy_jk + dz_kl*dz_jk) / jk_sq;

        Real fjx = (p - 1.0) * fix - q * flx;
        Real fjy = (p - 1.0) * fiy - q * fly;
        Real fjz = (p - 1.0) * fiz - q * flz;

        Real fkx = -p * fix + (q - 1.0) * flx;
        Real fky = -p * fiy + (q - 1.0) * fly;
        Real fkz = -p * fiz + (q - 1.0) * flz;

        #pragma omp atomic
        particles.fx[i] += fix;
        #pragma omp atomic
        particles.fy[i] += fiy;
        #pragma omp atomic
        particles.fz[i] += fiz;
        #pragma omp atomic
        particles.fx[j] += fjx;
        #pragma omp atomic
        particles.fy[j] += fjy;
        #pragma omp atomic
        particles.fz[j] += fjz;
        #pragma omp atomic
        particles.fx[k] += fkx;
        #pragma omp atomic
        particles.fy[k] += fky;
        #pragma omp atomic
        particles.fz[k] += fkz;
        #pragma omp atomic
        particles.fx[l] += flx;
        #pragma omp atomic
        particles.fy[l] += fly;
        #pragma omp atomic
        particles.fz[l] += flz;
    }
    last_potential_energy += local_pe;
}

void ForceCalculatorCPU::compute_special_pair_forces(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    ForceFieldParams& params = system.get_ff_params();
    SimulationBox& box = system.get_box();

    const auto& pairs = system.get_local_special_pairs();
    if (pairs.empty()) {
        return; // No special pairs defined
    }

    const ForceManager& force_manager = system.get_force_manager();
    const auto& force_config = force_manager.get_config();
    const bool do_lj = force_config.enable_lennard_jones;
    const bool do_coulomb = force_config.enable_coulomb;
    if (!do_lj && !do_coulomb) {
        return;
    }

    // Direct, unscreened LJ + Coulomb -- never subject to the normal
    // neighbor-list cutoff. Requires the pair to also be listed in
    // [exclusions] so the main pairwise loops skip it. Under Ewald/PME, the
    // reciprocal-space sum includes the erf(alpha*r)/r term unconditionally
    // for every charged pair, and compute_tiled_nonbonded()/
    // compute_graph_nonbonded() subtract that same term for excluded pairs
    // (do_coulomb=true), so the net Coulomb
    // contribution from those two routines is exactly zero here -- only the
    // scaled direct term added below survives.
    const Real k_coulomb = 332.0636; // kcal/mol*Angstrom / e^2 (matches the Ewald real-space routine)
    Real local_pe = 0.0;
    Real local_virial = 0.0;

    // PERF FIX: see compute_bond_forces()'s comment -- same parallelization,
    // same atomics-on-scatter-add + energy/virial-reduction pattern, no
    // change to the physics.
    const long num_pairs = static_cast<long>(pairs.size());
    #pragma omp parallel for reduction(+:local_pe, local_virial) schedule(static)
    for (long pi = 0; pi < num_pairs; ++pi) {
        const auto& pair = pairs[static_cast<size_t>(pi)];
        size_t i = pair.atom1;
        size_t j = pair.atom2;
        if (i >= particles.x.size() || j >= particles.x.size()) continue;

        Real dx = particles.x[i] - particles.x[j];
        Real dy = particles.y[i] - particles.y[j];
        Real dz = particles.z[i] - particles.z[j];
        box.apply_minimum_image(dx, dy, dz);
        Real r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < 1e-10) continue; // Avoid division by zero

        Real force_mag = 0.0;

        if (do_lj) {
            int type_i = particles.type[i];
            int type_j = particles.type[j];
            Real sigma = pair.sigma >= 0.0 ? pair.sigma : params.sigma_mixed[type_i][type_j];
            Real epsilon = pair.epsilon >= 0.0 ? pair.epsilon : params.epsilon_mixed[type_i][type_j];
            if (epsilon != 0.0) {
                Real r2inv = 1.0 / r2;
                Real attractive;
                Real repulsive;
                if (params.mie_repulsive_exponent == 12) {
                    const Real sig2 = sigma * sigma;
                    const Real sig6 = sig2 * sig2 * sig2;
                    attractive = sig6 * r2inv * r2inv * r2inv;
                    repulsive = attractive * attractive;
                } else {
                    const Real sig_r = sigma * std::sqrt(r2inv);
                    const Real sig3_r3 = sig_r * sig_r * sig_r;
                    attractive = sig3_r3 * sig3_r3;
                    repulsive = attractive * sig3_r3;
                }
                force_mag += params.mie_prefactor * epsilon * r2inv *
                    (params.mie_repulsive_exponent * repulsive -
                     params.mie_attractive_exponent * attractive) * pair.lj_scale;
                local_pe += params.mie_prefactor * epsilon *
                    (repulsive - attractive) * pair.lj_scale;
            }
        }

        if (do_coulomb) {
            Real qi = particles.charge[i];
            Real qj = particles.charge[j];
            if (qi != 0.0 && qj != 0.0) {
                Real r = std::sqrt(r2);
                Real qq = k_coulomb * qi * qj * pair.coulomb_scale;
                force_mag += qq / (r2 * r);
                local_pe += qq / r;
            }
        }

        if (force_mag == 0.0) continue;

        local_virial += r2 * force_mag; // W = r.F = r^2 * force_mag

        Real fx = force_mag * dx;
        Real fy = force_mag * dy;
        Real fz = force_mag * dz;

        #pragma omp atomic
        particles.fx[i] += fx;
        #pragma omp atomic
        particles.fy[i] += fy;
        #pragma omp atomic
        particles.fz[i] += fz;
        #pragma omp atomic
        particles.fx[j] -= fx;
        #pragma omp atomic
        particles.fy[j] -= fy;
        #pragma omp atomic
        particles.fz[j] -= fz;
    }

    last_potential_energy += local_pe;
    last_virial += local_virial;
}

void ForceCalculatorCPU::calculate_kinetic_energy(MDSystem& system) {
    ParticleData& particles = system.get_particles();
    size_t local_particles = system.get_local_particles();
    
    last_kinetic_energy = 0.0;
    
    for (size_t i = 0; i < local_particles; i++) {
        Real v2 = particles.vx[i] * particles.vx[i] + 
                  particles.vy[i] * particles.vy[i] + 
                  particles.vz[i] * particles.vz[i];
        last_kinetic_energy += 0.5 * particles.mass[i] * v2;
    }
    
    if (system.get_rank() == 0) {
        // Debug removed
    }
    
    // Sum across all MPI ranks
    Real global_ke = 0.0;
    MPI_Allreduce(&last_kinetic_energy, &global_ke, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    last_kinetic_energy = global_ke;
}
