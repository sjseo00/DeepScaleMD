#pragma once

// Piecewise-cubic tables of the real-space Ewald pair terms, indexed by the bits of the float r^2
// (list nonbonded kernel).
//
//   energy(r2) = erfc(alpha r) / r
//   force(r2)  = (erfc(alpha r) / r + 2 alpha / sqrt(pi) exp(-alpha^2 r^2)) / r^2   (= F / r)
//
// The caller multiplies both by k_coulomb q_i q_j. r^2 is split by its float exponent and the top
// EWALD_TABLE_BITS mantissa bits into intervals of constant relative width (1/32 .. 1/64 of r^2),
// so one table serves the whole range from 0.5 A^2 to the cutoff with the same accuracy. Each
// interval holds the cubic c0 + t (c1 + t (c2 + t c3)), t in [0, 1) the position inside the
// interval (linear in r^2), fitted in double through Chebyshev nodes. The cubic error is below
// 1e-8 relative to the value, under the rounding error of a float evaluation.
//
// Host side only (plus the shared constants); the device code reads the coefficient arrays.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>

constexpr int EWALD_TABLE_BITS = 5;                    // 32 intervals per octave of r^2
constexpr int EWALD_TABLE_E0 = -1;                     // the table starts at r^2 = 2^-1
constexpr int EWALD_TABLE_MAX_OCTAVES = 16;
constexpr unsigned EWALD_TABLE_BASE_BITS = static_cast<unsigned>(127 + EWALD_TABLE_E0) << 23;
constexpr int EWALD_TABLE_INDEX_SHIFT = 23 - EWALD_TABLE_BITS;

// Number of intervals needed so that every r^2 < cutoff_sq is inside the table.
inline int ewald_real_table_intervals(double cutoff_sq) {
    int top = static_cast<int>(std::ceil(std::log2(std::max(cutoff_sq, 1.0))));
    int octaves = std::min(std::max(top - EWALD_TABLE_E0, 1), EWALD_TABLE_MAX_OCTAVES);
    return octaves << EWALD_TABLE_BITS;
}

// Four coefficients per interval for the force and the energy term.
inline void ewald_real_table_build(double alpha, int intervals,
                                   std::vector<float>& force_coef, std::vector<float>& energy_coef) {
    force_coef.assign(static_cast<size_t>(intervals) * 4, 0.0f);
    energy_coef.assign(static_cast<size_t>(intervals) * 4, 0.0f);
    const double two_alpha_over_sqrt_pi = 2.0 * alpha / std::sqrt(3.14159265358979323846);

    auto energy_fn = [&](double r2) {
        const double r = std::sqrt(r2);
        return std::erfc(alpha * r) / r;
    };
    auto force_fn = [&](double r2) {
        const double r = std::sqrt(r2);
        return (std::erfc(alpha * r) / r + two_alpha_over_sqrt_pi * std::exp(-alpha * alpha * r2)) / r2;
    };

    // Chebyshev nodes of [0, 1].
    double nodes[4];
    for (int j = 0; j < 4; ++j) {
        nodes[j] = 0.5 - 0.5 * std::cos((2.0 * j + 1.0) * 3.14159265358979323846 / 8.0);
    }
    // Monomial coefficients from the four values: solve the Vandermonde system once per
    // function and interval (4x4, long double, partial pivoting).
    auto fit = [&](double y[4], double c[4]) {
        long double m[4][5];
        for (int row = 0; row < 4; ++row) {
            long double p = 1.0L;
            for (int col = 0; col < 4; ++col) {
                m[row][col] = p;
                p *= nodes[row];
            }
            m[row][4] = y[row];
        }
        for (int col = 0; col < 4; ++col) {
            int pivot = col;
            for (int row = col + 1; row < 4; ++row) {
                if (std::fabs(static_cast<double>(m[row][col])) > std::fabs(static_cast<double>(m[pivot][col]))) pivot = row;
            }
            for (int k = 0; k < 5; ++k) std::swap(m[col][k], m[pivot][k]);
            for (int row = 0; row < 4; ++row) {
                if (row == col) continue;
                const long double factor = m[row][col] / m[col][col];
                for (int k = col; k < 5; ++k) m[row][k] -= factor * m[col][k];
            }
        }
        for (int col = 0; col < 4; ++col) c[col] = static_cast<double>(m[col][4] / m[col][col]);
    };

    const int per_octave = 1 << EWALD_TABLE_BITS;
    for (int i = 0; i < intervals; ++i) {
        const int octave = i / per_octave;
        const int step = i % per_octave;
        const double base = std::ldexp(1.0, EWALD_TABLE_E0 + octave);
        const double r2_begin = base * (1.0 + static_cast<double>(step) / per_octave);
        const double width = base / per_octave;
        double yf[4], ye[4], cf[4], ce[4];
        for (int j = 0; j < 4; ++j) {
            const double r2 = r2_begin + nodes[j] * width;
            yf[j] = force_fn(r2);
            ye[j] = energy_fn(r2);
        }
        fit(yf, cf);
        fit(ye, ce);
        for (int k = 0; k < 4; ++k) {
            force_coef[static_cast<size_t>(i) * 4 + k] = static_cast<float>(cf[k]);
            energy_coef[static_cast<size_t>(i) * 4 + k] = static_cast<float>(ce[k]);
        }
    }
}

// Same arithmetic as the device code (for the build-time self check and offline tests).
// Returns false when r2 is outside the table.
inline bool ewald_real_table_eval(const std::vector<float>& coef, int intervals, float r2, float& value) {
    uint32_t bits;
    std::memcpy(&bits, &r2, sizeof(bits));
    const uint32_t shifted = bits - EWALD_TABLE_BASE_BITS;
    const uint32_t index = shifted >> EWALD_TABLE_INDEX_SHIFT;
    if (index >= static_cast<uint32_t>(intervals)) return false;
    const uint32_t fraction_bits = 0x3F800000u | ((shifted << EWALD_TABLE_BITS) & 0x007FFFFFu);
    float t;
    std::memcpy(&t, &fraction_bits, sizeof(t));
    t -= 1.0f;
    const float* c = &coef[static_cast<size_t>(index) * 4];
    value = ((c[3] * t + c[2]) * t + c[1]) * t + c[0];
    return true;
}
