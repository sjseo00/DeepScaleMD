#include "../include/neighbor_list.h"
#include "../include/tile_grid_info.h"
#include <cuda_runtime.h>
#include <map>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <climits>
#include <algorithm>

#ifndef NEIGHBOR_LIST_CUDA_CHECK
#define NEIGHBOR_LIST_CUDA_CHECK(call) do { \
    cudaError_t nlk_err = (call); \
    if (nlk_err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(nlk_err)); \
        exit(1); \
    } \
} while (0)
#endif

__global__ void build_cell_list_kernel(
    const Real* __restrict__ x,
    const Real* __restrict__ y,
    const Real* __restrict__ z,
    int* __restrict__ cell_start,
    int* __restrict__ cell_count,
    int* __restrict__ particle_cell_id,
    int n_particles,
    Real inv_cell_x, Real inv_cell_y, Real inv_cell_z,
    int nx, int ny, int nz,
    Real lx, Real ly, Real lz,
    bool periodic_x, bool periodic_y, bool periodic_z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    
    Real px = x[i];
    Real py = y[i];
    Real pz = z[i];
    
    if (periodic_x) px = px - floor(px / lx) * lx;
    if (periodic_y) py = py - floor(py / ly) * ly;
    if (periodic_z) pz = pz - floor(pz / lz) * lz;
    
    int cx = min(max(0, (int)(px * inv_cell_x)), nx - 1);
    int cy = min(max(0, (int)(py * inv_cell_y)), ny - 1);
    int cz = min(max(0, (int)(pz * inv_cell_z)), nz - 1);
    
    int cell_idx = cz * nx * ny + cy * nx + cx;
    particle_cell_id[i] = cell_idx;
    
    atomicAdd(&cell_count[cell_idx], 1);
}

__global__ void populate_cells_kernel(
    const int* __restrict__ particle_cell_id,
    int* __restrict__ cell_start,
    int* __restrict__ cell_particles,
    int* __restrict__ cell_counters,
    int n_particles)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    
    int cell_idx = particle_cell_id[i];
    int pos = atomicAdd(&cell_counters[cell_idx], 1);
    int cell_offset = cell_start[cell_idx];
    cell_particles[cell_offset + pos] = i;
}


constexpr int LJ_TILE_SIZE = 8;

__device__ inline unsigned long long lj_tile_local_morton(uint32_t x, uint32_t y, uint32_t z) {
    auto split_by_3 = [](uint32_t a) -> unsigned long long {
        unsigned long long v = a & 0x1fffffull;
        v = (v | v << 32) & 0x1f00000000ffffull;
        v = (v | v << 16) & 0x1f0000ff0000ffull;
        v = (v | v << 8)  & 0x100f00f00f00f00full;
        v = (v | v << 4)  & 0x10c30c30c30c30c3ull;
        v = (v | v << 2)  & 0x1249249249249249ull;
        return v;
    };
    return split_by_3(x) | (split_by_3(y) << 1) | (split_by_3(z) << 2);
}

__global__ void compute_tile_sort_keys_kernel(
    const Real* __restrict__ x, const Real* __restrict__ y, const Real* __restrict__ z,
    const int* __restrict__ particle_cell_id,
    unsigned long long* __restrict__ sort_keys,
    int n_total,
    int nx, int ny,
    Real cell_size_x, Real cell_size_y, Real cell_size_z,
    Real lx, Real ly, Real lz,
    bool periodic_x, bool periodic_y, bool periodic_z)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_total) return;

    int cell = particle_cell_id[i];
    int cz = cell / (nx * ny);
    int rem = cell % (nx * ny);
    int cy = rem / nx;
    int cx = rem % nx;

    Real px = x[i], py = y[i], pz = z[i];
    if (periodic_x) px = px - floor(px / lx) * lx;
    if (periodic_y) py = py - floor(py / ly) * ly;
    if (periodic_z) pz = pz - floor(pz / lz) * lz;

    Real rel_x = (px - cx * cell_size_x) / cell_size_x;
    Real rel_y = (py - cy * cell_size_y) / cell_size_y;
    Real rel_z = (pz - cz * cell_size_z) / cell_size_z;
    const uint32_t Q = 1u << 13;  // 13 bits/axis of local resolution
    uint32_t qx = (uint32_t)min(max(rel_x, (Real)0.0) * Q, (Real)(Q - 1));
    uint32_t qy = (uint32_t)min(max(rel_y, (Real)0.0) * Q, (Real)(Q - 1));
    uint32_t qz = (uint32_t)min(max(rel_z, (Real)0.0) * Q, (Real)(Q - 1));

    unsigned long long local_morton = lj_tile_local_morton(qx, qy, qz) & ((1ull << 40) - 1);
    sort_keys[i] = (((unsigned long long)(unsigned int)cell) << 40) | local_morton;
}

__global__ void compute_cell_tile_counts_kernel(
    const int* __restrict__ cell_count, int* __restrict__ tile_count, long long n_cells)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;
    tile_count[c] = (cell_count[c] + LJ_TILE_SIZE - 1) / LJ_TILE_SIZE;
}

__global__ void scatter_tile_cell_id_kernel(
    const int* __restrict__ tile_offset, const int* __restrict__ tile_count,
    int* __restrict__ tile_cell_id, long long n_cells)
{
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;
    int base = tile_offset[c];
    int n = tile_count[c];
    for (int k = 0; k < n; ++k) tile_cell_id[base + k] = (int)c;
}

__global__ void scatter_lj_tile_atoms_kernel(
    const int* __restrict__ tile_sorted_indices,
    const int* __restrict__ particle_cell_id,
    const int* __restrict__ cell_start,
    const int* __restrict__ tile_offset,
    int* __restrict__ tile_atoms,
    int n_total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) return;
    int p = tile_sorted_indices[idx];
    int cell = particle_cell_id[p];
    int pos_in_cell = idx - cell_start[cell];
    int tile_local = pos_in_cell / LJ_TILE_SIZE;
    int slot = pos_in_cell % LJ_TILE_SIZE;
    int global_tile = tile_offset[cell] + tile_local;
    tile_atoms[(long long)global_tile * LJ_TILE_SIZE + slot] = p;
}

__global__ void compute_tile_bboxes_kernel(
    const Real* __restrict__ x, const Real* __restrict__ y, const Real* __restrict__ z,
    const int* __restrict__ tile_atoms,
    const int* __restrict__ tile_cell_id,
    int nx, int ny,
    Real cell_size_x, Real cell_size_y, Real cell_size_z,
    Real lx, Real ly, Real lz,
    bool periodic_x, bool periodic_y, bool periodic_z,
    Real* __restrict__ tile_bbox_min, Real* __restrict__ tile_bbox_max,
    unsigned char* __restrict__ tile_flags, int n_local,
    int num_tiles)
{
    const int WARP_SIZE = 32;
    int tile = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (tile >= num_tiles) return;

    int cell = tile_cell_id[tile];
    int cz = cell / (nx * ny);
    int rem = cell % (nx * ny);
    int cy = rem / nx;
    int cx = rem % nx;

    int atom = (lane < LJ_TILE_SIZE) ? tile_atoms[tile * LJ_TILE_SIZE + lane] : -1;
    Real rel_x = (Real)1e30, rel_y = (Real)1e30, rel_z = (Real)1e30;      // for min-reduce
    Real rel_x2 = (Real)-1e30, rel_y2 = (Real)-1e30, rel_z2 = (Real)-1e30; // for max-reduce
    if (atom >= 0) {
        Real px = x[atom], py = y[atom], pz = z[atom];
        if (periodic_x) px = px - floor(px / lx) * lx;
        if (periodic_y) py = py - floor(py / ly) * ly;
        if (periodic_z) pz = pz - floor(pz / lz) * lz;
        rel_x = rel_x2 = px - cx * cell_size_x;
        rel_y = rel_y2 = py - cy * cell_size_y;
        rel_z = rel_z2 = pz - cz * cell_size_z;
    }

    // Bit 0: the tile holds an owned atom; bit 1: it holds a ghost atom.
    const unsigned owned_votes = __ballot_sync(0xffffffffu, atom >= 0 && atom < n_local);
    const unsigned ghost_votes = __ballot_sync(0xffffffffu, atom >= n_local);

    for (int offset = 16; offset > 0; offset >>= 1) {
        rel_x = min(rel_x, __shfl_down_sync(0xffffffffu, rel_x, offset));
        rel_y = min(rel_y, __shfl_down_sync(0xffffffffu, rel_y, offset));
        rel_z = min(rel_z, __shfl_down_sync(0xffffffffu, rel_z, offset));
        rel_x2 = max(rel_x2, __shfl_down_sync(0xffffffffu, rel_x2, offset));
        rel_y2 = max(rel_y2, __shfl_down_sync(0xffffffffu, rel_y2, offset));
        rel_z2 = max(rel_z2, __shfl_down_sync(0xffffffffu, rel_z2, offset));
    }
    if (lane == 0) {
        tile_flags[tile] = static_cast<unsigned char>((owned_votes ? 1u : 0u) | (ghost_votes ? 2u : 0u));
        tile_bbox_min[tile*3+0] = rel_x;  tile_bbox_min[tile*3+1] = rel_y;  tile_bbox_min[tile*3+2] = rel_z;
        tile_bbox_max[tile*3+0] = rel_x2; tile_bbox_max[tile*3+1] = rel_y2; tile_bbox_max[tile*3+2] = rel_z2;
    }
}


// Marks the cells that hold a tile with a ghost atom (flag bit 1). Every writer stores the same value.
__global__ void mark_ghost_cells_kernel(const unsigned char* __restrict__ tile_flags,
                                        const int* __restrict__ tile_cell_id,
                                        unsigned char* __restrict__ cell_has_ghost, int num_tiles)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tiles) return;
    if (tile_flags[t] & 2u) cell_has_ghost[tile_cell_id[t]] = 1;
}

constexpr int FINE_STENCIL_RADIUS = 2;
constexpr int FINE_STENCIL_SIDE = 2 * FINE_STENCIL_RADIUS + 1;                    // 5
constexpr int FINE_STENCIL_OFFSETS = FINE_STENCIL_SIDE * FINE_STENCIL_SIDE * FINE_STENCIL_SIDE; // 125

__global__ void build_lj_tile_pairs_fine_kernel(
    const int* __restrict__ tile_offset,
    const int* __restrict__ tile_count,
    const Real* __restrict__ tile_bbox_min,
    const Real* __restrict__ tile_bbox_max,
    const unsigned char* __restrict__ tile_flags,
    int* __restrict__ tile_pair_i,
    int* __restrict__ tile_pair_j,
    int* __restrict__ pair_counter,
    int pair_capacity,
    int nx, int ny, int nz,
    Real cell_size_x, Real cell_size_y, Real cell_size_z,
    bool periodic_x, bool periodic_y, bool periodic_z,
    Real cutoff_sq_extended,
    long long n_cells)
{
    const int WARP_SIZE = 32;
    long long warp_id = ((long long)blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    if (warp_id >= n_cells) return;

    long long home = warp_id;
    int cz = (int)(home / ((long long)nx * ny));
    int rem = (int)(home % ((long long)nx * ny));
    int cy = rem / nx;
    int cx = rem % nx;

    int n_tiles_home = tile_count[home];
    if (n_tiles_home == 0) return;
    int home_tile_base = tile_offset[home];

    for (int base = 0; base < FINE_STENCIL_OFFSETS; base += WARP_SIZE) {
        int offset_idx = base + lane;
        int neighbor_cell = -1;
        int off_dx = 0, off_dy = 0, off_dz = 0;
        if (offset_idx < FINE_STENCIL_OFFSETS) {
            int dz = offset_idx / (FINE_STENCIL_SIDE * FINE_STENCIL_SIDE) - FINE_STENCIL_RADIUS;
            int rem2 = offset_idx % (FINE_STENCIL_SIDE * FINE_STENCIL_SIDE);
            int dy = rem2 / FINE_STENCIL_SIDE - FINE_STENCIL_RADIUS;
            int dx = rem2 % FINE_STENCIL_SIDE - FINE_STENCIL_RADIUS;
            off_dx = dx; off_dy = dy; off_dz = dz;

            int ncx = cx + dx, ncy = cy + dy, ncz = cz + dz;
            bool valid = true;
            if (periodic_x) { if (ncx < 0) ncx += nx; if (ncx >= nx) ncx -= nx; }
            else if (ncx < 0 || ncx >= nx) valid = false;
            if (periodic_y) { if (ncy < 0) ncy += ny; if (ncy >= ny) ncy -= ny; }
            else if (ncy < 0 || ncy >= ny) valid = false;
            if (periodic_z) { if (ncz < 0) ncz += nz; if (ncz >= nz) ncz -= nz; }
            else if (ncz < 0 || ncz >= nz) valid = false;

            if (valid) neighbor_cell = ncz * nx * ny + ncy * nx + ncx;
        }

        for (int src_lane = 0; src_lane < WARP_SIZE; ++src_lane) {
            int nb_cell = __shfl_sync(0xffffffffu, neighbor_cell, src_lane);
            if (nb_cell < 0) continue;
            int dx = __shfl_sync(0xffffffffu, off_dx, src_lane);
            int dy = __shfl_sync(0xffffffffu, off_dy, src_lane);
            int dz = __shfl_sync(0xffffffffu, off_dz, src_lane);

            int n_tiles_nb = tile_count[nb_cell];
            if (n_tiles_nb == 0) continue;
            int nb_tile_base = tile_offset[nb_cell];
            int total_combos = n_tiles_home * n_tiles_nb;

            Real off_x = dx * cell_size_x, off_y = dy * cell_size_y, off_z = dz * cell_size_z;

            // The slot counter is a single address: one atomicAdd per surviving tile pair
            // (millions per build) serializes in the L2. The warp ranks its survivors with
            // a ballot and its leader reserves the whole block of slots at once. The trip
            // count is uniform over the warp, so every lane takes part in the ballot.
            for (int combo_base = 0; combo_base < total_combos; combo_base += WARP_SIZE) {
                const int combo = combo_base + lane;
                bool keep = false;
                int home_tile = 0, nb_tile = 0;
                if (combo < total_combos) {
                    const int ti = combo / n_tiles_nb;
                    const int tj = combo % n_tiles_nb;
                    home_tile = home_tile_base + ti;
                    nb_tile = nb_tile_base + tj;

                    // The list is a half list (home <= neighbor): every owned-owned
                    // pair appears once and Newton's third law supplies the other
                    // force. An owned-ghost pair has no such partner (the ghost's
                    // force belongs to another rank), so it must be listed with the
                    // OWNED atom's tile as home. When the owned atom sits in the
                    // higher-numbered tile that is the reverse orientation, emitted
                    // only where it can contain such a pair; the mask kernel keeps
                    // just the ghost neighbors of those entries.
                    keep = !(home_tile > nb_tile &&
                             !((tile_flags[home_tile] & 1u) && (tile_flags[nb_tile] & 2u)));
                    if (keep) {
                        Real hmin_x = tile_bbox_min[home_tile*3+0], hmax_x = tile_bbox_max[home_tile*3+0];
                        Real hmin_y = tile_bbox_min[home_tile*3+1], hmax_y = tile_bbox_max[home_tile*3+1];
                        Real hmin_z = tile_bbox_min[home_tile*3+2], hmax_z = tile_bbox_max[home_tile*3+2];
                        Real nmin_x = tile_bbox_min[nb_tile*3+0] + off_x, nmax_x = tile_bbox_max[nb_tile*3+0] + off_x;
                        Real nmin_y = tile_bbox_min[nb_tile*3+1] + off_y, nmax_y = tile_bbox_max[nb_tile*3+1] + off_y;
                        Real nmin_z = tile_bbox_min[nb_tile*3+2] + off_z, nmax_z = tile_bbox_max[nb_tile*3+2] + off_z;

                        Real gap_x = max((Real)0.0, max(hmin_x - nmax_x, nmin_x - hmax_x));
                        Real gap_y = max((Real)0.0, max(hmin_y - nmax_y, nmin_y - hmax_y));
                        Real gap_z = max((Real)0.0, max(hmin_z - nmax_z, nmin_z - hmax_z));

                        keep = gap_x*gap_x + gap_y*gap_y + gap_z*gap_z < cutoff_sq_extended;
                    }
                }

                const unsigned keep_mask = __ballot_sync(0xffffffffu, keep);
                if (keep_mask != 0u) {
                    int base_slot = 0;
                    if (lane == __ffs(keep_mask) - 1) base_slot = atomicAdd(pair_counter, __popc(keep_mask));
                    base_slot = __shfl_sync(0xffffffffu, base_slot, __ffs(keep_mask) - 1);
                    if (keep) {
                        const int slot = base_slot + __popc(keep_mask & ((1u << lane) - 1u));
                        if (slot < pair_capacity) {
                            tile_pair_i[slot] = home_tile;
                            tile_pair_j[slot] = nb_tile;
                        }
                    }
                }
            }
        }
    }
}

// Same search as is_excluded_pair() in force_calculator_cuda.cu: each atom's partner
// list is sorted and holds global ids.
__device__ __forceinline__ bool tile_list_is_excluded(const int* __restrict__ offsets,
                                                      const int* __restrict__ partners,
                                                      size_t atom_count,
                                                      int atom1, int atom2) {
    if (!offsets || !partners || atom1 < 0 ||
        static_cast<size_t>(atom1) >= atom_count) return false;
    const int begin = offsets[atom1];
    const int end = offsets[atom1 + 1];
    for (int index = begin; index < end; ++index) {
        const int candidate = partners[index];
        if (candidate == atom2) return true;
        if (candidate > atom2) break;
    }
    return false;
}

// Besides the Verlet mask this writes an exclusion mask with the same bit layout, so
// the force kernels test one bit instead of searching the partner list for every
// pair on every step. The partner lists and global ids do not change between list
// builds, so the mask stays valid for the life of the list.
__global__ void build_tile_pair_verlet_masks_kernel(
    const Real* __restrict__ x,
    const Real* __restrict__ y,
    const Real* __restrict__ z,
    const int* __restrict__ tile_atoms,
    const int* __restrict__ tile_pair_i,
    const int* __restrict__ tile_pair_j,
    unsigned long long* __restrict__ verlet_masks,
    unsigned long long* __restrict__ excl_masks,
    const int* __restrict__ global_id,
    const int* __restrict__ exclusion_offsets,
    const int* __restrict__ exclusion_partners,
    size_t exclusion_atom_count,
    unsigned long long* __restrict__ pair_counts,
    int n_local, int num_tile_pairs,
    Real cutoff_sq, Real outer_cutoff_sq,
    Real lx, Real ly, Real lz,
    bool periodic_x, bool periodic_y, bool periodic_z)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = e < num_tile_pairs;  // no early return: the warp reduces the counters below

    const int tile_i = active ? tile_pair_i[e] : 0;
    const int tile_j = active ? tile_pair_j[e] : 0;
    const bool same_tile = tile_i == tile_j;
    const bool reverse_entry = tile_i > tile_j;  // owned-in-home vs ghost-in-neighbor only
    unsigned long long mask = 0;
    unsigned long long excl = 0;
    unsigned long long inside_rc = 0;
    unsigned long long inside_outer = 0;

    // The distance test only has to separate pairs by the outer radius, which lies a whole
    // skin beyond the force cutoff, so it runs in float (the pair kernels subtract float
    // coordinates as well). The double version spent a division and a rounding in FP64 per
    // slot pair, 64 slot pairs per entry, and reloaded the j coordinates from global memory
    // for every i slot. The j side is loaded once per entry and kept in registers.
    const float lx_f = static_cast<float>(lx), ly_f = static_cast<float>(ly), lz_f = static_cast<float>(lz);
    const float inv_lx = 1.0f / lx_f, inv_ly = 1.0f / ly_f, inv_lz = 1.0f / lz_f;
    const float outer_sq_f = static_cast<float>(outer_cutoff_sq);
    const float rc_sq_f = static_cast<float>(cutoff_sq);

    int j_atom[LJ_TILE_SIZE], j_gid[LJ_TILE_SIZE];
    float jx[LJ_TILE_SIZE], jy[LJ_TILE_SIZE], jz[LJ_TILE_SIZE];
#pragma unroll
    for (int j_slot = 0; j_slot < LJ_TILE_SIZE; ++j_slot) {
        const int j = active ? tile_atoms[tile_j * LJ_TILE_SIZE + j_slot] : -1;
        j_atom[j_slot] = j;
        j_gid[j_slot] = 0;
        jx[j_slot] = jy[j_slot] = jz[j_slot] = 0.0f;
        if (j >= 0) {
            jx[j_slot] = static_cast<float>(x[j]);
            jy[j_slot] = static_cast<float>(y[j]);
            jz[j_slot] = static_cast<float>(z[j]);
            j_gid[j_slot] = global_id[j];
        }
    }

    for (int i_slot = 0; active && i_slot < LJ_TILE_SIZE; ++i_slot) {
        const int i = tile_atoms[tile_i * LJ_TILE_SIZE + i_slot];
        if (i < 0 || i >= n_local) continue;
        const float xi = static_cast<float>(x[i]);
        const float yi = static_cast<float>(y[i]);
        const float zi = static_cast<float>(z[i]);
        const int gid_i = global_id[i];

#pragma unroll
        for (int j_slot = 0; j_slot < LJ_TILE_SIZE; ++j_slot) {
            const int j = j_atom[j_slot];
            if (j < 0 || j == i) continue;
            if (same_tile && j < n_local && !(j > i)) continue;
            if (reverse_entry && j < n_local) continue;  // owned-owned pairs live in the forward entry

            float dx = xi - jx[j_slot];
            float dy = yi - jy[j_slot];
            float dz = zi - jz[j_slot];
            if (periodic_x) dx -= rintf(dx * inv_lx) * lx_f;
            if (periodic_y) dy -= rintf(dy * inv_ly) * ly_f;
            if (periodic_z) dz -= rintf(dz * inv_lz) * lz_f;
            const float r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < outer_sq_f) {
                mask |= 1ULL << (j_slot * LJ_TILE_SIZE + i_slot);
                if (tile_list_is_excluded(exclusion_offsets, exclusion_partners,
                                          exclusion_atom_count, gid_i, j_gid[j_slot])) {
                    excl |= 1ULL << (j_slot * LJ_TILE_SIZE + i_slot);
                }
                ++inside_outer;
                if (r2 < rc_sq_f) ++inside_rc;
            }
        }
    }

    if (active) {
        verlet_masks[e] = mask;
        excl_masks[e] = excl;
    }
    // Statistics only: one atomic per warp instead of two per thread (all on 2 addresses).
    for (int offset = 16; offset > 0; offset >>= 1) {
        inside_rc += __shfl_down_sync(0xffffffffu, inside_rc, offset);
        inside_outer += __shfl_down_sync(0xffffffffu, inside_outer, offset);
    }
    if ((threadIdx.x & 31) == 0) {
        if (inside_rc) atomicAdd(&pair_counts[0], inside_rc);
        if (inside_outer) atomicAdd(&pair_counts[1], inside_outer);
    }
}

// Bit 2 of the tile flags: this home tile has at least one pair-list entry whose
// neighbor tile holds a ghost atom. The ghost-only force pass skips every other
// home tile without walking its entries.
__global__ void mark_ghost_adjacent_tiles_kernel(
    const int* __restrict__ tile_pair_offsets,
    const int* __restrict__ tile_pair_j,
    unsigned char* __restrict__ tile_flags,
    int num_tiles)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tiles) return;
    unsigned char flags = tile_flags[t];
    if (flags & 1u) {
        for (int e = tile_pair_offsets[t]; e < tile_pair_offsets[t + 1]; ++e) {
            if (tile_flags[tile_pair_j[e]] & 2u) { flags |= 4u; break; }
        }
    }
    tile_flags[t] = flags;
}

namespace {
// Thrust allocates temporary storage with cudaMalloc and frees it with cudaFree on every
// sort/scan call; cudaFree synchronizes the device, and a tile-list build makes several
// such calls. This keeps the blocks and hands them out again. Sizes change from build to
// build, so a request takes any free block that is large enough but at most 4x too big
// (or at most 4 MB for the tiny scan buffers: every new block is at least 1 MB, so with a
// plain 4x rule a scan request never fits a block, allocates another one on every call
// and leaks it; those cudaMalloc calls also stalled for 0.3 s now and then), and new
// blocks are rounded up so later requests of similar size fit.
struct CachedThrustAllocator {
    typedef char value_type;
    char* allocate(std::ptrdiff_t bytes) {
        const size_t want = static_cast<size_t>(bytes);
        auto it = free_blocks().lower_bound(want);
        if (it != free_blocks().end() && it->first <= std::max<size_t>(4 * want, size_t(4) << 20)) {
            char* ptr = it->second;
            live_blocks()[ptr] = it->first;
            free_blocks().erase(it);
            return ptr;
        }
        size_t rounded = 1u << 20;
        while (rounded < want) rounded *= 2;
        char* ptr = nullptr;
        cudaMalloc(&ptr, rounded);
        live_blocks()[ptr] = rounded;
        return ptr;
    }
    void deallocate(char* ptr, size_t) {
        auto it = live_blocks().find(ptr);
        if (it == live_blocks().end()) return;
        free_blocks().emplace(it->second, ptr);
        live_blocks().erase(it);
    }
private:
    static std::multimap<size_t, char*>& free_blocks() { static std::multimap<size_t, char*> m; return m; }
    static std::map<char*, size_t>& live_blocks() { static std::map<char*, size_t> m; return m; }
};
}  // namespace

extern "C" void launch_cuda_tile_list_build(
    const Real* d_x, const Real* d_y, const Real* d_z,
    int n_total, int n_local,
    Real cutoff, Real verlet_skin,
    Real lx, Real ly, Real lz,
    bool periodic_x, bool periodic_y, bool periodic_z,
    cudaStream_t stream,
    const int* d_global_id,
    const int* d_exclusion_offsets,
    const int* d_exclusion_partners,
    size_t exclusion_atom_count,
    int** out_d_tile_atoms,
    int** out_d_tile_pair_i,
    int** out_d_tile_pair_j,
    int** out_d_tile_pair_offsets,
    unsigned long long** out_d_tile_pair_verlet_mask,
    unsigned long long** out_d_tile_pair_excl_mask,
    unsigned char** out_d_tile_flags,
    int* out_num_tiles,
    int* out_num_tile_pairs,
    unsigned long long* out_pairs_inside_rc,
    unsigned long long* out_pairs_inside_rc_plus_skin,
    bool tiles_only,
    TileGridInfo* grid_out)
{
    Real extended_cutoff = cutoff + verlet_skin;
    Real cutoff_sq = extended_cutoff * extended_cutoff;
    CachedThrustAllocator thrust_alloc;
    const auto thrust_policy = thrust::cuda::par(thrust_alloc).on(stream);
    const int MAX_CELLS_PER_DIM = 512;
    const int block_size = 256;
    const int num_blocks = (n_total + block_size - 1) / block_size;
    
    {
        const Real fine_cell_size_target = extended_cutoff / (Real)FINE_STENCIL_RADIUS;
        int fine_nx = std::max(1, (int)std::floor(lx / fine_cell_size_target));
        int fine_ny = std::max(1, (int)std::floor(ly / fine_cell_size_target));
        int fine_nz = std::max(1, (int)std::floor(lz / fine_cell_size_target));
        fine_nx = std::min(fine_nx, MAX_CELLS_PER_DIM);
        fine_ny = std::min(fine_ny, MAX_CELLS_PER_DIM);
        fine_nz = std::min(fine_nz, MAX_CELLS_PER_DIM);

        const bool fine_grid_size_ok =
            fine_nx >= FINE_STENCIL_SIDE && fine_ny >= FINE_STENCIL_SIDE && fine_nz >= FINE_STENCIL_SIDE;
        const bool fine_grid_coverage_ok = fine_grid_size_ok &&
            (Real)FINE_STENCIL_RADIUS * (lx / fine_nx) >= extended_cutoff &&
            (Real)FINE_STENCIL_RADIUS * (ly / fine_ny) >= extended_cutoff &&
            (Real)FINE_STENCIL_RADIUS * (lz / fine_nz) >= extended_cutoff;
        const bool fine_grid_viable = fine_grid_size_ok && fine_grid_coverage_ok;

        if (!fine_grid_viable) {
            if (out_d_tile_atoms)        *out_d_tile_atoms        = nullptr;
            if (out_d_tile_pair_i)       *out_d_tile_pair_i       = nullptr;
            if (out_d_tile_pair_j)       *out_d_tile_pair_j       = nullptr;
            if (out_d_tile_pair_offsets) *out_d_tile_pair_offsets = nullptr;
            if (out_d_tile_pair_verlet_mask) *out_d_tile_pair_verlet_mask = nullptr;
            if (out_d_tile_pair_excl_mask) *out_d_tile_pair_excl_mask = nullptr;
            if (out_d_tile_flags)        *out_d_tile_flags        = nullptr;
            if (out_num_tiles)           *out_num_tiles           = 0;
            if (out_num_tile_pairs)      *out_num_tile_pairs      = 0;
            if (out_pairs_inside_rc)      *out_pairs_inside_rc      = 0;
            if (out_pairs_inside_rc_plus_skin) *out_pairs_inside_rc_plus_skin = 0;
            return;
        }

        Real fine_cell_size_x = lx / fine_nx;
        Real fine_cell_size_y = ly / fine_ny;
        Real fine_cell_size_z = lz / fine_nz;
        Real fine_inv_cell_x = fine_nx / lx;
        Real fine_inv_cell_y = fine_ny / ly;
        Real fine_inv_cell_z = fine_nz / lz;
        long long fine_total_cells = (long long)fine_nx * fine_ny * fine_nz;

        // Pinned landing zone for the scalar readbacks below: [0],[1] tile-count scalars /
        // tile-pair count, [2..5] the two 64-bit pair statistics. A copy into a pageable
        // stack variable is staged through a driver buffer and blocks inside the call.
        static int* s_h_scalars = nullptr;
        if (!s_h_scalars) cudaMallocHost(&s_h_scalars, 6 * sizeof(int));

        static int* s_d_fine_cell_start   = nullptr;
        static int* s_d_fine_cell_count   = nullptr;
        static int* s_d_fine_particle_cell_id = nullptr;
        static int* s_d_fine_cell_particles   = nullptr;
        static int* s_d_fine_cell_counters    = nullptr;
        static long long s_fine_total_cells_capacity = 0;
        static int s_fine_n_total_capacity = 0;

        if (fine_total_cells > s_fine_total_cells_capacity) {
            long long new_capacity = (s_fine_total_cells_capacity > 0)
                ? std::max(fine_total_cells, s_fine_total_cells_capacity + s_fine_total_cells_capacity / 2)
                : fine_total_cells;
            if (s_d_fine_cell_start)    cudaFree(s_d_fine_cell_start);
            if (s_d_fine_cell_count)    cudaFree(s_d_fine_cell_count);
            if (s_d_fine_cell_counters) cudaFree(s_d_fine_cell_counters);
            cudaMalloc(&s_d_fine_cell_start,    new_capacity * sizeof(int));
            cudaMalloc(&s_d_fine_cell_count,    new_capacity * sizeof(int));
            cudaMalloc(&s_d_fine_cell_counters, new_capacity * sizeof(int));
            s_fine_total_cells_capacity = new_capacity;
        }
        if (n_total > s_fine_n_total_capacity) {
            int new_capacity = (s_fine_n_total_capacity > 0)
                ? std::max(n_total, s_fine_n_total_capacity + s_fine_n_total_capacity / 2)
                : n_total;
            if (s_d_fine_particle_cell_id) cudaFree(s_d_fine_particle_cell_id);
            if (s_d_fine_cell_particles)   cudaFree(s_d_fine_cell_particles);
            cudaMalloc(&s_d_fine_particle_cell_id, (size_t)new_capacity * sizeof(int));
            cudaMalloc(&s_d_fine_cell_particles,   (size_t)new_capacity * sizeof(int));
            s_fine_n_total_capacity = new_capacity;
        }

        cudaMemsetAsync(s_d_fine_cell_count,    0, (size_t)fine_total_cells * sizeof(int), stream);
        cudaMemsetAsync(s_d_fine_cell_counters, 0, (size_t)fine_total_cells * sizeof(int), stream);

        build_cell_list_kernel<<<num_blocks, block_size, 0, stream>>>(
            d_x, d_y, d_z,
            s_d_fine_cell_start, s_d_fine_cell_count, s_d_fine_particle_cell_id,
            n_total,
            fine_inv_cell_x, fine_inv_cell_y, fine_inv_cell_z,
            fine_nx, fine_ny, fine_nz,
            lx, ly, lz,
            periodic_x, periodic_y, periodic_z
        );
        NEIGHBOR_LIST_CUDA_CHECK(cudaGetLastError());

        thrust::device_ptr<int> fine_cnt_ptr(s_d_fine_cell_count);
        thrust::device_ptr<int> fine_start_ptr(s_d_fine_cell_start);
        thrust::exclusive_scan(thrust_policy, fine_cnt_ptr, fine_cnt_ptr + fine_total_cells, fine_start_ptr);

        populate_cells_kernel<<<num_blocks, block_size, 0, stream>>>(
            s_d_fine_particle_cell_id,
            s_d_fine_cell_start, s_d_fine_cell_particles, s_d_fine_cell_counters,
            n_total
        );
        NEIGHBOR_LIST_CUDA_CHECK(cudaGetLastError());

        static unsigned long long* s_d_sort_keys = nullptr;
        static int* s_d_tile_sorted_indices = nullptr;
        static long long s_sort_capacity = 0;

        if (n_total > s_sort_capacity) {
            long long new_capacity = (s_sort_capacity > 0)
                ? std::max((long long)n_total, s_sort_capacity + s_sort_capacity / 2)
                : (long long)n_total;
            if (s_d_sort_keys)           cudaFree(s_d_sort_keys);
            if (s_d_tile_sorted_indices) cudaFree(s_d_tile_sorted_indices);
            cudaMalloc(&s_d_sort_keys,           new_capacity * sizeof(unsigned long long));
            cudaMalloc(&s_d_tile_sorted_indices, new_capacity * sizeof(int));
            s_sort_capacity = new_capacity;
        }

        compute_tile_sort_keys_kernel<<<num_blocks, block_size, 0, stream>>>(
            d_x, d_y, d_z, s_d_fine_particle_cell_id, s_d_sort_keys, n_total,
            fine_nx, fine_ny, fine_cell_size_x, fine_cell_size_y, fine_cell_size_z,
            lx, ly, lz, periodic_x, periodic_y, periodic_z);
        NEIGHBOR_LIST_CUDA_CHECK(cudaGetLastError());

        thrust::sequence(thrust_policy,
                          thrust::device_pointer_cast(s_d_tile_sorted_indices),
                          thrust::device_pointer_cast(s_d_tile_sorted_indices + n_total));
        thrust::sort_by_key(thrust_policy,
                             thrust::device_pointer_cast(s_d_sort_keys),
                             thrust::device_pointer_cast(s_d_sort_keys + n_total),
                             thrust::device_pointer_cast(s_d_tile_sorted_indices));

        static int* s_d_tile_count = nullptr;
        static int* s_d_tile_offset = nullptr;
        static long long s_tile_cells_capacity = 0;

        static int* s_d_tile_atoms = nullptr;
        static int* s_d_tile_cell_id = nullptr;
        static Real* s_d_tile_bbox_min = nullptr;
        static Real* s_d_tile_bbox_max = nullptr;
        static unsigned char* s_d_tile_flags = nullptr;
        static unsigned char* s_d_cell_has_ghost = nullptr;
        static long long s_cell_ghost_capacity = 0;
        static long long s_tile_atoms_capacity = 0;

        static int* s_d_tile_pair_i = nullptr;
        static int* s_d_tile_pair_j = nullptr;
        static unsigned long long* s_d_tile_pair_verlet_mask = nullptr;
        static unsigned long long* s_d_tile_pair_excl_mask = nullptr;
        static long long s_tile_pairs_capacity = 0;

        static int* s_d_tile_pair_offsets = nullptr;
        static long long s_tile_pair_offsets_capacity = 0;

        static int* s_d_pair_counter = nullptr;
        static unsigned long long* s_d_verlet_pair_counts = nullptr;

        if (fine_total_cells > s_tile_cells_capacity) {
            if (s_d_tile_count)  cudaFree(s_d_tile_count);
            if (s_d_tile_offset) cudaFree(s_d_tile_offset);
            cudaMalloc(&s_d_tile_count,  fine_total_cells * sizeof(int));
            cudaMalloc(&s_d_tile_offset, fine_total_cells * sizeof(int));
            s_tile_cells_capacity = fine_total_cells;
        }

        int tile_block = 256;
        int tile_grid = (int)((fine_total_cells + tile_block - 1) / tile_block);
        compute_cell_tile_counts_kernel<<<tile_grid, tile_block, 0, stream>>>(
            s_d_fine_cell_count, s_d_tile_count, fine_total_cells);
        NEIGHBOR_LIST_CUDA_CHECK(cudaGetLastError());

        thrust::device_ptr<int> tcount_ptr(s_d_tile_count);
        thrust::device_ptr<int> toffset_ptr(s_d_tile_offset);
        thrust::exclusive_scan(thrust_policy, tcount_ptr, tcount_ptr + fine_total_cells, toffset_ptr);

        cudaMemcpyAsync(s_h_scalars,     s_d_tile_offset + (fine_total_cells - 1), sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(s_h_scalars + 1, s_d_tile_count  + (fine_total_cells - 1), sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        int total_tiles = std::max(0, static_cast<int>(s_h_scalars[0]) + static_cast<int>(s_h_scalars[1]));

        long long atoms_needed = (long long)total_tiles * LJ_TILE_SIZE;
        if (atoms_needed > s_tile_atoms_capacity) {
            if (s_d_tile_atoms)    cudaFree(s_d_tile_atoms);
            if (s_d_tile_cell_id)  cudaFree(s_d_tile_cell_id);
            if (s_d_tile_bbox_min) cudaFree(s_d_tile_bbox_min);
            if (s_d_tile_bbox_max) cudaFree(s_d_tile_bbox_max);
            if (s_d_tile_flags)    cudaFree(s_d_tile_flags);
            // 25% headroom: the tile count drifts as atoms move between cells, and a
            // reallocation (cudaFree synchronizes) in the middle of a run stalls the step.
            long long new_cap = std::max(atoms_needed + atoms_needed / 4,
                                         s_tile_atoms_capacity + s_tile_atoms_capacity / 2);
            long long new_tiles_cap = new_cap / LJ_TILE_SIZE;
            cudaMalloc(&s_d_tile_atoms, new_cap * sizeof(int));
            cudaMalloc(&s_d_tile_cell_id, new_tiles_cap * sizeof(int));
            cudaMalloc(&s_d_tile_bbox_min, new_tiles_cap * 3 * sizeof(Real));
            cudaMalloc(&s_d_tile_bbox_max, new_tiles_cap * 3 * sizeof(Real));
            cudaMalloc(&s_d_tile_flags,    new_tiles_cap * sizeof(unsigned char));
            s_tile_atoms_capacity = new_cap;
        }
        if (atoms_needed > 0) {
            cudaMemsetAsync(s_d_tile_atoms, 0xFF, atoms_needed * sizeof(int), stream);
            int scat_block = 256;
            int scat_grid = (n_total + scat_block - 1) / scat_block;
            scatter_lj_tile_atoms_kernel<<<scat_grid, scat_block, 0, stream>>>(
                s_d_tile_sorted_indices, s_d_fine_particle_cell_id, s_d_fine_cell_start, s_d_tile_offset,
                s_d_tile_atoms, n_total);
            NEIGHBOR_LIST_CUDA_CHECK(cudaGetLastError());

            scatter_tile_cell_id_kernel<<<tile_grid, tile_block, 0, stream>>>(
                s_d_tile_offset, s_d_tile_count, s_d_tile_cell_id, fine_total_cells);
            NEIGHBOR_LIST_CUDA_CHECK(cudaGetLastError());

            const int WARP_SIZE_B = 32;
            const int WARPS_PER_BLOCK_B = 256 / WARP_SIZE_B;
            int threads_per_block_b = WARPS_PER_BLOCK_B * WARP_SIZE_B;
            int num_blocks_b = (total_tiles + WARPS_PER_BLOCK_B - 1) / WARPS_PER_BLOCK_B;
            compute_tile_bboxes_kernel<<<num_blocks_b, threads_per_block_b, 0, stream>>>(
                d_x, d_y, d_z, s_d_tile_atoms, s_d_tile_cell_id,
                fine_nx, fine_ny, fine_cell_size_x, fine_cell_size_y, fine_cell_size_z,
                lx, ly, lz, periodic_x, periodic_y, periodic_z,
                s_d_tile_bbox_min, s_d_tile_bbox_max, s_d_tile_flags, n_local, total_tiles);
            NEIGHBOR_LIST_CUDA_CHECK(cudaGetLastError());
        }

        // With ghost atoms the list builder also needs to know which cells hold a ghost tile.
        const bool want_ghost_cells = tiles_only && n_total > n_local && total_tiles > 0;
        if (want_ghost_cells) {
            if (fine_total_cells > s_cell_ghost_capacity) {
                if (s_d_cell_has_ghost) cudaFree(s_d_cell_has_ghost);
                cudaMalloc(&s_d_cell_has_ghost, static_cast<size_t>(fine_total_cells));
                s_cell_ghost_capacity = fine_total_cells;
            }
            cudaMemsetAsync(s_d_cell_has_ghost, 0, static_cast<size_t>(fine_total_cells), stream);
            mark_ghost_cells_kernel<<<(total_tiles + 255) / 256, 256, 0, stream>>>(
                s_d_tile_flags, s_d_tile_cell_id, s_d_cell_has_ghost, total_tiles);
            NEIGHBOR_LIST_CUDA_CHECK(cudaGetLastError());
        }

        if (tiles_only) {
            // The caller builds its own lists from the cell grid and the tiles.
            if (out_d_tile_atoms)        *out_d_tile_atoms        = s_d_tile_atoms;
            if (out_d_tile_pair_i)       *out_d_tile_pair_i       = nullptr;
            if (out_d_tile_pair_j)       *out_d_tile_pair_j       = nullptr;
            if (out_d_tile_pair_offsets) *out_d_tile_pair_offsets = nullptr;
            if (out_d_tile_pair_verlet_mask) *out_d_tile_pair_verlet_mask = nullptr;
            if (out_d_tile_pair_excl_mask) *out_d_tile_pair_excl_mask = nullptr;
            if (out_d_tile_flags)        *out_d_tile_flags        = s_d_tile_flags;
            if (out_num_tiles)           *out_num_tiles           = total_tiles;
            if (out_num_tile_pairs)      *out_num_tile_pairs      = 0;
            if (out_pairs_inside_rc)      *out_pairs_inside_rc      = 0;
            if (out_pairs_inside_rc_plus_skin) *out_pairs_inside_rc_plus_skin = 0;
            if (grid_out) {
                grid_out->tile_offset = s_d_tile_offset;
                grid_out->tile_count = s_d_tile_count;
                grid_out->tile_cell_id = s_d_tile_cell_id;
                grid_out->tile_bbox_min = s_d_tile_bbox_min;
                grid_out->tile_bbox_max = s_d_tile_bbox_max;
                grid_out->nx = fine_nx;
                grid_out->ny = fine_ny;
                grid_out->nz = fine_nz;
                grid_out->cell_size_x = fine_cell_size_x;
                grid_out->cell_size_y = fine_cell_size_y;
                grid_out->cell_size_z = fine_cell_size_z;
                grid_out->stencil_radius = FINE_STENCIL_RADIUS;
                grid_out->cell_has_ghost = want_ghost_cells ? s_d_cell_has_ghost : nullptr;
            }
            return;
        }

        if (!s_d_pair_counter) cudaMalloc(&s_d_pair_counter, sizeof(int));

        const int WARP_SIZE_T = 32;
        const int WARPS_PER_BLOCK_T = 256 / WARP_SIZE_T;
        int threads_per_block_t = WARPS_PER_BLOCK_T * WARP_SIZE_T;
        int num_blocks_t = (int)((fine_total_cells + WARPS_PER_BLOCK_T - 1) / WARPS_PER_BLOCK_T);

        int host_pair_count = 0;
        for (int attempt = 0; attempt < 2; ++attempt) {
            cudaMemsetAsync(s_d_pair_counter, 0, sizeof(int), stream);
            int capacity = (int)std::min<long long>(s_tile_pairs_capacity, (long long)INT_MAX);
            if (total_tiles > 0) {
                build_lj_tile_pairs_fine_kernel<<<num_blocks_t, threads_per_block_t, 0, stream>>>(
                    s_d_tile_offset, s_d_tile_count, s_d_tile_bbox_min, s_d_tile_bbox_max,
                    s_d_tile_flags,
                    s_d_tile_pair_i, s_d_tile_pair_j, s_d_pair_counter, capacity,
                    fine_nx, fine_ny, fine_nz,
                    fine_cell_size_x, fine_cell_size_y, fine_cell_size_z,
                    periodic_x, periodic_y, periodic_z,
                    cutoff_sq, fine_total_cells);
                NEIGHBOR_LIST_CUDA_CHECK(cudaGetLastError());
            }
            cudaMemcpyAsync(s_h_scalars, s_d_pair_counter, sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            host_pair_count = static_cast<int>(s_h_scalars[0]);

            if (host_pair_count <= s_tile_pairs_capacity) break;

            if (s_d_tile_pair_i) cudaFree(s_d_tile_pair_i);
            if (s_d_tile_pair_j) cudaFree(s_d_tile_pair_j);
            if (s_d_tile_pair_verlet_mask) cudaFree(s_d_tile_pair_verlet_mask);
            if (s_d_tile_pair_excl_mask) cudaFree(s_d_tile_pair_excl_mask);
            s_tile_pairs_capacity = (long long)(host_pair_count * 1.25) + 64;
            cudaMalloc(&s_d_tile_pair_i, s_tile_pairs_capacity * sizeof(int));
            cudaMalloc(&s_d_tile_pair_j, s_tile_pairs_capacity * sizeof(int));
            cudaMalloc(&s_d_tile_pair_verlet_mask,
                       s_tile_pairs_capacity * sizeof(unsigned long long));
            cudaMalloc(&s_d_tile_pair_excl_mask,
                       s_tile_pairs_capacity * sizeof(unsigned long long));
        }

        if (total_tiles + 1 > s_tile_pair_offsets_capacity) {
            if (s_d_tile_pair_offsets) cudaFree(s_d_tile_pair_offsets);
            s_tile_pair_offsets_capacity = (long long)(total_tiles + 1);
            cudaMalloc(&s_d_tile_pair_offsets, s_tile_pair_offsets_capacity * sizeof(int));
        }
        if (total_tiles > 0 && host_pair_count > 0) {
            thrust::device_ptr<int> key_ptr(s_d_tile_pair_i);
            thrust::device_ptr<int> val_ptr(s_d_tile_pair_j);
            thrust::sort_by_key(thrust_policy, key_ptr, key_ptr + host_pair_count, val_ptr);

            thrust::counting_iterator<int> search_begin(0);
            thrust::device_ptr<int> offsets_ptr(s_d_tile_pair_offsets);
            thrust::lower_bound(thrust_policy, key_ptr, key_ptr + host_pair_count,
                                 search_begin, search_begin + (total_tiles + 1), offsets_ptr);
        } else if (total_tiles > 0) {
            cudaMemsetAsync(s_d_tile_pair_offsets, 0, s_tile_pair_offsets_capacity * sizeof(int), stream);
        }
        if (total_tiles > 0 && host_pair_count > 0) {
            mark_ghost_adjacent_tiles_kernel<<<(total_tiles + 255) / 256, 256, 0, stream>>>(
                s_d_tile_pair_offsets, s_d_tile_pair_j, s_d_tile_flags, total_tiles);
            NEIGHBOR_LIST_CUDA_CHECK(cudaGetLastError());
        }

        unsigned long long host_pair_counts[2] = {0, 0};
        if (!s_d_verlet_pair_counts) {
            cudaMalloc(&s_d_verlet_pair_counts, 2 * sizeof(unsigned long long));
        }
        cudaMemsetAsync(s_d_verlet_pair_counts, 0, 2 * sizeof(unsigned long long), stream);
        if (host_pair_count > 0) {
            const int mask_block = 128;
            const int mask_grid = (host_pair_count + mask_block - 1) / mask_block;
            build_tile_pair_verlet_masks_kernel<<<mask_grid, mask_block, 0, stream>>>(
                d_x, d_y, d_z, s_d_tile_atoms, s_d_tile_pair_i, s_d_tile_pair_j,
                s_d_tile_pair_verlet_mask, s_d_tile_pair_excl_mask,
                d_global_id, d_exclusion_offsets, d_exclusion_partners, exclusion_atom_count,
                s_d_verlet_pair_counts,
                n_local, host_pair_count, cutoff * cutoff, cutoff_sq,
                lx, ly, lz, periodic_x, periodic_y, periodic_z);
            NEIGHBOR_LIST_CUDA_CHECK(cudaGetLastError());
        }
        // The inside-cutoff counts are statistics only. Reading them back costs a full
        // stream sync right after the (long) mask kernel, so the outputs stay 0 and the
        // host moves on.

        if (out_d_tile_atoms)        *out_d_tile_atoms        = s_d_tile_atoms;
        if (out_d_tile_pair_i)       *out_d_tile_pair_i       = s_d_tile_pair_i;
        if (out_d_tile_pair_j)       *out_d_tile_pair_j       = s_d_tile_pair_j;
        if (out_d_tile_pair_offsets) *out_d_tile_pair_offsets = (total_tiles > 0) ? s_d_tile_pair_offsets : nullptr;
        if (out_d_tile_pair_verlet_mask) *out_d_tile_pair_verlet_mask = s_d_tile_pair_verlet_mask;
        if (out_d_tile_pair_excl_mask) *out_d_tile_pair_excl_mask = s_d_tile_pair_excl_mask;
        if (out_d_tile_flags)        *out_d_tile_flags        = s_d_tile_flags;
        if (out_num_tiles)           *out_num_tiles           = total_tiles;
        if (out_num_tile_pairs)      *out_num_tile_pairs      = host_pair_count;
        if (out_pairs_inside_rc)      *out_pairs_inside_rc      = host_pair_counts[0];
        if (out_pairs_inside_rc_plus_skin) *out_pairs_inside_rc_plus_skin = host_pair_counts[1];
    }
}
