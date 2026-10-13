#include "neighbor_list.h"
#include "md_types.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <mpi.h>
#include "pme_rank_split.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

struct TilePairRec {
    int home;
    int nb;
    Real sx, sy, sz;
};

inline Real aabb_min_distance_sq(Real amin_x, Real amin_y, Real amin_z,
                                 Real amax_x, Real amax_y, Real amax_z,
                                 Real bmin_x, Real bmin_y, Real bmin_z,
                                 Real bmax_x, Real bmax_y, Real bmax_z,
                                 Real shift_x, Real shift_y, Real shift_z) {
    bmin_x += shift_x; bmax_x += shift_x;
    bmin_y += shift_y; bmax_y += shift_y;
    bmin_z += shift_z; bmax_z += shift_z;

    const Real gap_x = std::max<Real>(0.0, std::max(amin_x - bmax_x, bmin_x - amax_x));
    const Real gap_y = std::max<Real>(0.0, std::max(amin_y - bmax_y, bmin_y - amax_y));
    const Real gap_z = std::max<Real>(0.0, std::max(amin_z - bmax_z, bmin_z - amax_z));
    return gap_x * gap_x + gap_y * gap_y + gap_z * gap_z;
}

void kd_split_atoms(int* idx, int begin, int end,
                    const Real* px, const Real* py, const Real* pz) {
    const int n = end - begin;
    if (n <= kCpuTileSize) return;

    Real minx = std::numeric_limits<Real>::max(), maxx = -std::numeric_limits<Real>::max();
    Real miny = minx, maxy = maxx, minz = minx, maxz = maxx;
    for (int k = begin; k < end; ++k) {
        const int a = idx[k];
        minx = std::min(minx, px[a]); maxx = std::max(maxx, px[a]);
        miny = std::min(miny, py[a]); maxy = std::max(maxy, py[a]);
        minz = std::min(minz, pz[a]); maxz = std::max(maxz, pz[a]);
    }
    const Real ex = maxx - minx, ey = maxy - miny, ez = maxz - minz;

    const int n_groups = (n + kCpuTileSize - 1) / kCpuTileSize;
    const int left_groups = n_groups / 2;
    const int split = left_groups * kCpuTileSize;

    if (ex >= ey && ex >= ez) {
        std::nth_element(idx + begin, idx + begin + split, idx + end,
                         [px](int a, int b) { return px[a] < px[b]; });
    } else if (ey >= ez) {
        std::nth_element(idx + begin, idx + begin + split, idx + end,
                         [py](int a, int b) { return py[a] < py[b]; });
    } else {
        std::nth_element(idx + begin, idx + begin + split, idx + end,
                         [pz](int a, int b) { return pz[a] < pz[b]; });
    }

    kd_split_atoms(idx, begin, begin + split, px, py, pz);
    kd_split_atoms(idx, begin + split, end, px, py, pz);
}

}  // namespace

SpatialVerletList::SpatialVerletList(Real cutoff, Real skin)
    : cutoff_(cutoff), skin_(skin) {
    max_displacement_threshold_ = (0.5 * skin_) * (0.5 * skin_);
    active_skin_ = 0.5 * skin_;
}

void SpatialVerletList::update(const ParticleData& particles, const SimulationBox& box, size_t num_local) {
    build(particles, box, num_local);
}

void SpatialVerletList::prune_active(const ParticleData& particles,
                                        const SimulationBox& box,
                                        size_t num_local) {
    const size_t local_n = (num_local > 0) ? num_local : particles.num_particles;
    const size_t total_n = particles.num_particles;
    active_degree_.assign(local_n, 0);
    const Real active_cutoff = cutoff_ + active_skin_;
    const Real cutoff_sq = active_cutoff * active_cutoff;
    const bool mpi_box = box.is_mpi_subdomain();

    auto is_active = [&](size_t i, int j) {
        if (j < 0 || static_cast<size_t>(j) >= total_n) return false;
        Real dx = particles.x[i] - particles.x[static_cast<size_t>(j)];
        Real dy = particles.y[i] - particles.y[static_cast<size_t>(j)];
        Real dz = particles.z[i] - particles.z[static_cast<size_t>(j)];
        if (!mpi_box || static_cast<size_t>(j) < local_n) {
            box.apply_minimum_image(dx, dy, dz);
        }
        return dx * dx + dy * dy + dz * dz < cutoff_sq;
    };

    for (size_t i = 0; i < local_n; ++i) {
        for (int e = offsets_[i]; e < offsets_[i + 1]; ++e) {
            const int j = neighbors_[static_cast<size_t>(e)];
            if (!is_active(i, j)) continue;
            ++active_degree_[i];
            if (j >= 0 && static_cast<size_t>(j) < local_n) {
                ++active_degree_[static_cast<size_t>(j)];
            }
        }
    }

    active_offsets_.assign(local_n + 1, 0);
    for (size_t i = 0; i < local_n; ++i) {
        active_offsets_[i + 1] = active_offsets_[i] + active_degree_[i];
    }
    active_neighbors_.resize(static_cast<size_t>(active_offsets_[local_n]));
    std::vector<int> cursor(active_offsets_.begin(), active_offsets_.end() - 1);

    for (size_t i = 0; i < local_n; ++i) {
        for (int e = offsets_[i]; e < offsets_[i + 1]; ++e) {
            const int j = neighbors_[static_cast<size_t>(e)];
            if (!is_active(i, j)) continue;
            active_neighbors_[static_cast<size_t>(cursor[i]++)] = j;
            if (j >= 0 && static_cast<size_t>(j) < local_n) {
                active_neighbors_[static_cast<size_t>(cursor[static_cast<size_t>(j)]++)] =
                    static_cast<int>(i);
            }
        }
    }

    last_prune_x_.assign(particles.x.begin(), particles.x.end());
    last_prune_y_.assign(particles.y.begin(), particles.y.end());
    last_prune_z_.assign(particles.z.begin(), particles.z.end());
}

bool SpatialVerletList::needs_active_prune(
    const ParticleData& particles, const SimulationBox& box, size_t /*num_local*/) const {
    if (!build_active_graph_) return false;
    if (last_prune_x_.size() != particles.num_particles || last_prune_x_.empty()) return true;

    Real local_max_sq = 0.0;
    for (size_t i = 0; i < particles.num_particles; ++i) {
        Real dx = particles.x[i] - last_prune_x_[i];
        Real dy = particles.y[i] - last_prune_y_[i];
        Real dz = particles.z[i] - last_prune_z_[i];
        box.apply_minimum_image(dx, dy, dz);
        local_max_sq = std::max(local_max_sq, dx * dx + dy * dy + dz * dz);
    }
    Real global_max_sq = local_max_sq;
    int mpi_init = 0;
    MPI_Initialized(&mpi_init);
    if (mpi_init) {
        MPI_Allreduce(&local_max_sq, &global_max_sq, 1, MPI_DOUBLE, MPI_MAX,
                      PmePpSplit::instance().pp_comm());
    }
    const Real threshold = 0.5 * active_skin_;
    return global_max_sq >= threshold * threshold;
}

namespace {
constexpr Real kForceRebuildSentinel = -1.0;
}

Real SpatialVerletList::compute_local_max_displacement_sq(
    const ParticleData& particles, const SimulationBox& box, size_t num_local) const {
    if (last_build_x_.empty()) return kForceRebuildSentinel;

    size_t cached = last_build_x_.size();
    size_t check_n = num_local;
    if (num_local != cached) {
        Real frac = (num_local > cached)
            ? static_cast<Real>(num_local - cached) / static_cast<Real>(cached)
            : static_cast<Real>(cached - num_local) / static_cast<Real>(cached);
        if (frac > 0.02) return kForceRebuildSentinel;
        check_n = std::min(num_local, cached);
    }

    Real local_max_disp_sq = 0.0;
    for (size_t i = 0; i < check_n; ++i) {
        Real dx = particles.x[i] - last_build_x_[i];
        Real dy = particles.y[i] - last_build_y_[i];
        Real dz = particles.z[i] - last_build_z_[i];
        box.apply_minimum_image(dx, dy, dz);
        Real dsq = dx * dx + dy * dy + dz * dz;
        local_max_disp_sq = std::max(local_max_disp_sq, dsq);
    }
    return local_max_disp_sq;
}

bool SpatialVerletList::needs_rebuild(const ParticleData& particles, const SimulationBox& box, size_t num_local) const {
    Real local_max_disp_sq = compute_local_max_displacement_sq(particles, box, num_local);
    int local_rebuild = (local_max_disp_sq < 0.0 || local_max_disp_sq > max_displacement_threshold_) ? 1 : 0;
    int global_rebuild = local_rebuild;
    int mpi_init = 0;
    MPI_Initialized(&mpi_init);
    if (mpi_init) {
        MPI_Allreduce(&local_rebuild, &global_rebuild, 1, MPI_INT, MPI_MAX, PmePpSplit::instance().pp_comm());
    }
    return global_rebuild > 0;
}

void SpatialVerletList::build(const ParticleData& particles, const SimulationBox& box, size_t num_local) {
    ++build_generation_;
    const size_t local_N = (num_local > 0) ? num_local : particles.num_particles;
    const size_t total_N = particles.num_particles;
    const bool has_ghosts = !safe_to_use_half_shell(local_N, total_N);

    const Real extended_cutoff = cutoff_ + skin_;
    const Real rcut2 = extended_cutoff * extended_cutoff;

    const bool split_x = box.is_split_x();
    const bool split_y = box.is_split_y();
    const bool split_z = box.is_split_z();
    const bool is_mpi_subdomain = split_x || split_y || split_z;
    const Real origin_x = (box.xhi > box.xlo) ? box.xlo : 0.0;
    const Real origin_y = (box.yhi > box.ylo) ? box.ylo : 0.0;
    const Real origin_z = (box.zhi > box.zlo) ? box.zlo : 0.0;
    // Margin is per-axis: only an MPI-split axis actually receives ghost
    // atoms from dsDecompX to fill it. An unsplit axis must keep its cell
    // grid tiling exactly one full periodic length (like the fully-serial
    // !is_mpi_subdomain case) so the xc+=nx/xc-=nx wrap below lands on the
    // true periodic image -- padding it with a margin here would offset the
    // grid origin and break that wrap arithmetic.
    const Real ghost_margin_x = split_x ? extended_cutoff : 0.0;
    const Real ghost_margin_y = split_y ? extended_cutoff : 0.0;
    const Real ghost_margin_z = split_z ? extended_cutoff : 0.0;

    const Real Lx_sub = (box.xhi > box.xlo) ? (box.xhi - box.xlo) : box.lx;
    const Real Ly_sub = (box.yhi > box.ylo) ? (box.yhi - box.ylo) : box.ly;
    const Real Lz_sub = (box.zhi > box.zlo) ? (box.zhi - box.zlo) : box.lz;

    const Real grid_origin_x = origin_x - ghost_margin_x;
    const Real grid_origin_y = origin_y - ghost_margin_y;
    const Real grid_origin_z = origin_z - ghost_margin_z;
    const Real Lx = Lx_sub + 2.0 * ghost_margin_x;
    const Real Ly = Ly_sub + 2.0 * ghost_margin_y;
    const Real Lz = Lz_sub + 2.0 * ghost_margin_z;

    const Real target_bin_size = extended_cutoff / 2.0;
    const int nx = std::max(1, static_cast<int>(Lx / target_bin_size));
    const int ny = std::max(1, static_cast<int>(Ly / target_bin_size));
    const int nz = std::max(1, static_cast<int>(Lz / target_bin_size));

    const Real bin_x = Lx / nx, bin_y = Ly / ny, bin_z = Lz / nz;
    const Real max_bin = std::max({bin_x, bin_y, bin_z});
    const int stencil_range = std::max(1, static_cast<int>(std::ceil(extended_cutoff / max_bin)));

    const Real inv_bin_x = nx / Lx, inv_bin_y = ny / Ly, inv_bin_z = nz / Lz;
    const int total_cells = nx * ny * nz;

    cell_of_.resize(total_N);
    std::vector<int> cix(total_N), ciy(total_N), ciz(total_N);
    #pragma omp parallel for schedule(static)
    for (long ii = 0; ii < static_cast<long>(total_N); ++ii) {
        size_t i = static_cast<size_t>(ii);
        Real rx = particles.x[i] - grid_origin_x;
        Real ry = particles.y[i] - grid_origin_y;
        Real rz = particles.z[i] - grid_origin_z;
        int ix = std::clamp(static_cast<int>(rx * inv_bin_x), 0, nx - 1);
        int iy = std::clamp(static_cast<int>(ry * inv_bin_y), 0, ny - 1);
        int iz = std::clamp(static_cast<int>(rz * inv_bin_z), 0, nz - 1);
        cix[i] = ix;
        ciy[i] = iy;
        ciz[i] = iz;
        cell_of_[i] = iz * nx * ny + iy * nx + ix;
    }

    std::vector<int> cell_count(static_cast<size_t>(total_cells), 0);
    for (size_t i = 0; i < total_N; ++i) ++cell_count[static_cast<size_t>(cell_of_[i])];
    cell_start_.assign(static_cast<size_t>(total_cells) + 1, 0);
    for (int c = 0; c < total_cells; ++c) {
        cell_start_[static_cast<size_t>(c) + 1] = cell_start_[static_cast<size_t>(c)] + cell_count[static_cast<size_t>(c)];
    }

    std::vector<int> cell_fill(static_cast<size_t>(total_cells), 0);
    std::vector<int> cell_atoms(total_N);
    for (size_t i = 0; i < total_N; ++i) {
        int c = cell_of_[i];
        cell_atoms[static_cast<size_t>(cell_start_[static_cast<size_t>(c)] + cell_fill[static_cast<size_t>(c)])] =
            static_cast<int>(i);
        ++cell_fill[static_cast<size_t>(c)];
    }

    std::vector<int> sdx, sdy, sdz;
    const int sr = stencil_range;
    sdx.reserve(static_cast<size_t>((2 * sr + 1) * (2 * sr + 1) * (2 * sr + 1)));
    sdy.reserve(sdx.capacity());
    sdz.reserve(sdx.capacity());
    for (int dz = -sr; dz <= sr; ++dz)
        for (int dy = -sr; dy <= sr; ++dy)
            for (int dx = -sr; dx <= sr; ++dx) {
                sdx.push_back(dx);
                sdy.push_back(dy);
                sdz.push_back(dz);
            }
    const int stencil_size = static_cast<int>(sdx.size());

    if (build_tile_graph_) {
        std::vector<int> tile_count_of_cell(static_cast<size_t>(total_cells));
        std::vector<int> tile_offset_of_cell(static_cast<size_t>(total_cells) + 1, 0);
        for (int c = 0; c < total_cells; ++c) {
            tile_count_of_cell[static_cast<size_t>(c)] =
                (cell_count[static_cast<size_t>(c)] + kCpuTileSize - 1) / kCpuTileSize;
        }
        for (int c = 0; c < total_cells; ++c) {
            tile_offset_of_cell[static_cast<size_t>(c) + 1] =
                tile_offset_of_cell[static_cast<size_t>(c)] + tile_count_of_cell[static_cast<size_t>(c)];
        }
        num_tiles_ = tile_offset_of_cell[static_cast<size_t>(total_cells)];

        const Real* px_tile = particles.x.data();
        const Real* py_tile = particles.y.data();
        const Real* pz_tile = particles.z.data();

        tile_atoms_.assign(static_cast<size_t>(num_tiles_) * kCpuTileSize, -1);
        #pragma omp parallel for schedule(static)
        for (int c = 0; c < total_cells; ++c) {
            const int base_tile = tile_offset_of_cell[static_cast<size_t>(c)];
            const int start = cell_start_[static_cast<size_t>(c)];
            const int cnt = cell_count[static_cast<size_t>(c)];
            if (cnt == 0) continue;

            thread_local std::vector<int> scratch_tl;
            scratch_tl.resize(static_cast<size_t>(cnt));
            for (int k = 0; k < cnt; ++k) {
                scratch_tl[static_cast<size_t>(k)] =
                    cell_atoms[static_cast<size_t>(start + k)];
            }

            kd_split_atoms(scratch_tl.data(), 0, cnt,
                           px_tile, py_tile, pz_tile);

            for (int k = 0; k < cnt; ++k) {
                const int atom = scratch_tl[static_cast<size_t>(k)];
                int tile_idx = base_tile + k / kCpuTileSize;
                int slot = k % kCpuTileSize;
                tile_atoms_[static_cast<size_t>(tile_idx) * kCpuTileSize + static_cast<size_t>(slot)] = atom;
            }
        }

        std::vector<uint8_t> tile_has_local(static_cast<size_t>(num_tiles_), 0);
        std::vector<Real> tile_min_x(static_cast<size_t>(num_tiles_), std::numeric_limits<Real>::max());
        std::vector<Real> tile_min_y(static_cast<size_t>(num_tiles_), std::numeric_limits<Real>::max());
        std::vector<Real> tile_min_z(static_cast<size_t>(num_tiles_), std::numeric_limits<Real>::max());
        std::vector<Real> tile_max_x(static_cast<size_t>(num_tiles_), -std::numeric_limits<Real>::max());
        std::vector<Real> tile_max_y(static_cast<size_t>(num_tiles_), -std::numeric_limits<Real>::max());
        std::vector<Real> tile_max_z(static_cast<size_t>(num_tiles_), -std::numeric_limits<Real>::max());

        #pragma omp parallel for schedule(static)
        for (int t = 0; t < num_tiles_; ++t) {
            bool has_any = false;
            for (int s = 0; s < kCpuTileSize; ++s) {
                const int atom = tile_atoms_[static_cast<size_t>(t) * kCpuTileSize + static_cast<size_t>(s)];
                if (atom < 0) continue;
                has_any = true;
                if (static_cast<size_t>(atom) < local_N) {
                    tile_has_local[static_cast<size_t>(t)] = 1;
                }
                const Real x = px_tile[atom];
                const Real y = py_tile[atom];
                const Real z = pz_tile[atom];
                tile_min_x[static_cast<size_t>(t)] = std::min(tile_min_x[static_cast<size_t>(t)], x);
                tile_min_y[static_cast<size_t>(t)] = std::min(tile_min_y[static_cast<size_t>(t)], y);
                tile_min_z[static_cast<size_t>(t)] = std::min(tile_min_z[static_cast<size_t>(t)], z);
                tile_max_x[static_cast<size_t>(t)] = std::max(tile_max_x[static_cast<size_t>(t)], x);
                tile_max_y[static_cast<size_t>(t)] = std::max(tile_max_y[static_cast<size_t>(t)], y);
                tile_max_z[static_cast<size_t>(t)] = std::max(tile_max_z[static_cast<size_t>(t)], z);
            }
            if (!has_any) {
                tile_min_x[static_cast<size_t>(t)] = tile_min_y[static_cast<size_t>(t)] = tile_min_z[static_cast<size_t>(t)] = 0.0;
                tile_max_x[static_cast<size_t>(t)] = tile_max_y[static_cast<size_t>(t)] = tile_max_z[static_cast<size_t>(t)] = 0.0;
            }
        }
        tile_has_local_ = tile_has_local;

        int max_threads = 1;
#ifdef _OPENMP
        max_threads = omp_get_max_threads();
#endif
        static std::vector<std::vector<TilePairRec>> thread_pairs;
        if (thread_pairs.size() != static_cast<size_t>(max_threads)) {
            thread_pairs.resize(static_cast<size_t>(max_threads));
        }
        for (auto& tp : thread_pairs) tp.clear();

        #pragma omp parallel
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            std::vector<TilePairRec>& local_pairs = thread_pairs[static_cast<size_t>(tid)];

            thread_local std::vector<int> nb_visit_token_tl;
            thread_local int current_token_tl = 0;
            if (nb_visit_token_tl.size() != static_cast<size_t>(total_cells)) {
                nb_visit_token_tl.assign(static_cast<size_t>(total_cells), 0);
                current_token_tl = 0;
            }
            std::vector<int> unique_nb_cells;
            std::vector<Real> unique_shift_x;
            std::vector<Real> unique_shift_y;
            std::vector<Real> unique_shift_z;
            unique_nb_cells.reserve(static_cast<size_t>(stencil_size));
            unique_shift_x.reserve(static_cast<size_t>(stencil_size));
            unique_shift_y.reserve(static_cast<size_t>(stencil_size));
            unique_shift_z.reserve(static_cast<size_t>(stencil_size));

            #pragma omp for schedule(dynamic, 32)
            for (int c = 0; c < total_cells; ++c) {
                const int n_home = tile_count_of_cell[static_cast<size_t>(c)];
                if (n_home == 0) continue;
                const int cz = c / (nx * ny);
                const int rem = c % (nx * ny);
                const int cy = rem / nx;
                const int cx = rem % nx;

                ++current_token_tl;
                if (current_token_tl == 0) {
                    std::fill(nb_visit_token_tl.begin(), nb_visit_token_tl.end(), 0);
                    current_token_tl = 1;
                }
                unique_nb_cells.clear();
                unique_shift_x.clear();
                unique_shift_y.clear();
                unique_shift_z.clear();
                for (int s = 0; s < stencil_size; ++s) {
                    int xc = cx + sdx[static_cast<size_t>(s)];
                    int yc = cy + sdy[static_cast<size_t>(s)];
                    int zc = cz + sdz[static_cast<size_t>(s)];
                    Real shift_x = 0.0, shift_y = 0.0, shift_z = 0.0;
                    // Wrap (and shift) an axis locally only when THIS axis is
                    // not MPI-split: an unsplit axis has no ghost atoms from
                    // dsDecompX (see topology.cpp's touches_unsplit_axis skip),
                    // so its periodicity must be resolved right here instead,
                    // exactly like the fully-serial (!is_mpi_subdomain) path.
                    // A single combined is_mpi_subdomain flag previously
                    // disabled wrapping on ALL axes whenever ANY axis was
                    // split (e.g. dims={2,1,1} or {2,2,1}), silently dropping
                    // periodic-boundary pairs on the unsplit axis/axes and
                    // causing energy blow-up at process counts that don't
                    // factor into a fully 3D grid (e.g. np=2, 4, or 7 PP
                    // ranks), while np=1 and perfectly-cubic counts (e.g. 8)
                    // never hit it.
                    if (split_x) {
                        if (xc < 0 || xc >= nx) continue;
                    } else if (box.periodic_x) {
                        if (xc < 0) { xc += nx; shift_x = -box.lx; }
                        else if (xc >= nx) { xc -= nx; shift_x = box.lx; }
                    } else if (xc < 0 || xc >= nx) continue;

                    if (split_y) {
                        if (yc < 0 || yc >= ny) continue;
                    } else if (box.periodic_y) {
                        if (yc < 0) { yc += ny; shift_y = -box.ly; }
                        else if (yc >= ny) { yc -= ny; shift_y = box.ly; }
                    } else if (yc < 0 || yc >= ny) continue;

                    if (split_z) {
                        if (zc < 0 || zc >= nz) continue;
                    } else if (box.periodic_z) {
                        if (zc < 0) { zc += nz; shift_z = -box.lz; }
                        else if (zc >= nz) { zc -= nz; shift_z = box.lz; }
                    } else if (zc < 0 || zc >= nz) continue;

                    int nb_cell = zc * nx * ny + yc * nx + xc;
                    if (nb_cell < 0 || nb_cell >= total_cells) continue;
                    if (nb_cell < c) continue;
                    if (tile_count_of_cell[static_cast<size_t>(nb_cell)] == 0) continue;
                    if (nb_visit_token_tl[static_cast<size_t>(nb_cell)] == current_token_tl) continue;
                    nb_visit_token_tl[static_cast<size_t>(nb_cell)] = current_token_tl;
                    unique_nb_cells.push_back(nb_cell);
                    unique_shift_x.push_back(shift_x);
                    unique_shift_y.push_back(shift_y);
                    unique_shift_z.push_back(shift_z);
                }

                const int home_base = tile_offset_of_cell[static_cast<size_t>(c)];
                for (int ti = 0; ti < n_home; ++ti) {
                    const int home_tile = home_base + ti;
                    for (size_t nb_idx = 0; nb_idx < unique_nb_cells.size(); ++nb_idx) {
                        const int nb_cell = unique_nb_cells[nb_idx];
                        // unique_shift_{x,y,z} are already 0 on any MPI-split
                        // axis (see the split_x/split_y/split_z stencil check
                        // above) and only nonzero on an unsplit periodic axis
                        // where this rank must apply the wrap itself.
                        const Real shift_x = unique_shift_x[nb_idx];
                        const Real shift_y = unique_shift_y[nb_idx];
                        const Real shift_z = unique_shift_z[nb_idx];
                        const int n_nb = tile_count_of_cell[static_cast<size_t>(nb_cell)];
                        const int nb_base = tile_offset_of_cell[static_cast<size_t>(nb_cell)];

                        auto maybe_emit_pair = [&](int nb_tile) {
                            if (!tile_has_local[static_cast<size_t>(home_tile)] &&
                                !tile_has_local[static_cast<size_t>(nb_tile)]) {
                                return;
                            }

                            const Real min_dist_sq = aabb_min_distance_sq(
                                tile_min_x[static_cast<size_t>(home_tile)],
                                tile_min_y[static_cast<size_t>(home_tile)],
                                tile_min_z[static_cast<size_t>(home_tile)],
                                tile_max_x[static_cast<size_t>(home_tile)],
                                tile_max_y[static_cast<size_t>(home_tile)],
                                tile_max_z[static_cast<size_t>(home_tile)],
                                tile_min_x[static_cast<size_t>(nb_tile)],
                                tile_min_y[static_cast<size_t>(nb_tile)],
                                tile_min_z[static_cast<size_t>(nb_tile)],
                                tile_max_x[static_cast<size_t>(nb_tile)],
                                tile_max_y[static_cast<size_t>(nb_tile)],
                                tile_max_z[static_cast<size_t>(nb_tile)],
                                shift_x, shift_y, shift_z);
                            if (min_dist_sq >= rcut2) {
                                return;
                            }

                            int emit_home = home_tile;
                            int emit_nb = nb_tile;
                            Real emit_sx = shift_x, emit_sy = shift_y, emit_sz = shift_z;
                            if (!tile_has_local[static_cast<size_t>(emit_home)] &&
                                tile_has_local[static_cast<size_t>(emit_nb)]) {
                                std::swap(emit_home, emit_nb);
                                emit_sx = -emit_sx; emit_sy = -emit_sy; emit_sz = -emit_sz;
                            }
                            local_pairs.push_back({emit_home, emit_nb, emit_sx, emit_sy, emit_sz});
                        };

                        if (nb_cell == c) {
                            for (int tj = 0; tj < n_nb; ++tj) {
                                const int nb_tile = nb_base + tj;
                                if (nb_tile < home_tile) continue;
                                maybe_emit_pair(nb_tile);
                            }
                        } else {
                            for (int tj = 0; tj < n_nb; ++tj) {
                                maybe_emit_pair(nb_base + tj);
                            }
                        }
                    }
                }
            }
        }  // end omp parallel

        tile_pair_offsets_.assign(static_cast<size_t>(num_tiles_) + 1, 0);
        size_t total_tile_pairs = 0;
        for (const auto& tp : thread_pairs) total_tile_pairs += tp.size();

        static std::vector<std::vector<int>> thread_hist;
        static std::vector<std::vector<int>> thread_cursor_base;
        if (thread_hist.size() != static_cast<size_t>(max_threads)) {
            thread_hist.resize(static_cast<size_t>(max_threads));
            thread_cursor_base.resize(static_cast<size_t>(max_threads));
        }
        for (auto& h : thread_hist) h.assign(static_cast<size_t>(num_tiles_), 0);
        for (auto& b : thread_cursor_base) b.assign(static_cast<size_t>(num_tiles_), 0);

        #pragma omp parallel for schedule(static)
        for (int t = 0; t < max_threads; ++t) {
            std::vector<int>& hist = thread_hist[static_cast<size_t>(t)];
            for (const auto& pr : thread_pairs[static_cast<size_t>(t)]) {
                ++hist[static_cast<size_t>(pr.home)];
            }
        }

        #pragma omp parallel for schedule(static)
        for (int tile = 0; tile < num_tiles_; ++tile) {
            int running = 0;
            for (int t = 0; t < max_threads; ++t) {
                thread_cursor_base[static_cast<size_t>(t)][static_cast<size_t>(tile)] = running;
                running += thread_hist[static_cast<size_t>(t)][static_cast<size_t>(tile)];
            }
            tile_pair_offsets_[static_cast<size_t>(tile) + 1] = running;
        }
        for (int t = 0; t < num_tiles_; ++t) {
            tile_pair_offsets_[static_cast<size_t>(t) + 1] += tile_pair_offsets_[static_cast<size_t>(t)];
        }

        tile_pair_other_.resize(total_tile_pairs);
        tile_pair_shift_x_.resize(total_tile_pairs);
        tile_pair_shift_y_.resize(total_tile_pairs);
        tile_pair_shift_z_.resize(total_tile_pairs);

        #pragma omp parallel for schedule(static)
        for (int t = 0; t < max_threads; ++t) {
            const std::vector<int>& base = thread_cursor_base[static_cast<size_t>(t)];
            std::vector<int> cursor(static_cast<size_t>(num_tiles_));
            for (int tile = 0; tile < num_tiles_; ++tile) {
                cursor[static_cast<size_t>(tile)] =
                    tile_pair_offsets_[static_cast<size_t>(tile)] + base[static_cast<size_t>(tile)];
            }
            for (const auto& pr : thread_pairs[static_cast<size_t>(t)]) {
                const size_t w = static_cast<size_t>(cursor[static_cast<size_t>(pr.home)]++);
                tile_pair_other_[w] = pr.nb;
                tile_pair_shift_x_[w] = pr.sx;
                tile_pair_shift_y_[w] = pr.sy;
                tile_pair_shift_z_[w] = pr.sz;
            }
        }

        tile_pair_verlet_mask_.assign(total_tile_pairs, 0);
        unsigned long long inside_outer = 0;
        unsigned long long inside_rc = 0;
        const Real force_rc2 = cutoff_ * cutoff_;
        #pragma omp parallel for schedule(static) reduction(+:inside_outer,inside_rc)
        for (int T = 0; T < num_tiles_; ++T) {
            const size_t tbase = static_cast<size_t>(T) * kCpuTileSize;
            for (int e = tile_pair_offsets_[static_cast<size_t>(T)];
                 e < tile_pair_offsets_[static_cast<size_t>(T) + 1]; ++e) {
                const int O = tile_pair_other_[static_cast<size_t>(e)];
                const bool same_tile = (O == T);
                const size_t obase = static_cast<size_t>(O) * kCpuTileSize;
                const Real shift_x = tile_pair_shift_x_[static_cast<size_t>(e)];
                const Real shift_y = tile_pair_shift_y_[static_cast<size_t>(e)];
                const Real shift_z = tile_pair_shift_z_[static_cast<size_t>(e)];
                std::uint64_t mask = 0;
                for (int slot = 0; slot < kCpuTileSize; ++slot) {
                    const int j = tile_atoms_[obase + static_cast<size_t>(slot)];
                    if (j < 0) continue;
                    const bool j_local = static_cast<size_t>(j) < local_N;
                    const Real jx = particles.x[static_cast<size_t>(j)] + shift_x;
                    const Real jy = particles.y[static_cast<size_t>(j)] + shift_y;
                    const Real jz = particles.z[static_cast<size_t>(j)] + shift_z;
                    for (int lane = 0; lane < kCpuTileSize; ++lane) {
                        const int i = tile_atoms_[tbase + static_cast<size_t>(lane)];
                        if (i < 0 || static_cast<size_t>(i) >= local_N || i == j) continue;
                        if (same_tile && j_local && !(j > i)) continue;
                        const Real dx = particles.x[static_cast<size_t>(i)] - jx;
                        const Real dy = particles.y[static_cast<size_t>(i)] - jy;
                        const Real dz = particles.z[static_cast<size_t>(i)] - jz;
                        const Real r2 = dx * dx + dy * dy + dz * dz;
                        if (r2 < rcut2) {
                            mask |= std::uint64_t{1} << (slot * kCpuTileSize + lane);
                            ++inside_outer;
                            if (r2 < force_rc2) ++inside_rc;
                        }
                    }
                }
                tile_pair_verlet_mask_[static_cast<size_t>(e)] = mask;
            }
        }
        pairs_inside_rc_ = static_cast<std::uint64_t>(inside_rc);
        pairs_inside_rc_plus_skin_ = static_cast<std::uint64_t>(inside_outer);
    } else {
        tile_atoms_.clear();
        tile_pair_offsets_.clear();
        tile_pair_other_.clear();
        tile_pair_verlet_mask_.clear();
        tile_pair_shift_x_.clear();
        tile_pair_shift_y_.clear();
        tile_pair_shift_z_.clear();
        tile_has_local_.clear();
        num_tiles_ = 0;
        pairs_inside_rc_ = 0;
        pairs_inside_rc_plus_skin_ = 0;
    }

    const bool half_shell = !has_ghosts;
    const bool pbc_x = box.periodic_x, pbc_y = box.periodic_y, pbc_z = box.periodic_z;
    const bool can_apply_minimum_image = (pbc_x || pbc_y || pbc_z);

    const Real* px = particles.x.data();
    const Real* py = particles.y.data();
    const Real* pz = particles.z.data();
    const int local_n_int = static_cast<int>(local_N);
    const int* cell_start_ptr = cell_start_.data();
    const int* cell_atoms_ptr = cell_atoms.data();
    const int* sdx_ptr = sdx.data();
    const int* sdy_ptr = sdy.data();
    const int* sdz_ptr = sdz.data();

    auto for_each_candidate = [&](size_t i, auto&& fn) {
        const int ix = cix[i], iy = ciy[i], iz = ciz[i];
        const Real xi = px[i], yi = py[i], zi = pz[i];
        const int i_int = static_cast<int>(i);
        const int home_cell_id = cell_of_[i];

        thread_local std::vector<unsigned int> visited;
        thread_local unsigned int token = 0;
        if (visited.size() < static_cast<size_t>(total_cells)) {
            visited.assign(static_cast<size_t>(total_cells), 0u);
            token = 0;
        }
        ++token;
        if (token == 0u) {
            std::fill(visited.begin(), visited.end(), 0u);
            token = 1u;
        }

        for (int s = 0; s < stencil_size; ++s) {
            if (half_shell) {
                if (sdz_ptr[s] < 0) continue;
                if (sdz_ptr[s] == 0 && sdy_ptr[s] < 0) continue;
                if (sdz_ptr[s] == 0 && sdy_ptr[s] == 0 && sdx_ptr[s] < 0) continue;
            }
            int xc = ix + sdx_ptr[s], yc = iy + sdy_ptr[s], zc = iz + sdz_ptr[s];
            // Same fix as the tile-based stencil above: wrap an axis's cell
            // index locally only when that axis is NOT MPI-split -- an
            // unsplit axis gets no ghost atoms from dsDecompX, so this rank
            // must still find its own periodic-image neighbors on that axis
            // itself (apply_minimum_image below then fixes up the distance).
            if (split_x) {
                if (xc < 0 || xc >= nx) continue;
            } else if (pbc_x) {
                if (xc < 0) xc += nx; else if (xc >= nx) xc -= nx;
            } else if (xc < 0 || xc >= nx) continue;

            if (split_y) {
                if (yc < 0 || yc >= ny) continue;
            } else if (pbc_y) {
                if (yc < 0) yc += ny; else if (yc >= ny) yc -= ny;
            } else if (yc < 0 || yc >= ny) continue;

            if (split_z) {
                if (zc < 0 || zc >= nz) continue;
            } else if (pbc_z) {
                if (zc < 0) zc += nz; else if (zc >= nz) zc -= nz;
            } else if (zc < 0 || zc >= nz) continue;

            int cell_id = zc * nx * ny + yc * nx + xc;
            if (cell_id < 0 || cell_id >= total_cells) continue;
            if (visited[static_cast<size_t>(cell_id)] == token) continue;
            visited[static_cast<size_t>(cell_id)] = token;

            const bool is_home_cell = (cell_id == home_cell_id);
            const bool apply_index_tiebreak = is_home_cell || !half_shell;
            for (int k = cell_start_ptr[cell_id]; k < cell_start_ptr[cell_id + 1]; ++k) {
                int j = cell_atoms_ptr[k];
                if (j == i_int) continue;
                if (apply_index_tiebreak && j < local_n_int && j < i_int) continue;
                Real ddx = xi - px[j], ddy = yi - py[j], ddz = zi - pz[j];
                const bool j_is_local = (j < local_n_int);
                if (can_apply_minimum_image && (!is_mpi_subdomain || j_is_local)) {
                    box.apply_minimum_image(ddx, ddy, ddz);
                }
                if (ddx * ddx + ddy * ddy + ddz * ddz < rcut2) {
                    fn(j);
                }
            }
        }
    };

    const bool need_atom_csr = build_active_graph_;
    if (need_atom_csr) {
    if (neighbor_capacity_per_atom_ <= 0) {
        const Real volume = Lx * Ly * Lz;
        const Real density = volume > 0.0 ? static_cast<Real>(total_N) / volume : 0.0;
        const Real sphere = (4.0 / 3.0) * M_PI * extended_cutoff * extended_cutoff * extended_cutoff;
        neighbor_capacity_per_atom_ = std::max(64, static_cast<int>(std::ceil(1.15 * density * sphere)));
    }

    per_atom_count_.resize(local_N);
    for (;;) {
        const size_t capacity = static_cast<size_t>(neighbor_capacity_per_atom_);
        const size_t padded_size = local_N * capacity;
        if (padded_neighbors_.size() != padded_size) {
            padded_neighbors_.resize(padded_size);
        }

        std::atomic<int> overflow_max{0};
        #pragma omp parallel for schedule(dynamic, 64)
        for (long ii = 0; ii < static_cast<long>(local_N); ++ii) {
            size_t i = static_cast<size_t>(ii);
            int* row = padded_neighbors_.data() + i * capacity;
            int k = 0;
            int true_count = 0;
            for_each_candidate(i, [&](int j) {
                if (static_cast<size_t>(k) < capacity) row[k++] = j;
                ++true_count;
            });
            per_atom_count_[i] = true_count;
            if (true_count > neighbor_capacity_per_atom_) {
                int prev = overflow_max.load(std::memory_order_relaxed);
                while (true_count > prev &&
                       !overflow_max.compare_exchange_weak(prev, true_count, std::memory_order_relaxed)) {
                }
            }
        }

        const int observed_overflow = overflow_max.load();
        if (observed_overflow <= neighbor_capacity_per_atom_) break;  // no row was truncated

        neighbor_capacity_per_atom_ = observed_overflow + observed_overflow / 4 + 8;
    }

    offsets_.resize(local_N + 1);
    for (size_t i = 0; i < local_N; ++i) {
        offsets_[i + 1] = offsets_[i] + per_atom_count_[i];
    }
    neighbors_.resize(static_cast<size_t>(offsets_[local_N]));

    {
        const size_t capacity = static_cast<size_t>(neighbor_capacity_per_atom_);
        #pragma omp parallel for schedule(dynamic, 64)
        for (long ii = 0; ii < static_cast<long>(local_N); ++ii) {
            size_t i = static_cast<size_t>(ii);
            const int* row = padded_neighbors_.data() + i * capacity;
            int* out = neighbors_.data() + offsets_[i];
            std::copy(row, row + per_atom_count_[i], out);
        }
    }
    } else {
        offsets_.assign(local_N + 1, 0);
        neighbors_.clear();
        per_atom_count_.clear();
        padded_neighbors_.clear();
        padded_neighbors_.shrink_to_fit();
    }

    if (build_active_graph_) {
        prune_active(particles, box, local_N);
    } else {
        active_neighbors_.clear();
        active_offsets_.clear();
        active_degree_.clear();
    }

    last_build_x_.assign(particles.x.begin(), particles.x.begin() + static_cast<std::ptrdiff_t>(local_N));
    last_build_y_.assign(particles.y.begin(), particles.y.begin() + static_cast<std::ptrdiff_t>(local_N));
    last_build_z_.assign(particles.z.begin(), particles.z.begin() + static_cast<std::ptrdiff_t>(local_N));
}
