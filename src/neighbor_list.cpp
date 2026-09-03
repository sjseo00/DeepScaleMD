#include "neighbor_list.h"
#include "md_types.h"
#include "mpi_utils.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <stdexcept>
#include <mpi.h>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>

extern "C" void launch_cuda_neighbor_list_build(
    const Real* d_x, const Real* d_y, const Real* d_z,
    int* d_neighbor_list,
    int* d_num_neighbors,
    int* d_neighbor_offset,
    int n_local, int n_total,
    Real cutoff, Real verlet_skin,
    Real lx, Real ly, Real lz,
    bool periodic_x, bool periodic_y, bool periodic_z,
    int max_neighbors_per_particle,
    cudaStream_t stream);

// Morton-optimized GPU kernel: GPU-side Morton sort + cell-range + neighbor search
extern "C" void launch_cuda_morton_neighbor_list_build(
    const Real* d_x, const Real* d_y, const Real* d_z,
    int* d_neighbor_list,
    int* d_num_neighbors,
    int* d_neighbor_offset,
    int n_local, int n_total,
    Real cutoff, Real verlet_skin,
    Real lx, Real ly, Real lz,
    bool periodic_x, bool periodic_y, bool periodic_z,
    int max_neighbors_per_particle,
    cudaStream_t stream,
    Real* d_sorted_x, Real* d_sorted_y, Real* d_sorted_z,
    int* d_sorted_indices,
    unsigned long long* d_morton_codes,
    int* d_cell_start, int* d_cell_end,
    int* d_compact_neighbors,
    int* d_total_neighbors_out);
#endif

#ifdef USE_SYCL
#include <sycl/sycl.hpp>
#endif

// Enhanced logging with rank information
// PERF FIX: cache MPI rank at first call to avoid repeated MPI_Initialized +
// MPI_Comm_rank overhead on every log call (hot path during neighbor build).
static int s_log_mpi_rank = -1;
static inline int get_log_rank() {
    if (s_log_mpi_rank < 0) {
        int init = 0;
        MPI_Initialized(&init);
        s_log_mpi_rank = 0;
        if (init) MPI_Comm_rank(MPI_COMM_WORLD, &s_log_mpi_rank);
    }
    return s_log_mpi_rank;
}

inline void log_message(int level, const std::string& message) {
    if (get_log_rank() == 0) {
        std::cout << "[NeighborList] " << message << std::endl;
    }
}

inline void log_warning(const std::string& message) {
    if (get_log_rank() == 0) {
        std::cerr << "[NeighborList WARNING] " << message << std::endl;
    }
}

inline void log_error(const std::string& message) {
    if (get_log_rank() == 0) {
        std::cerr << "[NeighborList ERROR] " << message << std::endl;
    }
}



// VerletCellListNeighborBuilder implementation
VerletCellListNeighborBuilder::VerletCellListNeighborBuilder(Real cutoff_val, Real skin, bool half_list) 
        : cutoff(cutoff_val), verlet_skin(skin), update_frequency(1), delay_steps(0), info_printed(false),
            stencil_size(0), use_half_list(half_list), check_enabled(true) {
    cell_size = cutoff + verlet_skin;  // Larger cell size for Verlet skin
    // Threshold: rebuild when any particle moves more than skin/2
    max_displacement_threshold = (verlet_skin * 0.5) * (verlet_skin * 0.5);
}

void VerletCellListNeighborBuilder::initialize() {
    // Initialization handled in build method
}

void VerletCellListNeighborBuilder::finalize() {
    cells.clear();
    last_build_x.clear();
    last_build_y.clear();
    last_build_z.clear();
}

bool VerletCellListNeighborBuilder::needs_rebuild(const ParticleData& particles, const SimulationBox& box, size_t num_local) {
    // If check is disabled, always return false (caller controls rebuild)
    if (!check_enabled) return false;
    
    // First build always needed
    if (last_build_x.empty() || last_build_y.empty() || last_build_z.empty()) {
        return true;
    }
    
    // Check if size changed.
    // PERFORMANCE NOTE: In MPI mode migration temporarily removes ghosts.
    // Only treat a size change as needing rebuild when it is a REAL change
    // (i.e. actual migration happened, detected by the caller).
    // Here we simply cap displacement comparison to min(num_local, cached) and
    // return true only when the discrepancy is more than ghost-strip noise
    // (>5% change) so that transient ghost removal does not trigger a rebuild.
    size_t cached = last_build_x.size();
    if (num_local != cached) {
        // Tolerate small fluctuations (ghost removal); force rebuild on real changes
        Real frac = (num_local > cached)
                    ? static_cast<Real>(num_local - cached) / cached
                    : static_cast<Real>(cached - num_local) / cached;
        if (frac > 0.02) return true;  // > 2% change → real migration
        // Otherwise use the smaller count for displacement check
        num_local = std::min(num_local, cached);
    }
    
    // Find maximum displacement since last build (local particles only)
    // CRITICAL: Apply minimum image convention to handle periodic boundaries correctly
    Real local_max_disp_sq = 0.0;
    for (size_t i = 0; i < num_local; ++i) {
        Real dx = particles.x[i] - last_build_x[i];
        Real dy = particles.y[i] - last_build_y[i];
        Real dz = particles.z[i] - last_build_z[i];
        
        // Apply minimum image convention for periodic boundaries
        box.apply_minimum_image(dx, dy, dz);
        
        Real disp_sq = dx*dx + dy*dy + dz*dz;
        local_max_disp_sq = std::max(local_max_disp_sq, disp_sq);
    }
    
    // Rebuild if any local particle moved more than threshold (skin/2).
    // PERFORMANCE FIX: Remove per-step MPI_Allreduce from needs_rebuild().
    // The caller (md_simulation.cpp) now calls needs_rebuild() only every
    // neighbor_list_update_freq steps, so one MPI_Allreduce every N steps
    // instead of every step.  All ranks use the same throttled schedule so
    // they will always agree on rebuild decisions.
    int local_rebuild = (local_max_disp_sq > max_displacement_threshold) ? 1 : 0;
    int global_rebuild = 0;
    
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized) {
        MPI_Allreduce(&local_rebuild, &global_rebuild, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    } else {
        global_rebuild = local_rebuild;
    }
    
    return global_rebuild > 0;
}

void VerletCellListNeighborBuilder::build(ParticleData& particles, SimulationBox& box, NeighborList& nlist, size_t num_local) {
    // LAMMPS-style: num_local = owned particles, num_particles = owned + ghost
    size_t local_N = (num_local > 0) ? num_local : particles.num_particles;
    size_t total_N = particles.num_particles;  // includes ghost particles

    // Extended cutoff for neighbor search (Verlet skin included)
    const Real extended_cutoff = cutoff + verlet_skin;
    const Real rcut2 = extended_cutoff * extended_cutoff;
    nlist.cutoff = extended_cutoff; // Store search cutoff; physical cutoff remains 'cutoff'
    nlist.list_type = "verlet_cell";

    // Determine if we are in MPI mode (subdomain) or serial mode (full box)
    bool is_mpi_subdomain = (box.xhi > box.xlo) && (box.xhi < box.lx || box.xlo > 0.0);
    // Fallback: always treat as subdomain if xhi != lx or xlo != 0
    // Actually simpler: check if xhi-xlo < lx
    is_mpi_subdomain = (box.xhi > box.xlo) && ((box.xhi - box.xlo) < box.lx - 1e-6);

    // Subdomain origin (local particle coordinates)
    Real origin_x = (box.xhi > box.xlo) ? box.xlo : 0.0;
    Real origin_y = (box.yhi > box.ylo) ? box.ylo : 0.0;
    Real origin_z = (box.zhi > box.zlo) ? box.zlo : 0.0;

    // For MPI mode: extend the cell grid to cover ghost particles.
    // Ghost particles can be located up to extended_cutoff outside the subdomain.
    // We add one extra cell-layer on each side to properly bin ghost particles.
    // This prevents the "clamp" issue where all ghosts pile into boundary cells.
    Real ghost_margin = is_mpi_subdomain ? extended_cutoff : 0.0;

    // Total domain extent for cell grid (subdomain + ghost region on each side)
    Real Lx_sub = (box.xhi > box.xlo) ? (box.xhi - box.xlo) : box.lx;
    Real Ly_sub = (box.yhi > box.ylo) ? (box.yhi - box.ylo) : box.ly;
    Real Lz_sub = (box.zhi > box.zlo) ? (box.zhi - box.zlo) : box.lz;

    // Grid covers [origin - ghost_margin, origin + L_sub + ghost_margin]
    Real grid_origin_x = origin_x - ghost_margin;
    Real grid_origin_y = origin_y - ghost_margin;
    Real grid_origin_z = origin_z - ghost_margin;
    Real Lx = Lx_sub + 2.0 * ghost_margin;
    Real Ly = Ly_sub + 2.0 * ghost_margin;
    Real Lz = Lz_sub + 2.0 * ghost_margin;

    // Compute cell grid dimensions: use extended_cutoff for optimal binning (LAMMPS style)
    // CRITICAL: Use floor (not round) so that actual_bin_size >= extended_cutoff,
    // which guarantees stencil_range=1 (27-cell search).
    // If we use round, bins can be over-subdivided → binsize < extended_cutoff → stencil=2
    // (125 cells) → ~4.6x more work per build.
    Real target_bin_size = extended_cutoff;
    
    // floor ensures binsize = Lx/nx >= target_bin_size (LAMMPS uses the same approach)
    nx = std::max(1, static_cast<int>(Lx / target_bin_size));
    ny = std::max(1, static_cast<int>(Ly / target_bin_size));
    nz = std::max(1, static_cast<int>(Lz / target_bin_size));

    // Calculate actual bin size and stencil range
    Real actual_bin_size_x = Lx / nx;
    Real actual_bin_size_y = Ly / ny;
    Real actual_bin_size_z = Lz / nz;
    Real max_bin_size = std::max({actual_bin_size_x, actual_bin_size_y, actual_bin_size_z});
    
    // Stencil range: how many bins we need to search in each direction
    // With floor-based binning above, stencil_range should always be 1 (27 cells).
    int stencil_range = static_cast<int>(std::ceil(extended_cutoff / max_bin_size));
    stencil_range = std::max(1, stencil_range); // At least 1

    const Real inv_lx_cell = nx / Lx;
    const Real inv_ly_cell = ny / Ly;
    const Real inv_lz_cell = nz / Lz;

    const int total_cells = nx * ny * nz;

    // ── LAMMPS-style flat (CSR) cell array ──────────────────────────────────
    // Replaces vector<vector<int>> cells which caused O(N) small heap
    // alloc/free per rebuild → poor cache performance.
    // We use two flat arrays:
    //   cell_head[c]  = first atom index in cell c (or -1 if empty)
    //   cell_next[i]  = next atom in the same cell as atom i (or -1)
    // This is the same linked-list bin approach used by LAMMPS.
    // Re-using persistent member vectors avoids reallocation between builds.
    // -----------------------------------------------------------------------

    // Resize persistent work arrays (member variables, avoid re-malloc every build)
    flat_cell_head.assign(total_cells, -1);  // -1 = empty cell
    flat_cell_next.resize(total_N, -1);

    // Precompute particle cell indices (reuse member arrays to avoid realloc)
    cx_arr.resize(total_N);
    cy_arr.resize(total_N);
    cz_arr.resize(total_N);
    
    // Precompute stencil (done once per build; grid dims rarely change)
    // Build 3D stencil for [-stencil_range..stencil_range]^3 offsets
    stencil_size = 0;
    int sr = stencil_range;
    int stencil_cap = (2*sr+1) * (2*sr+1) * (2*sr+1);
    stencil_dx.resize(stencil_cap);
    stencil_dy.resize(stencil_cap);
    stencil_dz.resize(stencil_cap);
    for (int dz_ = -sr; dz_ <= sr; ++dz_)
        for (int dy_ = -sr; dy_ <= sr; ++dy_)
            for (int dx_ = -sr; dx_ <= sr; ++dx_) {
                stencil_dx[stencil_size] = dx_;
                stencil_dy[stencil_size] = dy_;
                stencil_dz[stencil_size] = dz_;
                ++stencil_size;
            }
    
    // Fill linked-list cell structure for ALL particles (owned + ghost)
    for (size_t i = 0; i < total_N; ++i) {
        Real rx = particles.x[i] - grid_origin_x;
        Real ry = particles.y[i] - grid_origin_y;
        Real rz = particles.z[i] - grid_origin_z;
        
        int ix = static_cast<int>(rx * inv_lx_cell);
        int iy = static_cast<int>(ry * inv_ly_cell);
        int iz = static_cast<int>(rz * inv_lz_cell);

        ix = std::clamp(ix, 0, nx - 1);
        iy = std::clamp(iy, 0, ny - 1);
        iz = std::clamp(iz, 0, nz - 1);

        cx_arr[i] = ix; cy_arr[i] = iy; cz_arr[i] = iz;
        int cell_id = iz * nx * ny + iy * nx + ix;

        // Insert atom i at head of cell's linked list (reverse order, doesn't matter)
        if (cell_id >= 0 && cell_id < total_cells) {
            flat_cell_next[i] = flat_cell_head[cell_id];
            flat_cell_head[cell_id] = static_cast<int>(i);
        }
    }

    // Initialize neighbor list arrays (sized for LOCAL particles only)
    nlist.neighbors.clear();
    nlist.num_neighbors.assign(local_N, 0);
    nlist.neighbor_offset.assign(local_N, 0);

    // Reserve approximate neighbor capacity (based on subdomain density)
    size_t est_neighbors_per_particle = 0;
    if (local_N > 0) {
        Real subdomain_vol = Lx_sub * Ly_sub * Lz_sub;
        Real cell_volume = (subdomain_vol > 0.0) ? (subdomain_vol / static_cast<Real>(local_N)) : 1.0;
        Real sphere_volume = (4.0/3.0) * M_PI * extended_cutoff * extended_cutoff * extended_cutoff;
        // Half-list: ~half the neighbors for local-local (ghost neighbors still full)
        est_neighbors_per_particle = static_cast<size_t>(sphere_volume / cell_volume * 0.7);
        est_neighbors_per_particle = std::max<size_t>(8, std::min<size_t>(est_neighbors_per_particle, 512));
        nlist.neighbors.reserve(local_N * est_neighbors_per_particle);
    }

    // Neighbors (j) can be from owned OR ghost particles (0..total_N-1)
    const Real* px = particles.x.data();
    const Real* py = particles.y.data();
    const Real* pz = particles.z.data();

    // ── SINGLE-PASS serial neighbor list build ──────────────────────────────
    // LAMMPS-style: iterate local atoms, append neighbors directly.
    // Eliminates the two-pass (count + fill) overhead and OMP barrier cost.
    // For single-thread (OMP_NUM_THREADS=1) this is 2x faster:
    //   - Only one cell traversal instead of two
    //   - No OMP parallel region overhead (fork/join, barriers)
    //   - Sequential push_back → prefetcher-friendly
    // -----------------------------------------------------------------------

    const int* cell_head_ptr = flat_cell_head.data();
    const int* cell_next_ptr = flat_cell_next.data();

    auto append_particle_neighbors = [&](size_t i, std::vector<int>& output) -> int {
        int ix = cx_arr[i];
        int iy = cy_arr[i];
        int iz = cz_arr[i];
        Real xi = px[i];
        Real yi = py[i];
        Real zi = pz[i];
        int cnt = 0;

        // Use precomputed stencil to avoid triple-nested loop overhead
        const int* sdx = stencil_dx.data();
        const int* sdy = stencil_dy.data();
        const int* sdz = stencil_dz.data();
        for (int s = 0; s < stencil_size; ++s) {
            int xc = ix + sdx[s];
            int yc = iy + sdy[s];
            int zc = iz + sdz[s];
            if (is_mpi_subdomain) {
                if (xc < 0 || xc >= nx || yc < 0 || yc >= ny || zc < 0 || zc >= nz) continue;
            } else {
                if (box.periodic_x) { if (xc < 0) xc += nx; else if (xc >= nx) xc -= nx; }
                else if (xc < 0 || xc >= nx) continue;
                if (box.periodic_y) { if (yc < 0) yc += ny; else if (yc >= ny) yc -= ny; }
                else if (yc < 0 || yc >= ny) continue;
                if (box.periodic_z) { if (zc < 0) zc += nz; else if (zc >= nz) zc -= nz; }
                else if (zc < 0 || zc >= nz) continue;
            }
            int cell_id = zc * nx * ny + yc * nx + xc;
            if (cell_id < 0 || cell_id >= total_cells) continue;
            // Half-list (Newton ON, CPU only): for local-local, only j > i
            // Full-list (Newton OFF, GPU path): all pairs stored
            for (int j = cell_head_ptr[cell_id]; j >= 0; j = cell_next_ptr[j]) {
                if (j == static_cast<int>(i)) continue;
                if (use_half_list && j < static_cast<int>(local_N) && j < static_cast<int>(i)) continue;
                Real ddx = xi - px[j];
                Real ddy = yi - py[j];
                Real ddz = zi - pz[j];
                if (!is_mpi_subdomain) box.apply_minimum_image(ddx, ddy, ddz);
                if (ddx*ddx + ddy*ddy + ddz*ddz < rcut2) {
                    output.push_back(j);
                    ++cnt;
                }
            }
        }
        return cnt;
    };

#ifdef _OPENMP
    // A single shared push_back serializes the build.  Give each thread a
    // contiguous atom range and a private flat buffer, then concatenate once.
    // Keep small systems serial because OpenMP startup and merging dominate.
    const int requested_threads = omp_get_max_threads();
    const bool use_parallel_build = requested_threads > 1 && local_N >= 2048;
    if (use_parallel_build) {
        std::vector<std::vector<int>> thread_neighbors;
        std::vector<size_t> thread_begin;
        std::vector<size_t> thread_end;

        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            const int thread_count = omp_get_num_threads();

            #pragma omp single
            {
                thread_neighbors.resize(thread_count);
                thread_begin.resize(thread_count, 0);
                thread_end.resize(thread_count, 0);
            }

            const size_t begin = local_N * static_cast<size_t>(tid) / static_cast<size_t>(thread_count);
            const size_t end = local_N * static_cast<size_t>(tid + 1) / static_cast<size_t>(thread_count);
            thread_begin[tid] = begin;
            thread_end[tid] = end;

            std::vector<int>& local_neighbors = thread_neighbors[tid];
            local_neighbors.reserve((end - begin) * est_neighbors_per_particle);

            for (size_t i = begin; i < end; ++i) {
                nlist.neighbor_offset[i] = static_cast<int>(local_neighbors.size());
                nlist.num_neighbors[i] = append_particle_neighbors(i, local_neighbors);
            }
        }

        const int thread_count = static_cast<int>(thread_neighbors.size());
        std::vector<size_t> thread_base(thread_count, 0);
        size_t total_neighbors = 0;
        for (int tid = 0; tid < thread_count; ++tid) {
            thread_base[tid] = total_neighbors;
            total_neighbors += thread_neighbors[tid].size();
        }
        nlist.neighbors.resize(total_neighbors);

        #pragma omp parallel for schedule(static)
        for (int tid = 0; tid < thread_count; ++tid) {
            const size_t base = thread_base[tid];
            std::copy(thread_neighbors[tid].begin(), thread_neighbors[tid].end(),
                      nlist.neighbors.begin() + static_cast<std::ptrdiff_t>(base));
            const size_t begin = thread_begin[tid];
            const size_t end = thread_end[tid];
            for (size_t i = begin; i < end; ++i) {
                nlist.neighbor_offset[i] += static_cast<int>(base);
            }
        }
    } else
#endif
    {
        for (size_t i = 0; i < local_N; ++i) {
            nlist.neighbor_offset[i] = static_cast<int>(nlist.neighbors.size());
            nlist.num_neighbors[i] = append_particle_neighbors(i, nlist.neighbors);
        }
    }
    // (flat_cell_head/next, cx_arr/cy_arr/cz_arr are member vars, reused next build)

    
    // Print neighbor list information (LAMMPS-style logging) - only once per rank
    // Note: Each rank will log its own statistics
    if (!info_printed) {
        // Calculate max neighbors per atom and average for LOCAL particles
        int max_neighbors = 0;
        size_t total_neighbors = 0;
        if (!nlist.num_neighbors.empty()) {
            max_neighbors = *std::max_element(nlist.num_neighbors.begin(), nlist.num_neighbors.end());
            for (size_t i = 0; i < nlist.num_neighbors.size(); ++i) {
                total_neighbors += nlist.num_neighbors[i];
            }
        }
        Real avg_neighbors = (local_N > 0) ? static_cast<Real>(total_neighbors) / static_cast<Real>(local_N) : 0.0;
        
        // Calculate actual bin size (should be approximately extended_cutoff/2)
        Real bin_size = std::min({Lx / nx, Ly / ny, Lz / nz});
        
        // Estimate memory page size (typical value for memory management)
        
        std::cout << "\nNeighbor list info ..." << std::endl;
        std::cout << "  update every " << update_frequency << " steps, delay " << delay_steps << " steps, check yes" << std::endl;
        std::cout << "  max neighbors/atom: " << max_neighbors <<  std::endl;
        std::cout << "  master list distance cutoff = " << extended_cutoff << std::endl;
        std::cout << "  ghost atom cutoff = " << extended_cutoff << std::endl;
        std::cout << "  binsize = " << bin_size << ", bins = " << nx << " " << ny << " " << nz << std::endl;
        std::cout << "  1 neighbor lists, perpetual/occasional/extra = 1 0 0" << std::endl;
        std::cout << "  (1) pair lj/cut, perpetual" << std::endl;
        std::cout << "      attributes: full, newton off" << std::endl;
        std::cout << "      pair build: full/bin/cell" << std::endl;
        std::cout << "      stencil: full/bin/3d" << std::endl;
        std::cout << "      bin: standard" << std::endl;
        std::cout << std::endl;
        std::cout << "Total # of neighbors = " << total_neighbors << std::endl;
        std::cout << "Ave neighs/atom = " << avg_neighbors << std::endl;
        std::cout << std::endl;
        
        info_printed = true;  // Set flag to prevent future printing
    }

    // Save current positions for next rebuild check (local owned particles only)
    if (check_enabled) {
        last_build_x.resize(local_N);
        last_build_y.resize(local_N);
        last_build_z.resize(local_N);
        for (size_t i = 0; i < local_N; ++i) {
            last_build_x[i] = particles.x[i];
            last_build_y[i] = particles.y[i];
            last_build_z[i] = particles.z[i];
        }
    }
}

void VerletCellListNeighborBuilder::broadcast_neighbor_list(NeighborList& nlist, size_t n, int mpi_rank, int mpi_size) {
    if (mpi_size <= 1) return;

    struct NeighborListHeader {
        int total_neighbors;
        Real cutoff_val;
    } header;

    if (mpi_rank == 0) {
        header.total_neighbors = static_cast<int>(nlist.neighbors.size());
        header.cutoff_val = nlist.cutoff;
    }

    MPI_Bcast(&header, sizeof(header), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (mpi_rank != 0) {
        nlist.cutoff = header.cutoff_val;
        nlist.neighbors.resize(header.total_neighbors);
        nlist.num_neighbors.assign(n, 0);
        nlist.neighbor_offset.assign(n, 0);
    }

    if (header.total_neighbors > 0) {
        MPI_Bcast(nlist.neighbors.data(), header.total_neighbors, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(nlist.num_neighbors.data(), static_cast<int>(n), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(nlist.neighbor_offset.data(), static_cast<int>(n), MPI_INT, 0, MPI_COMM_WORLD);
    }
}

// Morton Z-order curve based neighbor list implementation
// Optimized for MPI parallelism and CUDA/SYCL GPU acceleration

MortonZOrderNeighborBuilder::MortonZOrderNeighborBuilder(Real cutoff_val, Real skin, int tile_size) 
    : cutoff(cutoff_val), verlet_skin(skin), morton_levels(10), update_frequency(1), delay_steps(0), info_printed(false) {
    (void)tile_size; // deprecated
    gpu_memory_allocated = false;
    gpu_available = false;
    gpu_device_id = 0;
    // Displacement check: rebuild only when max displacement > skin/2
    max_displacement_threshold = (verlet_skin * 0.5) * (verlet_skin * 0.5);
    displacement_check_enabled = true;
    has_last_build = false;
}

void MortonZOrderNeighborBuilder::initialize() {   
    // Initialize GPU context if available
    #ifdef USE_CUDA
    cudaError_t err = cudaGetDevice(&gpu_device_id);
    if (err == cudaSuccess) {
        gpu_available = true;
        log_message(1, "CUDA device " + std::to_string(gpu_device_id) + " detected for Morton Z-order neighbor list");
    } else {
        gpu_available = false;
        log_warning("CUDA not available for Morton Z-order, using CPU fallback");
    }
    #elif defined(USE_SYCL)
    try {
        sycl_queue = sycl::queue(sycl::gpu_selector_v);
        gpu_available = true;
        if (is_rank_zero()) {
            std::cout << "SYCL GPU queue created for Morton Z-order neighbor list" << std::endl;
        }
    } catch (const std::exception& e) {
        gpu_available = false;
        if (is_rank_zero()) {
            std::cout << "Warning: SYCL GPU not available, using CPU fallback: " << e.what() << std::endl;
        }
    }
    #else
    gpu_available = false;
    #endif
}

void MortonZOrderNeighborBuilder::finalize() {
    // Clean up GPU memory if allocated with proper error handling
    if (gpu_memory_allocated) {
        #ifdef USE_CUDA
        auto cuda_free_silent = [](void*& p) {
            if (p) {
                cudaFree(p);
                p = nullptr;
            }
        };
        cuda_free_silent(d_x);
        cuda_free_silent(d_y);
        cuda_free_silent(d_z);
        cuda_free_silent(d_perm);
        cuda_free_silent(d_invperm);
        cuda_free_silent(d_x_reord);
        cuda_free_silent(d_y_reord);
        cuda_free_silent(d_z_reord);
        cuda_free_silent(d_neighbor_list_tmp);
        cuda_free_silent(d_num_neighbors_tmp);
        cuda_free_silent(d_neighbor_offset_tmp);
        // Morton-specific buffers
        cuda_free_silent(d_morton_codes);
        cuda_free_silent(d_sorted_indices);
        cuda_free_silent(d_cell_start_morton);
        cuda_free_silent(d_cell_end_morton);
        cuda_free_silent(d_compact_neighbors);
        
        #elif defined(USE_SYCL)
        try {
            // SYCL path not implemented for Morton reorder buffers in this refactor.
        } catch (const std::exception& e) {
            if (is_rank_zero()) {
                std::cout << "Warning: SYCL memory deallocation failed: " << e.what() << std::endl;
            }
        }
        #endif
        gpu_memory_allocated = false;
    }
    
    // Clean up CPU data structures
    morton_codes.clear();
    particle_reorder.clear();
    inverse_reorder.clear();
    last_build_x.clear();
    last_build_y.clear();
    last_build_z.clear();
    has_last_build = false;
    gpu_capacity_particles = 0;
    gpu_capacity_local = 0;
    gpu_capacity_neighbor_slots = 0;
    gpu_capacity_cells = 0;
    
}

// Optimized Morton Z-order encoding using bit interleaving
// This creates a space-filling curve that preserves spatial locality
uint64_t MortonZOrderNeighborBuilder::morton_encode_3d(uint32_t x, uint32_t y, uint32_t z) {
    // Efficient bit interleaving using precomputed masks
    auto split_by_3 = [](uint32_t a) -> uint64_t {
        uint64_t x = a & 0x1fffff; // Use first 21 bits for high resolution
        x = (x | x << 32) & 0x1f00000000ffff;
        x = (x | x << 16) & 0x1f0000ff0000ff;
        x = (x | x << 8)  & 0x100f00f00f00f00f;
        x = (x | x << 4)  & 0x10c30c30c30c30c3;
        x = (x | x << 2)  & 0x1249249249249249;
        return x;
    };
    
    // Interleave bits: z,y,x pattern for better memory coalescing
    return split_by_3(x) | (split_by_3(y) << 1) | (split_by_3(z) << 2);
}

void MortonZOrderNeighborBuilder::morton_decode_3d(uint64_t morton, uint32_t& x, uint32_t& y, uint32_t& z) {
    // Reverse bit interleaving
    auto compact_by_3 = [](uint64_t x) -> uint32_t {
        x = x & 0x1249249249249249;
        x = (x | (x >> 2))  & 0x10c30c30c30c30c3;
        x = (x | (x >> 4))  & 0x100f00f00f00f00f;
        x = (x | (x >> 8))  & 0x1f0000ff0000ff;
        x = (x | (x >> 16)) & 0x1f00000000ffff;
        x = (x | (x >> 32)) & 0x1fffff;
        return static_cast<uint32_t>(x);
    };
    
    x = compact_by_3(morton);
    y = compact_by_3(morton >> 1);
    z = compact_by_3(morton >> 2);
}

void MortonZOrderNeighborBuilder::build_morton_permutation(const ParticleData& particles, const SimulationBox& box,
                                                           size_t n_local, size_t n_total) {
    morton_codes.clear();
    morton_codes.resize(n_total);

    // Use local subdomain extents when present; else use full box.
    const Real Lx = (box.xhi > box.xlo) ? (box.xhi - box.xlo) : box.lx;
    const Real Ly = (box.yhi > box.ylo) ? (box.yhi - box.ylo) : box.ly;
    const Real Lz = (box.zhi > box.zlo) ? (box.zhi - box.zlo) : box.lz;
    const Real origin_x = (box.xhi > box.xlo) ? box.xlo : 0.0;
    const Real origin_y = (box.yhi > box.ylo) ? box.ylo : 0.0;
    const Real origin_z = (box.zhi > box.zlo) ? box.zlo : 0.0;

    const uint32_t grid_res = 1u << morton_levels;
    const Real inv_lx = (Lx > 0) ? (static_cast<Real>(grid_res) / Lx) : 0;
    const Real inv_ly = (Ly > 0) ? (static_cast<Real>(grid_res) / Ly) : 0;
    const Real inv_lz = (Lz > 0) ? (static_cast<Real>(grid_res) / Lz) : 0;

    for (size_t i = 0; i < n_total; ++i) {
        // Shift to local subdomain frame and clamp.
        Real rx = particles.x[i] - origin_x;
        Real ry = particles.y[i] - origin_y;
        Real rz = particles.z[i] - origin_z;
        rx = std::clamp(rx, (Real)0, Lx * (Real)(1.0 - 1e-12));
        ry = std::clamp(ry, (Real)0, Ly * (Real)(1.0 - 1e-12));
        rz = std::clamp(rz, (Real)0, Lz * (Real)(1.0 - 1e-12));

        uint32_t x = static_cast<uint32_t>(rx * inv_lx);
        uint32_t y = static_cast<uint32_t>(ry * inv_ly);
        uint32_t z = static_cast<uint32_t>(rz * inv_lz);
        x = std::min<uint32_t>(x, grid_res - 1);
        y = std::min<uint32_t>(y, grid_res - 1);
        z = std::min<uint32_t>(z, grid_res - 1);

        morton_codes[i].code = morton_encode_3d(x, y, z);
        morton_codes[i].particle_id = static_cast<int>(i);
    }

    // Keep owned particles before ghosts.  Sorting the combined array and then
    // treating [0, n_local) as owned is invalid: a ghost can appear anywhere in
    // Morton order and would make the output indexing write past local-sized
    // arrays.  Spatially sort the two groups independently instead.
    const size_t owned_end = std::min(n_local, n_total);
    std::sort(morton_codes.begin(), morton_codes.begin() + owned_end);
    std::sort(morton_codes.begin() + owned_end, morton_codes.end());

    particle_reorder.assign(n_total, 0);
    inverse_reorder.assign(n_total, 0);
    for (size_t new_i = 0; new_i < n_total; ++new_i) {
        const int old_i = morton_codes[new_i].particle_id;
        particle_reorder[old_i] = static_cast<int>(new_i);
        inverse_reorder[new_i] = old_i;
    }
}

void MortonZOrderNeighborBuilder::build_reordered_position_arrays(
    const ParticleData& particles,
    std::vector<Real>& x_reord,
    std::vector<Real>& y_reord,
    std::vector<Real>& z_reord,
    size_t n_total) const {
    x_reord.resize(n_total);
    y_reord.resize(n_total);
    z_reord.resize(n_total);
    for (size_t new_i = 0; new_i < n_total; ++new_i) {
        const int old_i = inverse_reorder[new_i];
        x_reord[new_i] = particles.x[old_i];
        y_reord[new_i] = particles.y[old_i];
        z_reord[new_i] = particles.z[old_i];
    }
}

// Optimized Morton Z-order neighbor list build with Cell List for O(n) complexity
void MortonZOrderNeighborBuilder::build(ParticleData& particles, SimulationBox& box, NeighborList& nlist, size_t num_local) {
    // LAMMPS-style: num_local = owned particles, num_particles = owned + ghost
    const size_t local_N = (num_local > 0) ? num_local : particles.num_particles;
    const size_t total_N = particles.num_particles;

    // Extended cutoff for neighbor search (Verlet skin included)
    const Real extended_cutoff = cutoff + verlet_skin;
    const Real rcut2 = extended_cutoff * extended_cutoff;
    nlist.cutoff = extended_cutoff;
    nlist.list_type = "morton_z_order";

    // === Displacement check: skip rebuild if max displacement < skin/2 ===
    // This mirrors LAMMPS Verlet skin logic for Morton builder to avoid unnecessary rebuilds.
    // IMPORTANT: In GPU mode, md_simulation.cpp calls cuda_calc->download_from_device()
    // BEFORE calling this build() function, so particles.x/y/z are current.
    if (displacement_check_enabled && has_last_build) {
        const size_t cached = last_build_x.size();
        const bool size_changed = (local_N != cached);
        const size_t check_n = std::min(local_N, cached);
        Real max_disp_sq = 0.0;
        for (size_t i = 0; i < check_n; ++i) {
            Real dx = particles.x[i] - last_build_x[i];
            Real dy = particles.y[i] - last_build_y[i];
            Real dz = particles.z[i] - last_build_z[i];
            box.apply_minimum_image(dx, dy, dz);
            Real dsq = dx*dx + dy*dy + dz*dz;
            if (dsq > max_disp_sq) max_disp_sq = dsq;
        }

        int local_rebuild = (size_changed || max_disp_sq > max_displacement_threshold) ? 1 : 0;
        int global_rebuild = local_rebuild;
        int mpi_init = 0;
        MPI_Initialized(&mpi_init);
        if (mpi_init) {
            MPI_Allreduce(&local_rebuild, &global_rebuild, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        }

        // Preserve the existing list when no rank crossed the Verlet-skin
        // threshold.  Previously the result of this check was discarded and
        // every call continued into a full Morton sort and rebuild.
        if (global_rebuild == 0) return;
    }

    // === CUDA fast path: fully GPU-side Morton sort + neighbor build ===
#ifdef USE_CUDA
    if (gpu_available) {
        auto t_gpu_start = std::chrono::high_resolution_clock::now();

        const int max_neighbors_per_particle = 512;
        const size_t neighbor_slots = local_N * (size_t)max_neighbors_per_particle;

        // Cell grid parameters (same as verlet builder for consistency)
        const Real cell_size = extended_cutoff;
        const int cnx = std::max(1, (int)(box.lx / cell_size));
        const int cny = std::max(1, (int)(box.ly / cell_size));
        const int cnz = std::max(1, (int)(box.lz / cell_size));
        const size_t total_cells = (size_t)cnx * cny * cnz;

        const bool need_realloc =
            (gpu_capacity_particles < total_N) ||
            (gpu_capacity_local < local_N) ||
            (gpu_capacity_neighbor_slots < neighbor_slots) ||
            (gpu_capacity_cells < total_cells) ||
            (!gpu_memory_allocated);

        if (need_realloc) {
            std::cout << "[Morton-GPU] Allocating GPU buffers: N=" << total_N
                      << " local=" << local_N << " cells=" << total_cells
                      << " slots=" << neighbor_slots << std::endl;
            // Free old allocations first
            auto cuda_free_silent = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };
            cuda_free_silent(d_x); cuda_free_silent(d_y); cuda_free_silent(d_z);
            cuda_free_silent(d_x_reord); cuda_free_silent(d_y_reord); cuda_free_silent(d_z_reord);
            cuda_free_silent(d_neighbor_list_tmp);
            cuda_free_silent(d_num_neighbors_tmp);
            cuda_free_silent(d_neighbor_offset_tmp);
            cuda_free_silent(d_morton_codes);
            cuda_free_silent(d_sorted_indices);
            cuda_free_silent(d_cell_start_morton);
            cuda_free_silent(d_cell_end_morton);
            cuda_free_silent(d_perm); cuda_free_silent(d_invperm);
            cuda_free_silent(d_compact_neighbors);

            // Allocate all persistent buffers in one go
            cudaMalloc(&d_x, total_N * sizeof(Real));
            cudaMalloc(&d_y, total_N * sizeof(Real));
            cudaMalloc(&d_z, total_N * sizeof(Real));
            cudaMalloc(&d_x_reord, total_N * sizeof(Real));
            cudaMalloc(&d_y_reord, total_N * sizeof(Real));
            cudaMalloc(&d_z_reord, total_N * sizeof(Real));
            cudaMalloc(&d_neighbor_list_tmp, neighbor_slots * sizeof(int));
            cudaMalloc(&d_num_neighbors_tmp, local_N * sizeof(int));
            cudaMalloc(&d_neighbor_offset_tmp, local_N * sizeof(int));
            cudaMalloc(&d_morton_codes, total_N * sizeof(unsigned long long));
            cudaMalloc(&d_sorted_indices, total_N * sizeof(int));
            cudaMalloc(&d_cell_start_morton, total_cells * sizeof(int));
            cudaMalloc(&d_cell_end_morton, total_cells * sizeof(int));
            cudaMalloc(&d_compact_neighbors, neighbor_slots * sizeof(int));  // worst-case size
            // perm/invperm not needed for GPU-only path but keep for copy_to_gpu
            cudaMalloc(&d_perm, total_N * sizeof(int));
            cudaMalloc(&d_invperm, total_N * sizeof(int));

            gpu_capacity_particles = total_N;
            gpu_capacity_local = local_N;
            gpu_capacity_neighbor_slots = neighbor_slots;
            gpu_capacity_cells = total_cells;
            gpu_memory_allocated = true;
            std::cout << "[Morton-GPU] GPU buffer allocation complete" << std::endl;
        }

        // Upload original (unsorted) positions to GPU using async transfer
        cudaStream_t stream = 0;
        auto t_upload_start = std::chrono::high_resolution_clock::now();

        cudaMemcpyAsync(d_x, particles.x.data(), total_N * sizeof(Real), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_y, particles.y.data(), total_N * sizeof(Real), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_z, particles.z.data(), total_N * sizeof(Real), cudaMemcpyHostToDevice, stream);

        // Launch GPU-side Morton sort + neighbor list build
        // This calls: compute_morton_codes → thrust::sort_by_key → reorder_particles
        //             → compute_cell_ranges → build_neighbor_list_sorted
        //             → thrust::exclusive_scan → compact_neighbor_list
        launch_cuda_morton_neighbor_list_build(
            (const Real*)d_x, (const Real*)d_y, (const Real*)d_z,
            (int*)d_neighbor_list_tmp,
            (int*)d_num_neighbors_tmp,
            (int*)d_neighbor_offset_tmp,
            (int)local_N, (int)total_N,
            cutoff, verlet_skin,
            box.lx, box.ly, box.lz,
            box.periodic_x, box.periodic_y, box.periodic_z,
            max_neighbors_per_particle,
            stream,
            (Real*)d_x_reord, (Real*)d_y_reord, (Real*)d_z_reord,
            (int*)d_sorted_indices,
            (unsigned long long*)d_morton_codes,
            (int*)d_cell_start_morton, (int*)d_cell_end_morton,
            (int*)d_compact_neighbors,
            nullptr  // d_total_neighbors_out: computed on host
        );

        cudaStreamSynchronize(stream);
        auto t_kernel_end = std::chrono::high_resolution_clock::now();

        // Download compact results (much smaller than padded array)
        // After kernel: d_neighbor_offset_tmp[i] = compact CSR offset, d_num_neighbors_tmp[i] = count
        std::vector<int> h_num_neighbors(local_N);
        std::vector<int> h_neighbor_offset(local_N);
        cudaMemcpyAsync(h_num_neighbors.data(), d_num_neighbors_tmp, local_N * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_neighbor_offset.data(), d_neighbor_offset_tmp, local_N * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // Total neighbors = offset[last] + count[last]
        size_t total_neighbors = 0;
        if (local_N > 0) {
            total_neighbors = (size_t)h_neighbor_offset[local_N - 1] + (size_t)h_num_neighbors[local_N - 1];
        }

        // Download only the compact neighbor array (NOT the full padded array)
        std::vector<int> h_compact_neighbors(total_neighbors);
        if (total_neighbors > 0) {
            cudaMemcpy(h_compact_neighbors.data(), d_compact_neighbors, total_neighbors * sizeof(int), cudaMemcpyDeviceToHost);
        }

        auto t_download_end = std::chrono::high_resolution_clock::now();

        // Build output arrays directly from compact data (already in original index order)
        nlist.neighbors.assign(h_compact_neighbors.begin(), h_compact_neighbors.end());
        nlist.num_neighbors.assign(h_num_neighbors.begin(), h_num_neighbors.end());
        nlist.neighbor_offset.assign(h_neighbor_offset.begin(), h_neighbor_offset.end());

        auto t_gpu_end = std::chrono::high_resolution_clock::now();

        auto ms_kernel = std::chrono::duration_cast<std::chrono::microseconds>(t_kernel_end - t_upload_start).count();
        auto ms_download = std::chrono::duration_cast<std::chrono::microseconds>(t_download_end - t_kernel_end).count();
        auto ms_total = std::chrono::duration_cast<std::chrono::microseconds>(t_gpu_end - t_gpu_start).count();

        if (!info_printed) {
            int max_neighbors = 0;
            if (!nlist.num_neighbors.empty()) {
                max_neighbors = *std::max_element(nlist.num_neighbors.begin(), nlist.num_neighbors.end());
            }
            Real avg_n = (local_N > 0) ? (Real)total_neighbors / (Real)local_N : 0.0;
            Real bin_size = extended_cutoff;
            std::cout << "\nNeighbor list info ..." << std::endl;
            std::cout << "  update every " << update_frequency << " steps, delay " << delay_steps << " steps, check yes" << std::endl;
            std::cout << "  max neighbors/atom: " << max_neighbors << std::endl;
            std::cout << "  master list distance cutoff = " << extended_cutoff << std::endl;
            std::cout << "  ghost atom cutoff = " << extended_cutoff << std::endl;
            std::cout << "  binsize = " << bin_size << ", bins = " << cnx << " " << cny << " " << cnz << std::endl;
            std::cout << "  [Morton Z-Order GPU path: thrust sort + warp-per-particle search]" << std::endl;
            std::cout << std::endl;
            std::cout << "Total # of neighbors = " << total_neighbors << std::endl;
            std::cout << "Ave neighs/atom = " << avg_n << std::endl;
            std::cout << std::endl;
            info_printed = true;
        }

        // Save current positions for displacement check on next call
        last_build_x.assign(particles.x.begin(), particles.x.begin() + local_N);
        last_build_y.assign(particles.y.begin(), particles.y.begin() + local_N);
        last_build_z.assign(particles.z.begin(), particles.z.begin() + local_N);
        has_last_build = true;

        return;
    }
#endif

    // === CPU path: Morton-sorted cell list with flat-array two-pass build ===
    auto t_cpu_start = std::chrono::high_resolution_clock::now();

    // Step 1: Build Morton permutation for spatial locality
    auto t0 = std::chrono::high_resolution_clock::now();
    build_morton_permutation(particles, box, local_N, total_N);
    auto t1 = std::chrono::high_resolution_clock::now();

    // Step 2: Build reordered position arrays
    std::vector<Real> x_reord, y_reord, z_reord;
    build_reordered_position_arrays(particles, x_reord, y_reord, z_reord, total_N);
    auto t2 = std::chrono::high_resolution_clock::now();

    // Local subdomain extent (MPI-aware) or full box.
    const Real Lx = (box.xhi > box.xlo) ? (box.xhi - box.xlo) : box.lx;
    const Real Ly = (box.yhi > box.ylo) ? (box.yhi - box.ylo) : box.ly;
    const Real Lz = (box.zhi > box.zlo) ? (box.zhi - box.zlo) : box.lz;

    // Use same binning strategy as VerletCellListNeighborBuilder for apples-to-apples.
    const Real target_bin_size = extended_cutoff;
    // floor guarantees actual bin widths >= the search radius.  Rounding can
    // make bins slightly smaller and expand a 27-cell stencil to 125 cells.
    const int nx = std::max(1, static_cast<int>(Lx / target_bin_size));
    const int ny = std::max(1, static_cast<int>(Ly / target_bin_size));
    const int nz = std::max(1, static_cast<int>(Lz / target_bin_size));

    const Real actual_bin_size_x = Lx / nx;
    const Real actual_bin_size_y = Ly / ny;
    const Real actual_bin_size_z = Lz / nz;
    const Real max_bin_size = std::max({actual_bin_size_x, actual_bin_size_y, actual_bin_size_z});
    int stencil_range = static_cast<int>(std::ceil(extended_cutoff / max_bin_size));
    stencil_range = std::max(1, stencil_range);

    const Real inv_lx_cell = nx / Lx;
    const Real inv_ly_cell = ny / Ly;
    const Real inv_lz_cell = nz / Lz;

    const int total_cells = nx * ny * nz;

    // Subdomain origin
    const Real origin_x = (box.xhi > box.xlo) ? box.xlo : 0.0;
    const Real origin_y = (box.yhi > box.ylo) ? box.ylo : 0.0;
    const Real origin_z = (box.zhi > box.zlo) ? box.zlo : 0.0;

    // Step 3: Build cell arrays from Morton-reordered particles
    // Use flat arrays (cell_start + cell_count) instead of vector<vector<int>> for locality
    std::vector<int> cx(total_N), cy_arr(total_N), cz_arr(total_N);
    std::vector<int> cell_ids(total_N, -1);
    std::vector<int> cell_counts_arr(total_cells, 0);

    for (size_t new_i = 0; new_i < total_N; ++new_i) {
        Real rx = x_reord[new_i] - origin_x;
        Real ry = y_reord[new_i] - origin_y;
        Real rz = z_reord[new_i] - origin_z;
        int ix = std::clamp(static_cast<int>(rx * inv_lx_cell), 0, nx - 1);
        int iy = std::clamp(static_cast<int>(ry * inv_ly_cell), 0, ny - 1);
        int iz = std::clamp(static_cast<int>(rz * inv_lz_cell), 0, nz - 1);
        cx[new_i] = ix;
        cy_arr[new_i] = iy;
        cz_arr[new_i] = iz;
        const int cid = iz * nx * ny + iy * nx + ix;
        cell_ids[new_i] = cid;
        cell_counts_arr[cid]++;
    }

    // Build cell start offsets (prefix sum)
    std::vector<int> cell_start_arr(total_cells + 1, 0);
    for (int c = 0; c < total_cells; ++c) {
        cell_start_arr[c + 1] = cell_start_arr[c] + cell_counts_arr[c];
    }

    // Fill flat cell particle array (sorted order)
    std::vector<int> cell_fill(total_cells, 0);
    std::vector<int> flat_cell_particles(total_N);
    for (size_t new_i = 0; new_i < total_N; ++new_i) {
        const int cid = cell_ids[new_i];
        if (cid >= 0 && cid < total_cells) {
            flat_cell_particles[cell_start_arr[cid] + cell_fill[cid]] = (int)new_i;
            cell_fill[cid]++;
        }
    }

    auto t3 = std::chrono::high_resolution_clock::now();

    // Step 4: Two-pass flat-array neighbor list build (avoids vector<vector<int>> heap fragmentation)
    // Pass 1: count neighbors per atom (parallel)
    std::vector<int> count(local_N, 0);

    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t new_i = 0; new_i < local_N; ++new_i) {
        int cnt = 0;
        const int ix = cx[new_i];
        const int iy = cy_arr[new_i];
        const int iz = cz_arr[new_i];
        const Real xi = x_reord[new_i];
        const Real yi = y_reord[new_i];
        const Real zi = z_reord[new_i];

        for (int dz = -stencil_range; dz <= stencil_range; ++dz) {
            int zc = iz + dz;
            if (box.periodic_z) {
                if (zc < 0) zc += nz; else if (zc >= nz) zc -= nz;
            } else { if (zc < 0 || zc >= nz) continue; }

            for (int dy = -stencil_range; dy <= stencil_range; ++dy) {
                int yc = iy + dy;
                if (box.periodic_y) {
                    if (yc < 0) yc += ny; else if (yc >= ny) yc -= ny;
                } else { if (yc < 0 || yc >= ny) continue; }

                for (int dx = -stencil_range; dx <= stencil_range; ++dx) {
                    int xc = ix + dx;
                    if (box.periodic_x) {
                        if (xc < 0) xc += nx; else if (xc >= nx) xc -= nx;
                    } else { if (xc < 0 || xc >= nx) continue; }

                    const int cid = zc * nx * ny + yc * nx + xc;
                    if (cid < 0 || cid >= total_cells) continue;

                    const int cstart = cell_start_arr[cid];
                    const int cend   = cell_start_arr[cid + 1];
                    for (int ki = cstart; ki < cend; ++ki) {
                        const int new_j = flat_cell_particles[ki];
                        if (new_j == (int)new_i) continue;
                        Real ddx = xi - x_reord[new_j];
                        Real ddy = yi - y_reord[new_j];
                        Real ddz = zi - z_reord[new_j];
                        box.apply_minimum_image(ddx, ddy, ddz);
                        if (ddx*ddx + ddy*ddy + ddz*ddz < rcut2) ++cnt;
                    }
                }
            }
        }
        count[new_i] = cnt;
    }

    // Prefix sum for offsets (in Morton new_i order for contiguous writes)
    // We need to map new_i -> old_i for final output
    std::vector<int> new_offsets(local_N, 0);
    size_t total_neighbors_flat = 0;
    for (size_t new_i = 0; new_i < local_N; ++new_i) {
        new_offsets[new_i] = (int)total_neighbors_flat;
        total_neighbors_flat += (size_t)count[new_i];
    }

    // Initialize output arrays (indexed by original particle id)
    nlist.neighbors.resize(total_neighbors_flat);
    nlist.num_neighbors.assign(local_N, 0);
    nlist.neighbor_offset.assign(local_N, 0);

    // Fill num_neighbors and neighbor_offset in original indexing
    for (size_t new_i = 0; new_i < local_N; ++new_i) {
        const int old_i = inverse_reorder[new_i];
        nlist.num_neighbors[old_i] = count[new_i];
        nlist.neighbor_offset[old_i] = new_offsets[new_i];
    }

    auto t4 = std::chrono::high_resolution_clock::now();

    // Pass 2: fill neighbor list (parallel, writing to pre-computed offsets)
    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t new_i = 0; new_i < local_N; ++new_i) {
        const int ix = cx[new_i];
        const int iy = cy_arr[new_i];
        const int iz = cz_arr[new_i];
        const Real xi = x_reord[new_i];
        const Real yi = y_reord[new_i];
        const Real zi = z_reord[new_i];
        size_t write_pos = (size_t)new_offsets[new_i];

        for (int dz = -stencil_range; dz <= stencil_range; ++dz) {
            int zc = iz + dz;
            if (box.periodic_z) {
                if (zc < 0) zc += nz; else if (zc >= nz) zc -= nz;
            } else { if (zc < 0 || zc >= nz) continue; }

            for (int dy = -stencil_range; dy <= stencil_range; ++dy) {
                int yc = iy + dy;
                if (box.periodic_y) {
                    if (yc < 0) yc += ny; else if (yc >= ny) yc -= ny;
                } else { if (yc < 0 || yc >= ny) continue; }

                for (int dx = -stencil_range; dx <= stencil_range; ++dx) {
                    int xc = ix + dx;
                    if (box.periodic_x) {
                        if (xc < 0) xc += nx; else if (xc >= nx) xc -= nx;
                    } else { if (xc < 0 || xc >= nx) continue; }

                    const int cid = zc * nx * ny + yc * nx + xc;
                    if (cid < 0 || cid >= total_cells) continue;

                    const int cstart = cell_start_arr[cid];
                    const int cend   = cell_start_arr[cid + 1];
                    for (int ki = cstart; ki < cend; ++ki) {
                        const int new_j = flat_cell_particles[ki];
                        if (new_j == (int)new_i) continue;
                        Real ddx = xi - x_reord[new_j];
                        Real ddy = yi - y_reord[new_j];
                        Real ddz = zi - z_reord[new_j];
                        box.apply_minimum_image(ddx, ddy, ddz);
                        if (ddx*ddx + ddy*ddy + ddz*ddz < rcut2) {
                            // Map reordered index back to original
                            nlist.neighbors[write_pos++] = inverse_reorder[(size_t)new_j];
                        }
                    }
                }
            }
        }
    }

    auto t5 = std::chrono::high_resolution_clock::now();

    auto us_sort  = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    auto us_reord = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    auto us_cells = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();
    auto us_pass1 = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
    auto us_pass2 = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();
    auto us_total = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t_cpu_start).count();

    if (!info_printed) {
        std::cout << "[Morton-CPU] Build complete:"
                  << " sort=" << us_sort << "us"
                  << " reorder=" << us_reord << "us"
                  << " cells=" << us_cells << "us"
                  << " pass1=" << us_pass1 << "us"
                  << " pass2=" << us_pass2 << "us"
                  << " total=" << us_total << "us"
                  << " N=" << total_N << " local=" << local_N
                  << " neighbors=" << total_neighbors_flat << std::endl;
    }

    if (!info_printed) {
        int max_neighbors = 0;
        if (!nlist.num_neighbors.empty()) {
            max_neighbors = *std::max_element(nlist.num_neighbors.begin(), nlist.num_neighbors.end());
        }
        Real avg_n = (local_N > 0) ? (Real)total_neighbors_flat / (Real)local_N : 0.0;
        Real bin_size = std::min({Lx / nx, Ly / ny, Lz / nz});

        std::cout << "\nNeighbor list info ..." << std::endl;
        std::cout << "  update every " << update_frequency << " steps, delay " << delay_steps << " steps, check yes" << std::endl;
        std::cout << "  max neighbors/atom: " << max_neighbors << std::endl;
        std::cout << "  master list distance cutoff = " << extended_cutoff << std::endl;
        std::cout << "  ghost atom cutoff = " << extended_cutoff << std::endl;
        std::cout << "  binsize = " << bin_size << ", bins = " << nx << " " << ny << " " << nz << std::endl;
        std::cout << "  [Morton Z-Order CPU path: Morton sort + flat-array two-pass build]" << std::endl;
        std::cout << std::endl;
        std::cout << "Total # of neighbors = " << total_neighbors_flat << std::endl;
        std::cout << "Ave neighs/atom = " << avg_n << std::endl;
        std::cout << std::endl;
        info_printed = true;
    }

    if (gpu_available) {
        copy_to_gpu(nlist);
    }

    // Save current positions for displacement check on next call
    last_build_x.assign(particles.x.begin(), particles.x.begin() + local_N);
    last_build_y.assign(particles.y.begin(), particles.y.begin() + local_N);
    last_build_z.assign(particles.z.begin(), particles.z.begin() + local_N);
    has_last_build = true;
}

void MortonZOrderNeighborBuilder::copy_to_gpu(NeighborList& nlist) {
    if (!gpu_available) {
        return;
    }
    #ifdef USE_CUDA
    // copy_to_gpu is a no-op: CUDA force calculators read from NeighborList directly.
    (void)nlist;
    return;
    #endif
}


// Factory function implementation
std::unique_ptr<NeighborListBuilder> create_neighbor_list_builder(
    NeighborListType type, Real cutoff, const std::string& config, int mpi_rank) {
    
    int rank = mpi_rank;
    
    switch (type) {
        case NeighborListType::VERLET_CELL_LIST: {
            Real skin = 1.0; // Default Verlet skin
            
            // Parse config string for skin distance if provided
            if (!config.empty()) {
                try {
                    skin = std::stod(config);
                } catch (...) {
                    if (rank == 0) {
                        std::cout << "Warning: Invalid skin distance in config, using default 1.0" << std::endl;
                    }
                }
            }
            
            if (rank == 0) {
                std::cout << "Creating Verlet-Cell list neighbor builder (skin: " 
                          << skin << ")" << std::endl;
            }
            return std::make_unique<VerletCellListNeighborBuilder>(cutoff, skin);
        }
            
        case NeighborListType::MORTON_Z_ORDER_CSR: {
            Real skin = 2.0; // Default skin for stability
            int tile_size = 256; // Default tile size (optimized for GPU warps)
            
            // Parse config string for skin distance if provided
            if (!config.empty()) {
                try {
                    skin = std::stod(config);
                } catch (...) {
                    if (rank == 0) {
                        std::cout << "Warning: Invalid skin distance in config, using default 2.0" << std::endl;
                    }
                }
            }
            
            if (rank == 0) {
                std::cout << "Creating Morton Z-order CSR neighbor builder (skin: " 
                          << skin << ", tile size: " << tile_size << ")" << std::endl;
            }
            return std::make_unique<MortonZOrderNeighborBuilder>(cutoff, skin, tile_size);
        }
        
        default:
            if (rank == 0) {
                std::cout << "Unknown neighbor list type, defaulting to verlet cell list" << std::endl;
            }
            return std::make_unique<VerletCellListNeighborBuilder>(cutoff, 2.0);
    }
}
