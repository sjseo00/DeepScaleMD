#include "MLIPStep.cuh"

#include <cub/cub.cuh>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#define MLIP_STEP_CUDA_CHECK(call)                                                                              \
    do {                                                                                                        \
        cudaError_t step_err_ = (call);                                                                         \
        if (step_err_ != cudaSuccess) {                                                                         \
            throw std::runtime_error(std::string("MLIPStep: ") + #call + ": " + cudaGetErrorString(step_err_)); \
        }                                                                                                       \
    } while (0)

namespace {

constexpr int kThreads = 256;

// What the image kernels need, by value.
struct ImageSpec {
    double box[3];
    double origin[3];       // lo of the subdomain: where an image axis is wrapped from
    double region_lo[3];    // images are kept inside [region_lo, region_hi] on an image axis
    double region_hi[3];
    int shells[3];          // shifts run over -shells..shells; 0 on an axis without images
};

__device__ __forceinline__ double wrap_coord(double v, double origin, double box, int shells) {
    return shells > 0 ? v - box * floor((v - origin) / box) : v;
}

// Calls emit(shifted x, y, z) for every periodic image of (px, py, pz) inside the region, in a fixed order.
template <class Emit>
__device__ __forceinline__ void for_each_image(const ImageSpec& s, double px, double py, double pz, Emit emit) {
    for (int sx = -s.shells[0]; sx <= s.shells[0]; ++sx) {
        const double qx = px + sx * s.box[0];
        if (s.shells[0] > 0 && (qx < s.region_lo[0] || qx > s.region_hi[0])) continue;
        for (int sy = -s.shells[1]; sy <= s.shells[1]; ++sy) {
            const double qy = py + sy * s.box[1];
            if (s.shells[1] > 0 && (qy < s.region_lo[1] || qy > s.region_hi[1])) continue;
            for (int sz = -s.shells[2]; sz <= s.shells[2]; ++sz) {
                if (sx == 0 && sy == 0 && sz == 0) continue;
                const double qz = pz + sz * s.box[2];
                if (s.shells[2] > 0 && (qz < s.region_lo[2] || qz > s.region_hi[2])) continue;
                emit(qx, qy, qz);
            }
        }
    }
}

__global__ void count_images_kernel(int64_t n, const double* x, const double* y, const double* z, ImageSpec spec,
                                    int64_t* counts) {
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    const double px = wrap_coord(x[i], spec.origin[0], spec.box[0], spec.shells[0]);
    const double py = wrap_coord(y[i], spec.origin[1], spec.box[1], spec.shells[1]);
    const double pz = wrap_coord(z[i], spec.origin[2], spec.box[2], spec.shells[2]);
    int64_t count = 0;
    for_each_image(spec, px, py, pz, [&](double, double, double) { ++count; });
    counts[i] = count;
}

__global__ void total_kernel(int64_t n, const int64_t* offsets, const int64_t* counts, long long* total) {
    *total = offsets[n - 1] + counts[n - 1];
}

// Writes the (wrapped) MD atoms and then their images behind them: the images of atom i fill the slots
// n + offsets[i] ... n + offsets[i] + counts[i] - 1. With eowner (edge shifts) it also writes, for every slot, the MD atom it
// copies and its offset from that atom's (wrapped) position.
__global__ void extend_kernel(int64_t n, const double* x, const double* y, const double* z, const int* type,
                              ImageSpec spec, const int64_t* offsets, double* ex, double* ey, double* ez,
                              int* etype, int32_t* eowner, double* esx, double* esy, double* esz) {
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    const double px = wrap_coord(x[i], spec.origin[0], spec.box[0], spec.shells[0]);
    const double py = wrap_coord(y[i], spec.origin[1], spec.box[1], spec.shells[1]);
    const double pz = wrap_coord(z[i], spec.origin[2], spec.box[2], spec.shells[2]);
    ex[i] = px;
    ey[i] = py;
    ez[i] = pz;
    etype[i] = type[i];
    if (eowner) {
        eowner[i] = static_cast<int32_t>(i);
        esx[i] = 0.0;
        esy[i] = 0.0;
        esz[i] = 0.0;
    }
    int64_t slot = n + offsets[i];
    for_each_image(spec, px, py, pz, [&](double qx, double qy, double qz) {
        ex[slot] = qx;
        ey[slot] = qy;
        ez[slot] = qz;
        etype[slot] = type[i];
        if (eowner) {
            eowner[slot] = static_cast<int32_t>(i);
            esx[slot] = qx - px;
            esy[slot] = qy - py;
            esz[slot] = qz - pz;
        }
        ++slot;
    });
}

// md_f[i] += f[i] + sum of the forces on the images of i, summed in slot order.
__global__ void add_forces_kernel(int64_t n, const double* fx, const double* fy, const double* fz,
                                  const int64_t* counts, const int64_t* offsets, double* mx, double* my, double* mz) {
    const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    double ax = fx[i], ay = fy[i], az = fz[i];
    if (counts) {
        const int64_t first = n + offsets[i];
        for (int64_t k = 0; k < counts[i]; ++k) {
            ax += fx[first + k];
            ay += fy[first + k];
            az += fz[first + k];
        }
    }
    mx[i] += ax;
    my[i] += ay;
    mz[i] += az;
}

__global__ void virial_terms_kernel(int64_t n, const double* x, const double* y, const double* z, const double* fx,
                                    const double* fy, const double* fz, double* terms) {
    const int64_t p = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (p >= n) return;
    terms[p] = x[p] * fx[p] + y[p] * fy[p] + z[p] * fz[p];
}

inline unsigned blocks_for(int64_t n) { return static_cast<unsigned>((n + kThreads - 1) / kThreads); }

}  // namespace

MLIPStep::MLIPStep() {
    MLIP_STEP_CUDA_CHECK(cudaMalloc(&d_total_, sizeof(long long)));
    MLIP_STEP_CUDA_CHECK(cudaMallocHost(&h_total_, sizeof(long long)));
    MLIP_STEP_CUDA_CHECK(cudaMalloc(&d_sum_, sizeof(double)));
    MLIP_STEP_CUDA_CHECK(cudaMallocHost(&h_sum_, sizeof(double)));
}

MLIPStep::~MLIPStep() {
    cudaFree(d_ext_x_);
    cudaFree(d_ext_y_);
    cudaFree(d_ext_z_);
    cudaFree(d_ext_type_);
    cudaFree(d_ext_owner_);
    cudaFree(d_ext_sx_);
    cudaFree(d_ext_sy_);
    cudaFree(d_ext_sz_);
    cudaFree(d_counts_);
    cudaFree(d_offsets_);
    cudaFree(d_fx_);
    cudaFree(d_fy_);
    cudaFree(d_fz_);
    cudaFree(d_terms_);
    cudaFree(d_temp_);
    cudaFree(d_total_);
    cudaFreeHost(h_total_);
    cudaFree(d_sum_);
    cudaFreeHost(h_sum_);
}

template <class T>
void MLIPStep::ensure(T*& ptr, size_t& capacity, size_t needed) {
    if (needed <= capacity) return;
    const size_t grown = std::max(needed, capacity + capacity / 2);
    MLIP_STEP_CUDA_CHECK(cudaFree(ptr));   // synchronizes the device, only when a buffer grows
    ptr = nullptr;
    MLIP_STEP_CUDA_CHECK(cudaMalloc(&ptr, grown * sizeof(T)));
    capacity = grown;
}

void MLIPStep::prepare(const double* d_x, const double* d_y, const double* d_z, const int* d_type, int64_t n_owned,
                       int64_t n_md, double cutoff, const MLIPStepGeometry& g, cudaStream_t stream) {
    stream_ = stream;
    n_owned_ = n_owned;
    n_md_ = n_md;
    md_x_ = d_x; md_y_ = d_y; md_z_ = d_z; md_type_ = d_type;
    have_images_ = false;
    shift_active_ = false;
    n_all_ = n_md;
    int64_t n_ext = n_md;       // atoms the edge search sees: the MD atoms and their images

    const bool any_image = g.image_axis[0] || g.image_axis[1] || g.image_axis[2];
    // How far beyond the box an image is wanted on an image axis: as deep as the model reaches, or, with edge shifts, only
    // as far as a neighbor of an atom can be (the images are searched for neighbors, the model never sees them as atoms).
    const bool want_shifts = g.edge_shifts && any_image;
    const double image_reach = want_shifts ? cutoff + 1e-8 : g.depth + g.slack;
    ImageSpec spec;
    for (int a = 0; a < 3; ++a) {
        spec.box[a] = g.box[a];
        spec.origin[a] = g.lo[a];
        spec.region_lo[a] = g.lo[a] - image_reach;
        spec.region_hi[a] = g.hi[a] + image_reach;
        spec.shells[a] = 0;
        if (g.image_axis[a]) {
            if (!(g.box[a] > 0.0)) throw std::invalid_argument("MLIPStep::prepare: image axis with a non-positive box length");
            spec.shells[a] = std::max(1, static_cast<int>(std::ceil(image_reach / g.box[a])));
        }
    }

    if (n_md > 0 && any_image) {
        ensure(d_counts_, cap_counts_, static_cast<size_t>(n_md));
        ensure(d_offsets_, cap_offsets_, static_cast<size_t>(n_md));
        size_t scan_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, d_counts_, d_offsets_, static_cast<int>(n_md), stream);
        ensure(d_temp_, cap_temp_, scan_bytes);

        count_images_kernel<<<blocks_for(n_md), kThreads, 0, stream>>>(n_md, d_x, d_y, d_z, spec, d_counts_);
        cub::DeviceScan::ExclusiveSum(d_temp_, scan_bytes, d_counts_, d_offsets_, static_cast<int>(n_md), stream);
        total_kernel<<<1, 1, 0, stream>>>(n_md, d_offsets_, d_counts_, d_total_);
        MLIP_STEP_CUDA_CHECK(cudaGetLastError());
        MLIP_STEP_CUDA_CHECK(cudaMemcpyAsync(h_total_, d_total_, sizeof(long long), cudaMemcpyDeviceToHost, stream));
        MLIP_STEP_CUDA_CHECK(cudaStreamSynchronize(stream));
        const int64_t n_images = *h_total_;

        if (n_images > 0) {
            have_images_ = true;
            shift_active_ = want_shifts;
            n_ext = n_md + n_images;
            n_all_ = shift_active_ ? n_md : n_ext;      // what the model sees
            const size_t total = static_cast<size_t>(n_ext);
            ensure(d_ext_x_, cap_ext_x_, total);
            ensure(d_ext_y_, cap_ext_y_, total);
            ensure(d_ext_z_, cap_ext_z_, total);
            ensure(d_ext_type_, cap_ext_type_, total);
            if (shift_active_) {
                ensure(d_ext_owner_, cap_ext_owner_, total);
                ensure(d_ext_sx_, cap_ext_sx_, total);
                ensure(d_ext_sy_, cap_ext_sy_, total);
                ensure(d_ext_sz_, cap_ext_sz_, total);
            }
            extend_kernel<<<blocks_for(n_md), kThreads, 0, stream>>>(
                n_md, d_x, d_y, d_z, d_type, spec, d_offsets_, d_ext_x_, d_ext_y_, d_ext_z_, d_ext_type_,
                shift_active_ ? d_ext_owner_ : nullptr, shift_active_ ? d_ext_sx_ : nullptr,
                shift_active_ ? d_ext_sy_ : nullptr, shift_active_ ? d_ext_sz_ : nullptr);
            MLIP_STEP_CUDA_CHECK(cudaGetLastError());
        } else {
            // No image reaches into the region, but the atoms were not wrapped: keep the MD arrays as they are.
        }
    }

    const double* x = have_images_ ? d_ext_x_ : d_x;
    const double* y = have_images_ ? d_ext_y_ : d_y;
    const double* z = have_images_ ? d_ext_z_ : d_z;

    // Work arrays for the model's forces, zeroed.
    const size_t total = static_cast<size_t>(std::max<int64_t>(n_all_, 1));
    ensure(d_fx_, cap_fx_, total);
    ensure(d_fy_, cap_fy_, total);
    ensure(d_fz_, cap_fz_, total);
    MLIP_STEP_CUDA_CHECK(cudaMemsetAsync(d_fx_, 0, total * sizeof(double), stream));
    MLIP_STEP_CUDA_CHECK(cudaMemsetAsync(d_fy_, 0, total * sizeof(double), stream));
    MLIP_STEP_CUDA_CHECK(cudaMemsetAsync(d_fz_, 0, total * sizeof(double), stream));

    // Edges over everything the model sees. The box is the subdomain plus everything that can lie around it.
    const double reach = g.depth + 2.0 * g.slack + 1.0;
    double box_lo[3], box_hi[3];
    for (int a = 0; a < 3; ++a) {
        box_lo[a] = g.lo[a] - reach;
        box_hi[a] = g.hi[a] + reach;
    }
    if (shift_active_) {
        EdgeBuilder::Images images;
        images.n_centers = n_md;
        images.d_owner = d_ext_owner_;
        images.d_shift_x = d_ext_sx_;
        images.d_shift_y = d_ext_sy_;
        images.d_shift_z = d_ext_sz_;
        edges_.build(x, y, z, n_ext, cutoff, stream, box_lo, box_hi, &images);
    } else {
        edges_.build(x, y, z, n_ext, cutoff, stream, box_lo, box_hi);
    }
}

MLIPInputs MLIPStep::inputs() const {
    MLIPInputs in;
    in.d_x = have_images_ ? d_ext_x_ : md_x_;
    in.d_y = have_images_ ? d_ext_y_ : md_y_;
    in.d_z = have_images_ ? d_ext_z_ : md_z_;
    in.d_type = have_images_ ? d_ext_type_ : md_type_;
    in.d_edge_index = edges_.edge_index();
    in.d_edge_shift = shift_active_ ? edges_.edge_shift() : nullptr;
    in.n_owned = n_owned_;
    in.n_all = n_all_;
    in.n_edges = edges_.edge_count();
    in.stream = stream_;
    return in;
}

MLIPOutputs MLIPStep::outputs(bool need_energy) const {
    MLIPOutputs out;
    out.d_fx = d_fx_;
    out.d_fy = d_fy_;
    out.d_fz = d_fz_;
    out.need_energy = need_energy;
    return out;
}

void MLIPStep::add_forces(double* d_fx, double* d_fy, double* d_fz, cudaStream_t stream) {
    if (n_md_ <= 0) return;
    add_forces_kernel<<<blocks_for(n_md_), kThreads, 0, stream>>>(n_md_, d_fx_, d_fy_, d_fz_,
                                                                   (have_images_ && !shift_active_) ? d_counts_ : nullptr, d_offsets_,
                                                                   d_fx, d_fy, d_fz);
    MLIP_STEP_CUDA_CHECK(cudaGetLastError());
}

double MLIPStep::virial(cudaStream_t stream) {
    if (shift_active_) {
        // The images are not atoms the model saw, so the sum of r.F over atoms misses the shift term: the model's own
        // virial has it (strain derivative, MDShiftWrapper).
        if (!model_virial_valid_) {
            throw std::runtime_error("MLIPStep::virial: the model with edge shifts returned no virial (set_model_virial)");
        }
        return model_virial_;
    }
    if (n_all_ <= 0) return 0.0;
    ensure(d_terms_, cap_terms_, static_cast<size_t>(n_all_));
    size_t bytes = 0;
    cub::DeviceReduce::Sum(nullptr, bytes, d_terms_, d_sum_, static_cast<int>(n_all_), stream);
    ensure(d_temp_, cap_temp_, bytes);
    const double* x = have_images_ ? d_ext_x_ : md_x_;
    const double* y = have_images_ ? d_ext_y_ : md_y_;
    const double* z = have_images_ ? d_ext_z_ : md_z_;
    virial_terms_kernel<<<blocks_for(n_all_), kThreads, 0, stream>>>(n_all_, x, y, z, d_fx_, d_fy_, d_fz_, d_terms_);
    cub::DeviceReduce::Sum(d_temp_, bytes, d_terms_, d_sum_, static_cast<int>(n_all_), stream);
    MLIP_STEP_CUDA_CHECK(cudaGetLastError());
    MLIP_STEP_CUDA_CHECK(cudaMemcpyAsync(h_sum_, d_sum_, sizeof(double), cudaMemcpyDeviceToHost, stream));
    MLIP_STEP_CUDA_CHECK(cudaStreamSynchronize(stream));
    return *h_sum_;
}
