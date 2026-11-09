// Restart files: the state a run needs to continue where an earlier run stopped.
//
// Binary, native byte order (the cluster and this code are little-endian x86-64 / aarch64), all reals double:
//   char[8]  "DSMDRST1"
//   int64    step, number of atoms
//   double   timestep (fs), box lx ly lz (A)
//   int32    periodic x y z
//   string   thermostat name, then int32 count + double[count] thermostat state (Nose-Hoover: xi)
//   string   barostat name,   then int32 count + double[count] barostat state (Parrinello-Rahman: h, h_dot)
//   double   x[N] y[N] z[N] (A), vx[N] vy[N] vz[N] (A/fs), in global atom id order (= coordinate file order)
//   char[8]  "DSMDEND1"
// A string is an int32 length and that many chars.
//
// Only positions, velocities, box, step and the thermostat/barostat variables are kept; the topology, force field
// and every setting still come from the YAML file and the files it names. Random number generators (Langevin,
// Andersen, the barostat piston noise) start new streams, so a stochastic run continues statistically, not bit for bit.

#include "md_simulation.h"
#include "pme_rank_split.h"

#include <mpi.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

static_assert(sizeof(Real) == sizeof(double), "restart files store Real arrays as double");

constexpr char kRestartMagic[8] = {'D', 'S', 'M', 'D', 'R', 'S', 'T', '1'};
constexpr char kRestartEndMagic[8] = {'D', 'S', 'M', 'D', 'E', 'N', 'D', '1'};

template <typename T>
void put(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void put_string(std::ofstream& out, const std::string& text) {
    put<int32_t>(out, static_cast<int32_t>(text.size()));
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
}

void put_reals(std::ofstream& out, const std::vector<Real>& values) {
    put<int32_t>(out, static_cast<int32_t>(values.size()));
    for (Real v : values) put<double>(out, static_cast<double>(v));
}

template <typename T>
T get(std::ifstream& in, const std::string& path) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("restart file " + path + " ends early (truncated or not a dsMD restart file)");
    return value;
}

std::string get_string(std::ifstream& in, const std::string& path) {
    const int32_t length = get<int32_t>(in, path);
    if (length < 0 || length > 4096) throw std::runtime_error("restart file " + path + " is damaged (bad string length)");
    std::string text(static_cast<size_t>(length), '\0');
    if (length > 0) in.read(&text[0], length);
    if (!in) throw std::runtime_error("restart file " + path + " ends early");
    return text;
}

std::vector<Real> get_reals(std::ifstream& in, const std::string& path) {
    const int32_t count = get<int32_t>(in, path);
    if (count < 0 || count > 1024) throw std::runtime_error("restart file " + path + " is damaged (bad state length)");
    std::vector<Real> values(static_cast<size_t>(count));
    for (auto& v : values) v = static_cast<Real>(get<double>(in, path));
    return values;
}

void get_array(std::ifstream& in, std::vector<double>& values, const std::string& path) {
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(double)));
    if (!in) throw std::runtime_error("restart file " + path + " ends early (truncated)");
}

void bcast_string(std::string& text, MPI_Comm comm) {
    int length = static_cast<int>(text.size());
    MPI_Bcast(&length, 1, MPI_INT, 0, comm);
    text.resize(static_cast<size_t>(length));
    if (length > 0) MPI_Bcast(&text[0], length, MPI_CHAR, 0, comm);
}

void bcast_reals(std::vector<Real>& values, MPI_Comm comm) {
    int count = static_cast<int>(values.size());
    MPI_Bcast(&count, 1, MPI_INT, 0, comm);
    values.resize(static_cast<size_t>(count));
    if (count > 0) MPI_Bcast(values.data(), count, MPI_DOUBLE, 0, comm);
}

}  // namespace

bool MDSimulation::should_write_restart_step(int step) const {
    if (restart_output_freq_ <= 0) return false;
    return (step % restart_output_freq_ == 0) || (step == start_step_ + total_steps);
}

void MDSimulation::read_restart_file() {
    MPI_Comm comm = PmePpSplit::instance().pp_comm();
    const int rank = system->get_rank();

    // Rank 0 holds the global arrays at this point (load_standard_xyz, before setup_domain_decomposition scatters them).
    std::string error;
    int64_t step = 0;
    double header[4] = {0.0, 0.0, 0.0, 0.0};   // dt, lx, ly, lz
    int periodic[3] = {1, 1, 1};
    std::string thermostat_name, barostat_name;
    std::vector<Real> thermostat_state, barostat_state;

    if (rank == 0) {
        try {
            std::ifstream in(restart_file_, std::ios::binary);
            if (!in.is_open()) throw std::runtime_error("cannot open restart file " + restart_file_);
            char magic[8];
            in.read(magic, 8);
            if (!in || std::memcmp(magic, kRestartMagic, 8) != 0) {
                throw std::runtime_error(restart_file_ + " is not a dsMD restart file (DSMDRST1)");
            }
            step = get<int64_t>(in, restart_file_);
            const int64_t natoms = get<int64_t>(in, restart_file_);
            for (double& h : header) h = get<double>(in, restart_file_);
            for (int& p : periodic) p = get<int32_t>(in, restart_file_);
            thermostat_name = get_string(in, restart_file_);
            thermostat_state = get_reals(in, restart_file_);
            barostat_name = get_string(in, restart_file_);
            barostat_state = get_reals(in, restart_file_);

            ParticleData& particles = system->get_particles();
            const size_t n = particles.num_particles;
            if (natoms != static_cast<int64_t>(n)) {
                throw std::runtime_error("restart file " + restart_file_ + " has " + std::to_string(natoms) +
                                         " atoms, the coordinate file " + coordinate_file + " has " + std::to_string(n));
            }
            if (step < 0 || step > static_cast<int64_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("restart file " + restart_file_ + " has an invalid step " + std::to_string(step));
            }
            if (!(header[1] > 0.0 && header[2] > 0.0 && header[3] > 0.0)) {
                throw std::runtime_error("restart file " + restart_file_ + " has an invalid box");
            }

            std::vector<double> buffer(n);
            std::vector<Real>* targets[6] = {&particles.x, &particles.y, &particles.z,
                                             &particles.vx, &particles.vy, &particles.vz};
            for (std::vector<Real>* target : targets) {
                get_array(in, buffer, restart_file_);
                for (size_t i = 0; i < n; ++i) {
                    const int gid = particles.global_id[i];
                    if (gid < 0 || static_cast<size_t>(gid) >= n) {
                        throw std::runtime_error("invalid global atom id while reading the restart file");
                    }
                    (*target)[i] = static_cast<Real>(buffer[static_cast<size_t>(gid)]);
                }
            }
            char end[8];
            in.read(end, 8);
            if (!in || std::memcmp(end, kRestartEndMagic, 8) != 0) {
                throw std::runtime_error("restart file " + restart_file_ + " is truncated (no end marker)");
            }
        } catch (const std::exception& e) {
            error = e.what();
        }
    }

    bcast_string(error, comm);
    if (!error.empty()) throw std::runtime_error(error);

    MPI_Bcast(&step, 1, MPI_INT64_T, 0, comm);
    MPI_Bcast(header, 4, MPI_DOUBLE, 0, comm);
    MPI_Bcast(periodic, 3, MPI_INT, 0, comm);
    bcast_string(thermostat_name, comm);
    bcast_reals(thermostat_state, comm);
    bcast_string(barostat_name, comm);
    bcast_reals(barostat_state, comm);

    SimulationBox& box = system->get_box();
    const bool periodic_differs = (box.periodic_x != (periodic[0] != 0)) || (box.periodic_y != (periodic[1] != 0)) ||
                                  (box.periodic_z != (periodic[2] != 0));
    const Real yaml_box[3] = {box.lx, box.ly, box.lz};
    box.lx = header[1];
    box.ly = header[2];
    box.lz = header[3];
    box.xlo = 0.0; box.xhi = box.lx;
    box.ylo = 0.0; box.yhi = box.ly;
    box.zlo = 0.0; box.zhi = box.lz;
    box.update_box_length();

    start_step_ = static_cast<int>(step);
    current_step = start_step_;
    restarted_ = true;
    restart_thermostat_name_ = thermostat_name;
    restart_thermostat_state_ = thermostat_state;
    restart_barostat_name_ = barostat_name;
    restart_barostat_state_ = barostat_state;

    // Start-up std::cout is silenced: notes go to stderr.
    if (rank == 0) {
        std::cerr << "[restart] " << restart_file_ << ": step " << start_step_ << ", box " << box.lx << " x " << box.ly
                  << " x " << box.lz << " A";
        if (yaml_box[0] != box.lx || yaml_box[1] != box.ly || yaml_box[2] != box.lz) {
            std::cerr << " (YAML box " << yaml_box[0] << " x " << yaml_box[1] << " x " << yaml_box[2] << " not used)";
        }
        std::cerr << std::endl;
        if (header[0] > 0.0 && header[0] != dt) {
            std::cerr << "[restart] note: the restart file was written with timestep " << header[0]
                      << " fs, this run uses " << dt << " fs" << std::endl;
        }
        if (periodic_differs) {
            std::cerr << "[restart] note: periodic_x/y/z differ from the run that wrote the restart file; "
                      << "the YAML values are used" << std::endl;
        }
    }
}

void MDSimulation::apply_restart_dynamics_state() {
    if (!restarted_) return;
    const bool rank0 = system->get_rank() == 0;

    if (thermostat) {
        const size_t expected = thermostat->restart_state().size();
        if (restart_thermostat_name_ == thermostat_type && restart_thermostat_state_.size() == expected) {
            thermostat->set_restart_state(restart_thermostat_state_);
        } else if (expected > 0 && rank0) {
            std::cerr << "[restart] note: the restart file has thermostat '" << restart_thermostat_name_
                      << "', this run uses '" << thermostat_type << "': the thermostat starts from its initial state"
                      << std::endl;
        }
    }
    if (barostat) {
        const size_t expected = barostat->restart_state().size();
        if (restart_barostat_name_ == barostat_type && restart_barostat_state_.size() == expected) {
            barostat->set_restart_state(restart_barostat_state_);
        } else if (expected > 0 && rank0) {
            std::cerr << "[restart] note: the restart file has barostat '" << restart_barostat_name_
                      << "', this run uses '" << barostat_type << "': the barostat starts from its initial state"
                      << std::endl;
        }
    }
}

void MDSimulation::write_restart_file(int step) {
    const auto start = std::chrono::high_resolution_clock::now();

    // Positions and velocities live on the GPU in a CUDA run; the gather is collective over the PP ranks.
    sync_output_state_to_host_if_needed();
    SerialTrajectoryFrame frame;
    gather_particles_for_serial_output(frame, true, false);

    if (system->get_rank() == 0) {
        const std::vector<Real> thermostat_state = thermostat ? thermostat->restart_state() : std::vector<Real>();
        const std::vector<Real> barostat_state = barostat ? barostat->restart_state() : std::vector<Real>();
        const SimulationBox& box = system->get_box();
        const std::string temporary = restart_output_ + ".tmp";
        bool ok = false;
        {
            std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
            if (out.is_open()) {
                out.write(kRestartMagic, 8);
                put<int64_t>(out, static_cast<int64_t>(step));
                put<int64_t>(out, static_cast<int64_t>(frame.x.size()));
                put<double>(out, static_cast<double>(dt));
                put<double>(out, static_cast<double>(box.lx));
                put<double>(out, static_cast<double>(box.ly));
                put<double>(out, static_cast<double>(box.lz));
                put<int32_t>(out, box.periodic_x ? 1 : 0);
                put<int32_t>(out, box.periodic_y ? 1 : 0);
                put<int32_t>(out, box.periodic_z ? 1 : 0);
                put_string(out, thermostat ? thermostat_type : std::string("none"));
                put_reals(out, thermostat_state);
                put_string(out, barostat ? barostat_type : std::string("none"));
                put_reals(out, barostat_state);
                const std::vector<Real>* arrays[6] = {&frame.x, &frame.y, &frame.z, &frame.vx, &frame.vy, &frame.vz};
                for (const std::vector<Real>* values : arrays) {
                    out.write(reinterpret_cast<const char*>(values->data()),
                              static_cast<std::streamsize>(values->size() * sizeof(double)));
                }
                out.write(kRestartEndMagic, 8);
                out.flush();
                ok = static_cast<bool>(out);
            }
        }
        std::error_code ec;
        if (ok) std::filesystem::rename(temporary, restart_output_, ec);
        if (!ok || ec) {
            // A failed restart write does not stop the run; the previous restart file (if any) is left as it was.
            std::cerr << "[restart] WARNING: could not write " << restart_output_ << " at step " << step
                      << (ec ? " (" + ec.message() + ")" : std::string()) << std::endl;
        } else {
            std::cout << "[restart] wrote " << restart_output_ << " at step " << step << std::endl;
        }
    }

    const double seconds =
        std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
    cumulative_timing.output_time += seconds;
}
