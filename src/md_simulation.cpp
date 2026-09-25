#include "md_simulation.h"
#include "neighbor_list.h"
#include "force_calculator_cuda.h"
#include "constraint_solver.h"
#include "thermostat.h"
#include "barostat.h"
#include "halo_backend.h"
#include "comm_backend.h"
#include "simulation_backend_factory.h"
#include "pme_rank_split.h"
#include "pme_server.h"
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <array>
#include <set>
#include <filesystem>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include "force_libtorch/runtime/mlip_runtime.h"
#include "force_libtorch/mlip_types.h"
using namespace mlip;

namespace {
template <typename F>
double measure_seconds(F&& fn) {
    auto start = std::chrono::high_resolution_clock::now();
    fn();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    return (value != nullptr) && (std::string(value) == "1");
}

class ScopedCoutSilencer {
public:
    ScopedCoutSilencer() : previous_(std::cout.rdbuf(sink_.rdbuf())) {}
    ~ScopedCoutSilencer() { std::cout.rdbuf(previous_); }

    ScopedCoutSilencer(const ScopedCoutSilencer&) = delete;
    ScopedCoutSilencer& operator=(const ScopedCoutSilencer&) = delete;

private:
    std::ostringstream sink_;
    std::streambuf* previous_;
};

class ScopedStreamFormat {
public:
    explicit ScopedStreamFormat(std::ios& stream)
        : stream_(stream), flags_(stream.flags()), precision_(stream.precision()),
          fill_(stream.fill()) {}
    ~ScopedStreamFormat() {
        stream_.flags(flags_);
        stream_.precision(precision_);
        stream_.fill(fill_);
    }

    ScopedStreamFormat(const ScopedStreamFormat&) = delete;
    ScopedStreamFormat& operator=(const ScopedStreamFormat&) = delete;

private:
    std::ios& stream_;
    std::ios::fmtflags flags_;
    std::streamsize precision_;
    char fill_;
};

std::string format_count(size_t value) {
    std::string result = std::to_string(value);
    for (std::ptrdiff_t pos = static_cast<std::ptrdiff_t>(result.size()) - 3;
         pos > 0; pos -= 3) {
        result.insert(static_cast<size_t>(pos), ",");
    }
    return result;
}

std::string format_real(Real value, int precision = 6) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    std::string result = stream.str();
    const size_t decimal = result.find('.');
    if (decimal != std::string::npos) {
        while (!result.empty() && result.back() == '0') result.pop_back();
        if (!result.empty() && result.back() == '.') result.pop_back();
    }
    return result.empty() ? "0" : result;
}

std::string display_name(std::string value) {
    if (value == "velocity_verlet") return "Velocity Verlet";
    if (value == "leap_frog" || value == "leapfrog") return "Leap-frog";
    if (value == "nose_hoover") return "Nose-Hoover";
    if (value == "parrinello_rahman") return "Parrinello-Rahman";
    if (value == "builtin") return "Built-in";
    std::replace(value.begin(), value.end(), '_', ' ');
    if (!value.empty()) {
        value[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(value[0])));
    }
    return value;
}

void dcd_write_record(std::ofstream& f, const char* data, int32_t nbytes) {
    f.write(reinterpret_cast<const char*>(&nbytes), sizeof(int32_t));
    f.write(data, nbytes);
    f.write(reinterpret_cast<const char*>(&nbytes), sizeof(int32_t));
}

std::string trim_copy(const std::string& input) {
    size_t start = input.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }
    size_t end = input.find_last_not_of(" \t\r\n");
    return input.substr(start, end - start + 1);
}

std::set<std::string> parse_force_components(const std::string& value) {
    std::set<std::string> result;
    std::stringstream stream(value);
    std::string component;
    while (std::getline(stream, component, ',')) {
        component = trim_copy(component);
        if (!component.empty()) result.insert(component);
    }
    return result;
}

std::string uppercase_copy(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return s;
}

std::map<std::string, std::string> read_dsmd_input_manifest(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Cannot open dsMD_input manifest: " + filename);
    std::map<std::string, std::string> values;
    std::string line;
    while (std::getline(file, line)) {
        const size_t comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        const size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = trim_copy(line.substr(0, colon));
        std::string value = trim_copy(line.substr(colon + 1));
        if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                                  (value.front() == '\'' && value.back() == '\''))) {
            value = value.substr(1, value.size() - 2);
        }
        if (!key.empty() && !value.empty()) values[key] = value;
    }
    if (values["schema_version"] != "1")
        throw std::runtime_error("dsMD_input manifest schema_version must be 1");
    if (values["format"] != "dsMD_input")
        throw std::runtime_error("dsMD_input manifest format must be dsMD_input");
    return values;
}
}  // namespace


MDSimulation::MDSimulation() : 
    cuda_calc_cached_(nullptr),
    using_cuda_backend_cached_(false),
    mlip_spec_file_(""),
    coordinate_file(""),
    topology_file(""),
    parameter_file(""),
    target_temperature(0.0),
    output_prefix("md_output"),
    trajectory_output_freq(100),
    energy_output_freq(10),
    cutoff_distance(12.0),
    lj_switch_distance(-1.0),
    neighbor_list_update_freq(10),  // Default update frequency: every 10 steps
    verlet_skin(2.0),               // Default Verlet skin distance
    neighbor_rebuild_hi_frac(0.5),
    neighbor_rebuild_lo_frac(0.4),
    neighbor_displacement_check_stride(0),
    neighbor_list_builds(0),        // Initialize counter
    force_calculation_mode("builtin"),  // Default: use builtin force calculations
    force_calculation_type(""),         // Default: no specific ML type
    mlip_model_path(""),
    mlip_model_type(""),
    mlip_model_size("small"),
    mlip_device("cpu"),
    mlip_dtype("float32"),
    integrator_type("verlet"),
    thermostat_type("none"),
    tau_t(0.1),
    barostat_type("none"),
    target_pressure(1.0),
    tau_p(1.0),
    output_freq(10),                     // Default: every 10 steps
    output_info("STEP, T, KE"),          // Default output info
    trajectory_format("xyz"),            // Default: XYZ format
    trajectory_include_velocities(false), // Default: no velocities
    trajectory_include_forces(false),    // Default: no forces
    current_step(0), 
    total_steps(0), 
    dt(0.001), 
    python_force_path("python_force"), 
    python_force_enabled(false),
    cached_potential_energy(0.0),
    neighbor_rebuild_forced(false),
    adaptive_rebuild_armed_(true) {
    system = std::make_unique<MDSystem>();
}

MDSimulation::~MDSimulation() = default;

void MDSimulation::rebuild_output_info_items_cache() {
    output_fields_cache.clear();
    std::vector<std::string> unknown_tokens;

    std::stringstream ss(output_info);
    std::string item;
    while (std::getline(ss, item, ',')) {
        std::string token = uppercase_copy(trim_copy(item));
        if (token.empty()) {
            continue;
        }

        if (token == "STEP") output_fields_cache.push_back(OutputField::STEP);
        else if (token == "T") output_fields_cache.push_back(OutputField::T);
        else if (token == "KE") output_fields_cache.push_back(OutputField::KE);
        else if (token == "PE") output_fields_cache.push_back(OutputField::PE);
        else if (token == "ETOTAL") output_fields_cache.push_back(OutputField::ETOTAL);
        else if (token == "P" || token == "PRESS") output_fields_cache.push_back(OutputField::PRESS);
        else if (token == "MAX_SPEED") output_fields_cache.push_back(OutputField::MAX_SPEED);
        else if (token == "AVG_SPEED") output_fields_cache.push_back(OutputField::AVG_SPEED);
        else if (token == "FMAX") output_fields_cache.push_back(OutputField::FMAX);
        else unknown_tokens.push_back(token);
    }

    if (system && system->get_rank() == 0) {
        for (const std::string& token : unknown_tokens) {
            std::cerr << "Warning: unknown output_info token '" << token
                      << "' will be ignored." << std::endl;
        }

        if (output_fields_cache.empty()) {
            std::cerr << "Warning: output_info has no valid tokens; falling back to STEP,T,KE." << std::endl;
        }
    }
}

void MDSimulation::initialize(const std::string& config_file) {
    double init_time = 0.0;
    {
        ScopedCoutSilencer silence_routine_initialization_messages;
        init_time = measure_seconds([&]() {
            initialize_impl(config_file);
        });
    }
    cumulative_timing.init_time += init_time;
    print_initialization_summary(config_file);
}

void MDSimulation::print_initialization_summary(const std::string& config_file) const {
    if (!system || system->get_rank() != 0) return;
    ScopedStreamFormat restore_cout_format(std::cout);

    constexpr int label_width = 24;
    constexpr const char* rule =
        "==============================================================================";

    auto item = [&](const std::string& label, const auto& value, int indent = 2) {
        std::cout << std::string(static_cast<size_t>(indent), ' ')
                  << std::left << std::setw(label_width - indent) << label
                  << ": " << value << '\n';
    };
    auto section = [](const char* title) {
        std::cout << '\n' << title << '\n';
    };
    auto continuation = [&](const std::string& value) {
        std::cout << std::string(static_cast<size_t>(label_width + 2), ' ')
                  << value << '\n';
    };
    auto interval = [](int frequency) {
        return frequency > 0 ? "Every " + std::to_string(frequency) + " steps"
                             : std::string("Disabled");
    };

    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::string backend = system->get_force_manager().get_backend();
    backend = uppercase_copy(backend);

    std::string force_mode;
    if (enable_builtin_forces_) force_mode = "Built-in";
    if (enable_python_forces_) force_mode += (force_mode.empty() ? "Python" : " + Python");
    if (enable_libtorch_forces_) force_mode += (force_mode.empty() ? "LibTorch" : " + LibTorch");
    if (force_mode.empty()) force_mode = display_name(force_calculation_mode);

    const SimulationBox& box = system->get_box();
    const ForceFieldParams& ff = system->get_ff_params();
    const Topology& topology = system->get_topology();
    const size_t particle_count = topology_file.empty()
        ? system->get_particles().num_particles
        : static_cast<size_t>(topology.get_num_atoms());

    std::ostringstream box_dimensions;
    box_dimensions << std::fixed << std::setprecision(3)
                   << box.lx << " x " << box.ly << " x " << box.lz << " A";

    std::string periodic_axes;
    if (box.periodic_x) periodic_axes += "X";
    if (box.periodic_y) periodic_axes += periodic_axes.empty() ? "Y" : ", Y";
    if (box.periodic_z) periodic_axes += periodic_axes.empty() ? "Z" : ", Z";
    if (periodic_axes.empty()) periodic_axes = "None";

    std::string trajectory_formats;
    for (const std::string& format : trajectory_formats_) {
        if (!trajectory_formats.empty()) trajectory_formats += ", ";
        trajectory_formats += uppercase_copy(format);
    }
    if (trajectory_formats.empty()) trajectory_formats = "Disabled";

    std::vector<std::string> trajectory_filenames;
    for (const std::string& format : trajectory_formats_) {
        if (format == "xyz") trajectory_filenames.push_back(output_prefix + "_<step>.xyz");
        else if (format == "dump") trajectory_filenames.push_back(output_prefix + "_dump.<step>");
        else if (format == "dcd") trajectory_filenames.push_back(output_prefix + ".dcd");
        else if (format == "bp") trajectory_filenames.push_back(output_prefix + ".bp");
        else if (format == "h5md") trajectory_filenames.push_back(output_prefix + ".h5");
    }
    std::string trajectory_files;
    for (const std::string& filename : trajectory_filenames) {
        if (!trajectory_files.empty()) trajectory_files += ", ";
        trajectory_files += filename;
    }
    if (trajectory_files.empty()) trajectory_files = "None";

    std::vector<std::string> console_field_names;
    std::stringstream fields(output_info);
    std::string field;
    while (std::getline(fields, field, ',')) {
        field = uppercase_copy(trim_copy(field));
        std::string label;
        if (field == "STEP") label = "Step";
        else if (field == "T") label = "Temperature";
        else if (field == "KE") label = "Kinetic energy";
        else if (field == "PE") label = "Potential energy";
        else if (field == "ETOTAL") label = "Total energy";
        else if (field == "P" || field == "PRESS") label = "Pressure";
        else if (field == "MAX_SPEED") label = "Maximum speed";
        else if (field == "AVG_SPEED") label = "Average speed";
        else if (field == "FMAX") label = "Maximum force";
        if (!label.empty()) console_field_names.push_back(label);
    }
    std::vector<std::string> console_field_lines;
    for (const std::string& label : console_field_names) {
        const std::string addition = (console_field_lines.empty() || console_field_lines.back().empty())
            ? label : ", " + label;
        if (console_field_lines.empty() ||
            console_field_lines.back().size() + addition.size() > 50) {
            if (!console_field_lines.empty()) console_field_lines.back() += ',';
            console_field_lines.push_back(label);
        } else {
            console_field_lines.back() += addition;
        }
    }

    std::cout << rule << '\n'
              << "  DeepScaleMD - Molecular Dynamics Simulation\n"
              << rule << '\n';

    section("Runtime");
    item("MPI processes", world_size);
    item("Compute backend", backend);
    item("Configuration", config_file);
    item("Force calculation", force_mode);
    item("Nonbonded kernel", display_name(nonbonded_kernel_) + ", " + display_name(force_precision_) + " precision");
    item("Skin distance", format_real(verlet_skin) + " A");

    section("Simulation");
    item("Integrator", display_name(integrator_type));
    item("Timestep", format_real(dt) + " fs");
    item("Total steps", total_steps);
    item("Target temperature", format_real(target_temperature) + " K");
    item("Thermostat", display_name(thermostat_type));
    item("Barostat", display_name(barostat_type));
    item("Periodic boundaries", periodic_axes);

    if (minimizer_type_ != "none" && !minimizer_type_.empty()) {
        section("Energy Minimization");
        item("Method", uppercase_copy(minimizer_type_));
        item("Maximum steps", minimizer_max_steps_);
        item("Force tolerance", format_real(minimizer_force_tolerance_) + " kcal/mol/A");
    }

    section("System");
    item("Particles", format_count(particle_count));
    item("Atom types", ff.num_atom_types);
    item("Box dimensions", box_dimensions.str());
    item("Coordinate file", coordinate_file);
    if (!parameter_file.empty()) item("Parameter file", parameter_file);
    if (!topology_file.empty()) item("Topology file", topology_file);

    if (!topology_file.empty()) {
        std::cout << "\n  Topology\n";
        item("Bonds", format_count(topology.get_bonds().size()), 4);
        item("Angles", format_count(topology.get_angles().size()), 4);
        item("Dihedrals", format_count(topology.get_dihedrals().size()), 4);
        item("Impropers", format_count(topology.get_impropers().size()), 4);
        item("Special pairs", format_count(topology.get_special_pairs().size()), 4);
        item("Constraints", format_count(topology.get_constraints().size()), 4);
        item("Exclusions", format_count(topology.get_exclusion_keys().size()), 4);
    }

    section("Force Field");
    item("Nonbonded potential", "Mie (" + std::to_string(ff.mie_repulsive_exponent) +
                                ", " + std::to_string(ff.mie_attractive_exponent) + ")");
    item("Cutoff distance", format_real(cutoff_distance) + " A");
    if (lj_switch_distance > 0.0) {
        item("LJ switch distance", format_real(lj_switch_distance) + " A");
    }
    item("Neighbor-list update", interval(neighbor_list_update_freq));

    section("Output");
    item("Output prefix", output_prefix);
    item("Console interval", interval(output_freq));
    item("Console fields", console_field_lines.empty() ? "Default" : console_field_lines.front());
    for (size_t i = 1; i < console_field_lines.size(); ++i) {
        continuation(console_field_lines[i]);
    }
    item("Trajectory interval", trajectory_formats_.empty() ? "Disabled" : interval(trajectory_output_freq));
    item("Trajectory format", trajectory_formats);
    item("Trajectory filename", trajectory_files);
    item("Include velocities", trajectory_include_velocities ? "Yes" : "No");
    item("Include forces", trajectory_include_forces ? "Yes" : "No");

    section("Initialization");
    item("Coordinates", "Loaded " + format_count(particle_count) + " particles");
    item("Coordinate shift", "Center of mass -> box center");
    item("Atom types", topology_file.empty() ? "Not assigned" : "Assigned");
    item("Charges", topology_file.empty() ? "Not assigned" : "Assigned");
    item("Initial velocities", format_real(target_temperature) + " K");

    if (force_precision_ == "mixed") {
        std::cout << '\n' << std::string(78, '-') << '\n'
                  << "  WARNING\n"
                  << "  Mixed-precision nonbonded calculations are enabled.\n"
                  << "  Validate wall-clock performance and energy conservation before production use.\n"
                  << std::string(78, '-') << '\n';
    }

    std::cout << "\n  Initialization complete - ready to start simulation.\n"
              << rule << std::endl;
}

void MDSimulation::initialize_impl(const std::string& config_file) {
    system->load_config(config_file);

    parse_simulation_config(config_file);

    // GROMACS-style PP/PME rank split (force_calculation.pme_ranks, default 0
    // = disabled). Must happen right here: pme_ranks_ is known as soon as
    // parse_simulation_config() returns, and nothing rank-consequential has
    // run yet (system->load_config() above is identical on every rank).
    // pme_ranks_ == 0 leaves PmePpSplit inactive, so pp_comm() reports
    // MPI_COMM_WORLD everywhere below/downstream and behavior is unchanged.
    PmePpSplit::instance().initialize(pme_ranks_);
    if (PmePpSplit::instance().is_pme_rank()) {
#ifdef USE_CUDA
        // A PME group of exactly one rank runs GPU-resident (cuFFT, already
        // proven single-GPU code, see PmeServerCuda). A group of more than
        // one rank would need a genuinely multi-GPU distributed FFT
        // (cuFFTMp + NVSHMEM) -- a much larger, higher-risk, and currently
        // unimplemented dependency (see docs/discussion history). Rather
        // than guess at that unfamiliar API with no way to verify it here,
        // a CUDA build's PME group falls back to the CPU FFTW-MPI engine
        // (PmeServer) in that case: the PME group is a small set of
        // dedicated ranks doing one global FFT, not the GPU-resident
        // real-space work PP ranks do, so running it on CPU is not a
        // performance regression on the part of the pipeline that matters.
        if (PmePpSplit::instance().pme_size() == 1) {
            pme_server_cuda_ = std::make_unique<PmeServerCuda>(PmePpSplit::instance());
            pme_server_cuda_->run(total_steps, cutoff_distance);
        } else {
            pme_server_ = std::make_unique<PmeServer>(PmePpSplit::instance());
            pme_server_->run(total_steps, cutoff_distance);
        }
#else
        pme_server_ = std::make_unique<PmeServer>(PmePpSplit::instance());
        pme_server_->run(total_steps, cutoff_distance);
#endif
        return;  // never touches system->, force_calc, thermostat, barostat, I/O
    }
    if (PmePpSplit::instance().active()) {
        system->set_pp_communicator(PmePpSplit::instance().pp_comm());
        if (system->get_rank() == 0) {
            std::cout << "[PME split] world_size=" << (PmePpSplit::instance().pp_size()
                                                        + PmePpSplit::instance().pme_size())
                      << " ranks: " << PmePpSplit::instance().pp_size() << " PP + "
                      << PmePpSplit::instance().pme_size() << " PME" << std::endl;
        }
    }

    system->load_coordinates(coordinate_file);
    
    if (!topology_file.empty()) {
        system->load_topology(topology_file, parameter_file);
        system->setup_force_field_params();
        
        if (system->get_rank() == 0) {
            std::cout << "Topology and force field parameters loaded" << std::endl;
        }
    }
    
    system->initialize_velocities(target_temperature);
    
    system->setup_domain_decomposition();

    force_calc = create_force_calculator_for_build(system->get_rank());
    force_calc->set_nonbonded_kernel(nonbonded_kernel_);
    if (system->get_rank() == 0) {
        std::cout << "[Nonbonded kernel] loaded force_calculation.nonbonded_kernel = "
                  << nonbonded_kernel_ << std::endl;
        if (nonbonded_kernel_ == "graph") {
            std::cout << "[Nonbonded kernel] graph dispatch enabled: outer CSR(rc+skin)"
                      << " -> displacement-triggered active directed CSR(rc+skin/2)"
                      << " -> exact rc force check"
                      << " (legacy tile graph disabled)"
                      << std::endl;
        } else {
            std::cout << "[Nonbonded kernel] tiled dispatch enabled (legacy path)"
                      << std::endl;
        }
    }
    cuda_calc_cached_ = dynamic_cast<ForceCalculatorCUDA*>(force_calc.get());
    using_cuda_backend_cached_ = (cuda_calc_cached_ != nullptr);

    halo_backend_ = create_halo_backend(force_calculation_mode, force_calc.get());
    comm_backend_ = create_comm_backend();
#ifdef USE_CUDA
    if (cuda_calc_cached_) {
        cuda_calc_cached_->set_builtin_forces_enabled(enable_builtin_forces_);
        cuda_calc_cached_->set_mixed_precision(force_precision_ == "mixed");
        if (force_precision_ == "mixed" && system->get_rank() == 0) {
            std::cout << "[Force precision] mixed precision enabled: LJ/Ewald "
                      << "real-space pair math runs in float, forces/energy/"
                      << "virial still accumulate in double" << std::endl;
        }
        cuda_calc_cached_->set_pme_cluster_spread_enabled(pme_cluster_spread_);
        if (pme_cluster_spread_ && system->get_rank() == 0) {
            std::cout << "[PME] thread-block-cluster spread kernel requested -- "
                      << "unverified on real Hopper hardware; only takes "
                      << "effect if this rank's GPU is compute capability "
                      << ">= 9.0, otherwise the usual kernel runs."
                      << std::endl;
        }
    }
#endif
    if (!cuda_calc_cached_ && force_precision_ == "mixed") {
        force_calc->set_mixed_precision(true);
    }

    if (enable_python_forces_) {
        python_interface = std::make_unique<PythonForceInterface>();
        python_interface->initialize_python(python_force_path);
        if (system->get_rank() == 0) {
            std::cout << "Python force calculation interface initialized" << std::endl;
        }

        {
            if (is_using_cuda_backend() && mlip_device != "cpu") {
                mlip_device = "cuda";
            }
            if (mlip_dtype.empty()) mlip_dtype = "float32";
            const std::string effective_model_type =
                !mlip_model_type.empty() ? mlip_model_type : force_calculation_type;
            if (system->get_rank() == 0) {
                if (effective_model_type.empty()) {
                    std::cout << "Using Python force provider without MLIP model initialization" << std::endl;
                } else {
                    std::cout << "Initializing Python MLIP (type=" << effective_model_type << ")..." << std::endl;
                    std::cout << "  model_path: " << mlip_model_path << std::endl;
                    std::cout << "  model_type: " << effective_model_type << std::endl;
                    std::cout << "  model_size: " << mlip_model_size << std::endl;
                    std::cout << "  device:     " << mlip_device << std::endl;
                    std::cout << "  dtype:      " << mlip_dtype << std::endl;
                }
            }
            if (!effective_model_type.empty()) {
                python_interface->initialize_ml_model(
                    mlip_model_path, effective_model_type, mlip_model_size,
                    mlip_device, mlip_dtype, mlip_extra_kwargs);
            }
        }
    }

    if (enable_libtorch_forces_) {
#ifndef USE_CUDA
        throw std::runtime_error(
            "[LibTorch] libtorch force mode requires a CUDA build (USE_CUDA). "
            "Current build does not enable CUDA.");
#else
        if (!is_using_cuda_backend()) {
            throw std::runtime_error(
                "[LibTorch] libtorch force mode requires an active CUDA backend. "
                "Current run is not using ForceCalculatorCUDA.");
        }

        if (mlip_spec_file_.empty()) {
            throw std::runtime_error(
                "[LibTorch] force_calculation.mode = libtorch requires "
                "'mlip_spec_file' key in the MD YAML pointing to a model spec file.");
        }

        if (system->get_rank() == 0) {
            std::cout << "[MlipRuntime] Initializing C++ MLIP runtime...\n"
                      << "  spec: " << mlip_spec_file_ << "\n";
        }

        MlipExecutionContext ctx;
        ctx.mpi_rank  = system->get_rank();
        ctx.mpi_size  = system->get_size();
        ctx.device_id = 0;   // TODO: multi-GPU rank mapping

        mlip_runtime_ = std::make_unique<mlip::MlipRuntime>();

        {
            auto* fc_cuda = cuda_calc_cached_;
            if (fc_cuda) fc_cuda->pre_register_mlip_sources();
        }

        mlip_runtime_->initialize(mlip_spec_file_, ctx);

        if (system->get_rank() == 0) {
            std::cout << "[MlipRuntime] ✅ Runtime ready. cutoff="
                      << mlip_runtime_->cutoff() << " Å\n";
        }

        auto* fc_cuda = cuda_calc_cached_;
        if (fc_cuda) {
            fc_cuda->set_mlip_runtime(static_cast<void*>(mlip_runtime_.get()));
        }
#endif
    }
    
    Real cutoff = cutoff_distance;
    if (system->get_rank() == 0) {
        std::cout << "Using cpu_spatial_verlet neighbor list with skin distance: "
                  << verlet_skin << std::endl;
    }

    neighbor_list_ = create_neighbor_list(NeighborListBackend::CPU_SPATIAL_VERLET, cutoff, verlet_skin);
    neighbor_list_->set_nonbonded_kernel(nonbonded_kernel_);
    neighbor_list_->set_mlip_edge_list_needed(enable_libtorch_forces_);
    system->set_neighbor_list(neighbor_list_.get());

    initialize_integrator();
    initialize_thermostat();
    initialize_barostat();

    if (!system->get_topology().get_constraints().empty()) {
        bool is_velocity_verlet = dynamic_cast<VelocityVerletIntegrator*>(integrator.get()) != nullptr;
#ifdef USE_CUDA
        is_velocity_verlet = is_velocity_verlet ||
            dynamic_cast<VelocityVerletIntegratorCUDA*>(integrator.get()) != nullptr;
#endif
        if (!is_velocity_verlet) {
            throw std::runtime_error(
                "[Constraints] SHAKE/RATTLE are only implemented for the velocity_verlet integrator.");
        }
        constraints_enabled_ = true;
        if (!cuda_calc_cached_) {
            constraint_solver_ = std::make_unique<ConstraintSolver>();
        }
        if (system->get_rank() == 0) {
            std::cout << "[Constraints] SHAKE/RATTLE enabled ("
                      << (cuda_calc_cached_ ? "CUDA" : "CPU") << ") for "
                      << system->get_topology().get_constraints().size()
                      << " constraint(s)" << std::endl;
            if (system->get_size() > 1) {
                std::cout << "[Constraints] MPI mode: cross-rank constraints "
                          << "(one endpoint on a ghost slot) are skipped by "
                          << "the CPU solver to avoid inconsistent owner/ghost "
                          << "state updates." << std::endl;
            }
        }
    }

    initialize_trajectory_writers();
}

bool MDSimulation::is_using_cuda_backend() const {
    return using_cuda_backend_cached_;
}

void MDSimulation::single_step() {
    auto step_start = std::chrono::high_resolution_clock::now();
    
    ForceCalculatorCUDA* cuda_calc = cuda_calc_cached_;

    double integrate1_time = 0.0;
    handle_integrate1_phase(cuda_calc, integrate1_time);
    
    double comm_time = 0.0;   // Track halo communication time
    double migrate_time = 0.0; // Track migration check/exchange time
    
    double nlist_build_time = 0.0;
    handle_neighbor_phase(nlist_build_time, comm_time, migrate_time);
    
    double force_time = 0.0;
    handle_force_phase(cuda_calc, force_time);

    double integrate2_time = 0.0;
    handle_integrate2_phase(cuda_calc, integrate2_time);
    
    double thermostat_time = 0.0;
    double ke_time = 0.0;
    double barostat_time = 0.0;
    double modify_time = 0.0;
    handle_modify_phase(
        cuda_calc,
        thermostat_time, ke_time,
        barostat_time, modify_time);
    
    auto step_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(step_end - step_start).count();

    accumulate_step_timing(
        integrate1_time, nlist_build_time, comm_time,
        migrate_time,
        force_time, integrate2_time,
        thermostat_time, ke_time,
        barostat_time, modify_time,
        total_time);
        
    current_step++;
}

void MDSimulation::prepare_first_step() {
    ForceCalculatorCUDA* cuda_calc = cuda_calc_cached_;

    if (integrator_type == "leap_frog" || integrator_type == "leapfrog") {
        integrator->initialize_velocities(system->get_particles(), dt, system->get_local_particles());
    }

    if (cuda_calc) {
#ifdef USE_CUDA
        cuda_calc->initial_upload_to_device(*system);
#endif
    }

    double neighbor_time = 0.0;
    double comm_time = 0.0;
    double migrate_time = 0.0;
    handle_neighbor_phase(neighbor_time, comm_time, migrate_time);

    double force_time = 0.0;
    handle_force_phase(cuda_calc, force_time);
    
    if (force_calc) {
        force_calc->calculate_kinetic_energy(*system);
    }

}

void MDSimulation::handle_integrate1_phase(ForceCalculatorCUDA* cuda_calc, double& integrate1_time) {
    integrate1_time = measure_seconds([&]() {
        if (cuda_calc) {
#ifdef USE_CUDA
            if (constraints_enabled_) {
                cuda_calc->snapshot_constraint_positions(*system);
            }
            cuda_calc->integrate_on_device(*system, dt, integrator_type);
            if (constraints_enabled_) {
                if (!cuda_calc->apply_shake_on_device(*system, dt)) {
                    throw std::runtime_error(
                        "[Constraints] CUDA SHAKE did not converge; reduce the timestep or check the constraint geometry.");
                }
            }
            return;
#else
#endif
        }
        if (constraints_enabled_) {
            ParticleData& particles = system->get_particles();
            const size_t n = particles.x.size();
            constraint_pre_step_x_.resize(n);
            constraint_pre_step_y_.resize(n);
            constraint_pre_step_z_.resize(n);
            std::copy(particles.x.begin(), particles.x.begin() + n, constraint_pre_step_x_.begin());
            std::copy(particles.y.begin(), particles.y.begin() + n, constraint_pre_step_y_.begin());
            std::copy(particles.z.begin(), particles.z.begin() + n, constraint_pre_step_z_.begin());
        }
        integrator->integrate(system->get_particles(), system->get_box(), dt, system->get_local_particles());
        if (constraints_enabled_) {
            if (!constraint_solver_->apply_shake(*system, dt,
                    constraint_pre_step_x_, constraint_pre_step_y_, constraint_pre_step_z_)) {
                throw std::runtime_error(
                    "[Constraints] CPU SHAKE did not converge; reduce the timestep or check the constraint geometry.");
            }
        }
    });
}

void MDSimulation::handle_neighbor_phase(double& nlist_build_time, double& comm_time, double& migrate_time) {
    const bool periodic_rebuild_due = (neighbor_list_builds == 0) ||
        (current_step > 0 && current_step % neighbor_list_update_freq == 0);
    bool should_rebuild = periodic_rebuild_due;
    bool reason_periodic = periodic_rebuild_due;
    bool reason_adaptive = false;
    bool reason_forced = false;
    bool reason_mpi_sync = false;
    migrate_time = 0.0;

    if (periodic_rebuild_due) {
        adaptive_rebuild_armed_ = true;
    }

    if (!should_rebuild && neighbor_list_builds > 0) {
        // Ghost membership/order is now only refreshed at an actual neighbor
        // rebuild (dsDecompX's halo plan is reused between rebuilds; see
        // dsDecompX/src/halo.cpp Decomposition::refresh_halo_buffers). Under
        // MPI that means a stride-gated skip of this check is no longer just
        // a Verlet-skin approximation -- it is a real window where an atom
        // can drift into range of a neighbor that was never added as a
        // ghost, silently dropping that pair's force. This scan is a cheap
        // local O(local_N) pass, so run it every step when distributed.
        const int displacement_check_stride =
            comm_backend_->is_distributed()
                ? 1
                : (neighbor_displacement_check_stride > 0)
                    ? neighbor_displacement_check_stride
                    : std::max(1, neighbor_list_update_freq / 4);
        const bool run_displacement_check =
            (current_step > 0) && (current_step % displacement_check_stride == 0);

        if (!run_displacement_check) {
            should_rebuild = false;
        } else {
        const Real skin = (verlet_skin > 0.0) ? verlet_skin : 2.0;
        const Real hi_frac = neighbor_rebuild_hi_frac;
        const Real lo_frac = neighbor_rebuild_lo_frac;
        const Real disp_hi_sq = (skin * hi_frac) * (skin * hi_frac);
        const Real disp_lo_sq = (skin * lo_frac) * (skin * lo_frac);

#ifdef USE_CUDA
        const bool using_cuda_backend = (cuda_calc_cached_ != nullptr);
        if (using_cuda_backend) {
            const Real max_disp_sq = cuda_calc_cached_->compute_max_displacement_on_device(
                system->get_local_particles(), system->get_box());
            if (max_disp_sq <= disp_lo_sq) {
                adaptive_rebuild_armed_ = true;
            }
            should_rebuild = adaptive_rebuild_armed_ && (max_disp_sq > disp_hi_sq);
            if (should_rebuild) {
                adaptive_rebuild_armed_ = false;
                reason_adaptive = true;
            }
        } else
#endif
        if (!should_rebuild) {
            Real local_max_disp_sq = neighbor_list_->compute_local_max_displacement_sq(
                system->get_particles(), system->get_box(), system->get_local_particles());
            if (local_max_disp_sq < 0.0) {
                local_max_disp_sq = std::numeric_limits<Real>::max();
            }
            Real max_disp_sq = local_max_disp_sq;
            int mpi_init = 0;
            MPI_Initialized(&mpi_init);
            if (mpi_init) {
                MPI_Allreduce(&local_max_disp_sq, &max_disp_sq, 1, MPI_DOUBLE, MPI_MAX, PmePpSplit::instance().pp_comm());
            }
            if (max_disp_sq <= disp_lo_sq) {
                adaptive_rebuild_armed_ = true;
            }
            should_rebuild = adaptive_rebuild_armed_ && (max_disp_sq > disp_hi_sq);
            if (should_rebuild) {
                adaptive_rebuild_armed_ = false;
                reason_adaptive = true;
            }
        }
        }
    }

    if (neighbor_rebuild_forced) {
        should_rebuild = true;
        reason_forced = true;
        adaptive_rebuild_armed_ = true;
        neighbor_rebuild_forced = false;
    }

    const bool local_rebuild_decision = should_rebuild;
    comm_backend_->unify_rebuild_decision(should_rebuild);
    if (!local_rebuild_decision && should_rebuild) {
        reason_mpi_sync = true;
    }
    const bool global_rebuild_epoch = should_rebuild;

    ForceCalculatorCUDA* cuda_calc = cuda_calc_cached_;
    bool gpu_single_rank = comm_backend_->is_gpu_single_rank(cuda_calc != nullptr);
    if (gpu_single_rank) {
        comm_time = 0.0;
    } else {
        auto comm_start = std::chrono::high_resolution_clock::now();

        if (should_rebuild) {
            auto migrate_start = std::chrono::high_resolution_clock::now();
            comm_backend_->migrate_particles_if_needed(
                *system, true, force_calc.get());
            auto migrate_end = std::chrono::high_resolution_clock::now();
            migrate_time += std::chrono::duration<double>(
                migrate_end - migrate_start).count();
        }

        bool gpu_direct_halo_done = false;
        if (halo_backend_) {
            gpu_direct_halo_done = halo_backend_->try_gpu_direct_halo(
                *system, force_calc.get(), should_rebuild);
        }

        if (cuda_calc && !gpu_direct_halo_done) {
            throw std::runtime_error(
                "CUDA execution requires the dsDecompX GPU-resident halo backend; "
                "host ParticleData fallback is disabled");
        }

        if (!gpu_direct_halo_done) {
            system->sync_halo_particles(should_rebuild);
        }

#ifdef USE_CUDA
        if (cuda_calc) {
            bool late_rebuild = !global_rebuild_epoch && should_rebuild;
            comm_backend_->unify_rebuild_decision(late_rebuild);
            if (late_rebuild) {
                reason_mpi_sync = true;

                auto migrate_start = std::chrono::high_resolution_clock::now();
                comm_backend_->migrate_particles_if_needed(
                    *system, true, force_calc.get());
                auto migrate_end = std::chrono::high_resolution_clock::now();
                migrate_time += std::chrono::duration<double>(
                    migrate_end - migrate_start).count();

                should_rebuild = true;
                gpu_direct_halo_done = halo_backend_->try_gpu_direct_halo(
                    *system, force_calc.get(), should_rebuild);
            }
        }
#endif

        if (halo_backend_) {
            halo_backend_->on_post_halo(
                *system, force_calc.get(), should_rebuild, gpu_direct_halo_done);
        }

#ifdef USE_CUDA
        if (cuda_calc) {
            const auto& cfg = system->get_force_manager().get_config();
            if (cfg.enable_bonds || cfg.enable_angles || cfg.enable_dihedrals) {
                cuda_calc->refresh_bonded_topology_on_device(*system);
            }
        }
#endif

        auto comm_end = std::chrono::high_resolution_clock::now();
        const double distributed_phase_time =
            std::chrono::duration<double>(comm_end - comm_start).count();
        comm_time = std::max(0.0, distributed_phase_time - migrate_time);
    }

    nlist_build_time = measure_seconds([&]() {
        if (should_rebuild) {
            if (reason_periodic) ++rebuild_reason_periodic_;
            if (reason_adaptive) ++rebuild_reason_adaptive_;
            if (reason_forced) ++rebuild_reason_forced_;
            if (reason_mpi_sync) ++rebuild_reason_mpi_sync_;
            if (!reason_periodic && !reason_adaptive && !reason_forced && !reason_mpi_sync) {
                ++rebuild_reason_other_;
            }
        }
        handle_neighbor_rebuild_if_needed(should_rebuild);
        if (nonbonded_kernel_ == "graph" && !cuda_calc_cached_ && neighbor_list_builds > 0) {
            const bool prune_due = neighbor_list_->needs_active_prune(
                system->get_particles(), system->get_box(),
                system->get_local_particles());
            if (prune_due) {
                neighbor_list_->prune_active(system->get_particles(), system->get_box(),
                                             system->get_local_particles());
                ++active_graph_prunes_;
            } else {
                ++active_graph_reuses_;
            }
        }
    });
}

void MDSimulation::compute_forces_with_mode(bool download_positions_for_ml) {
    ForceCalculatorCUDA* cuda_calc = cuda_calc_cached_;

    if (force_calc && (enable_builtin_forces_ || enable_libtorch_forces_ || cuda_calc)) {
        force_calc->compute_forces(*system);
    } else {
        zero_forces();
    }

    Real python_potential_energy = 0.0;

    if (enable_python_forces_ && python_interface) {
        SimulationBox& box = system->get_box();

        if (download_positions_for_ml && cuda_calc) {
#ifdef USE_CUDA
            cuda_calc->download_positions_for_ml(
                system->get_particles(), system->get_num_particles());
            cuda_calc->download_forces_for_ml(
                system->get_particles(), system->get_num_particles());
#endif
        }

        python_interface->compute_force(
            system->get_particles(), system->get_ff_params(),
            system->get_neighbor_list(),
            box.lx, box.ly, box.lz,
            box.periodic_x, box.periodic_y, box.periodic_z);
        python_potential_energy = static_cast<Real>(python_interface->get_last_energy());

        if (cuda_calc) {
#ifdef USE_CUDA
            cuda_calc->upload_forces_from_ml(
                system->get_particles(), system->get_num_particles());
#endif
        }
    }

    cached_potential_energy =
        (force_calc ? force_calc->get_potential_energy() : 0.0) + python_potential_energy;
}

void MDSimulation::reverse_halo_for_bonded_if_needed() {
    if (!comm_backend_->is_distributed()) {
        return;
    }

    const auto& cfg = system->get_force_manager().get_config();
    const bool has_bonded = cfg.enable_bonds || cfg.enable_angles || cfg.enable_dihedrals;
    const bool has_special_pairs =
        (cfg.enable_lennard_jones || cfg.enable_coulomb) &&
        !system->get_local_special_pairs().empty();

    if (!has_bonded && !has_special_pairs) {
        return;
    }

#ifdef USE_CUDA
    ForceCalculatorCUDA* cuda_calc = cuda_calc_cached_;
    if (cuda_calc) {
        system->reverse_halo_on_device(*cuda_calc);
        return;
    }
#endif
    system->perform_reverse_halo_exchange();
}

void MDSimulation::handle_neighbor_rebuild_if_needed(bool should_rebuild) {
    if (!should_rebuild) {
        return;
    }

    ForceCalculatorCUDA* cuda_calc = cuda_calc_cached_;
    if (cuda_calc) {
#ifdef USE_CUDA
        cuda_calc->build_neighbor_list_on_device(*system);
#else
        neighbor_list_->build(system->get_particles(), system->get_box(), system->get_local_particles());
#endif
    } else {
        neighbor_list_->build(system->get_particles(), system->get_box(), system->get_local_particles());
    }

    neighbor_list_builds++;

}

void MDSimulation::handle_force_phase(ForceCalculatorCUDA* cuda_calc, double& force_time) {
    force_time = measure_seconds([&]() {
        if (!cuda_calc) {
            zero_forces();
        }

        compute_forces_with_mode(true);
        reverse_halo_for_bonded_if_needed();

        if (!cuda_calc && comm_backend_->is_distributed() &&
            env_flag_enabled("MD_CHECK_FORCE_BALANCE")) {
            const ParticleData& p = system->get_particles();
            const size_t n_local = system->get_local_particles();
            Real local_sum[3] = {0.0, 0.0, 0.0};
            for (size_t i = 0; i < n_local; ++i) {
                local_sum[0] += p.fx[i];
                local_sum[1] += p.fy[i];
                local_sum[2] += p.fz[i];
            }
            Real global_sum[3] = {0.0, 0.0, 0.0};
            MPI_Reduce(local_sum, global_sum, 3, MPI_DOUBLE, MPI_SUM, 0, PmePpSplit::instance().pp_comm());
            int mpi_rank = 0;
            MPI_Comm_rank(PmePpSplit::instance().pp_comm(), &mpi_rank);
            if (mpi_rank == 0) {
                const Real net = std::sqrt(global_sum[0] * global_sum[0] +
                                           global_sum[1] * global_sum[1] +
                                           global_sum[2] * global_sum[2]);
                if (net > 1e-3) {
                    std::cerr << "[MD_CHECK_FORCE_BALANCE] step " << current_step
                              << ": |sum(F)| = " << net
                              << " (expected ~0) -- asymmetric cross-rank force"
                              << std::endl;
                }
            }
        }
    });
}

void MDSimulation::handle_integrate2_phase(ForceCalculatorCUDA* cuda_calc, double& integrate2_time) {
    integrate2_time = measure_seconds([&]() {
        if (cuda_calc) {
#ifdef USE_CUDA
            cuda_calc->complete_velocity_update_on_device(*system, dt, integrator_type);
            if (constraints_enabled_) {
                if (!cuda_calc->apply_rattle_on_device(*system)) {
                    throw std::runtime_error(
                        "[Constraints] CUDA RATTLE did not converge; reduce the timestep or check the constraint geometry.");
                }
            }
            return;
#else
#endif
        }
        integrator->complete_velocity_update(system->get_particles(), dt, system->get_local_particles());
        if (constraints_enabled_) {
            if (!constraint_solver_->apply_rattle(*system)) {
                throw std::runtime_error(
                    "[Constraints] CPU RATTLE did not converge; reduce the timestep or check the constraint geometry.");
            }
        }
    });
}

void MDSimulation::accumulate_step_timing(double integrate1_time, double nlist_build_time, double comm_time,
                                          double migrate_time,
                                          double force_time, double integrate2_time,
                                          double thermostat_time, double ke_time,
                                          double barostat_time, double modify_time,
                                          double total_time) {
    last_timing.nlist_build_ms = nlist_build_time * 1000.0;
    last_timing.comm_ms = comm_time * 1000.0;
    last_timing.migrate_ms = migrate_time * 1000.0;
    last_timing.force_ms = force_time * 1000.0;
    last_timing.integrate_ms = (integrate1_time + integrate2_time) * 1000.0;
    last_timing.thermostat_ms = thermostat_time * 1000.0;
    last_timing.ke_ms = ke_time * 1000.0;
    last_timing.barostat_ms = barostat_time * 1000.0;
    last_timing.total_ms = total_time * 1000.0;
    last_timing.force_components_ms.clear();

    cumulative_timing.neighbor_time += nlist_build_time;
    cumulative_timing.force_time += force_time;
    cumulative_timing.integration_time += (integrate1_time + integrate2_time);
    cumulative_timing.thermostat_time += thermostat_time;
    cumulative_timing.ke_time += ke_time;
    cumulative_timing.barostat_time += barostat_time;
    cumulative_timing.comm_time += comm_time;
    cumulative_timing.migration_time += migrate_time;
    cumulative_timing.total_time += total_time;
    cumulative_timing.step_count++;

    if (timing_trace_force_ && force_calc) {
        for (const auto& component : force_calc->get_last_force_component_timings()) {
            last_timing.force_components_ms.emplace_back(component.name, component.ms);

            auto it = std::find_if(cumulative_timing.force_component_time.begin(),
                                    cumulative_timing.force_component_time.end(),
                                    [&](const auto& entry) { return entry.first == component.name; });
            if (it != cumulative_timing.force_component_time.end()) {
                it->second += component.ms / 1000.0;
            } else {
                cumulative_timing.force_component_time.emplace_back(component.name, component.ms / 1000.0);
            }
        }
    }

    (void)modify_time;
}

void MDSimulation::handle_thermostat_phase(ForceCalculatorCUDA* cuda_calc,
                                           double& thermostat_time) {
    thermostat_time = measure_seconds([&]() {
        if (thermostat) {
            if (cuda_calc) {
#ifdef USE_CUDA
                thermostat->apply_device(cuda_calc->get_device_vx(), cuda_calc->get_device_vy(),
                                         cuda_calc->get_device_vz(), cuda_calc->get_device_mass(),
                                         system->get_local_particles(), dt, 0);
#else
                thermostat->apply(system->get_particles(), dt, system->get_local_particles());
#endif
            } else {
                thermostat->apply(system->get_particles(), dt, system->get_local_particles());
            }
        }
    });
}

void MDSimulation::handle_ke_output_phase(double& ke_time) {
    ke_time = measure_seconds([&]() {
        int next_step = current_step + 1;
        bool ke_needed = needs_ke_update_for_step(next_step);
        if (force_calc && ke_needed) {
            force_calc->calculate_kinetic_energy(*system);
        }
    });
}

void MDSimulation::handle_barostat_phase(ForceCalculatorCUDA* cuda_calc,
                                         double& barostat_time) {
    barostat_time = measure_seconds([&]() {
        if (barostat) {
            if (force_calc) {
                // calculate_kinetic_energy() does its own local-only reduction
                // over the ranks's owned particles followed by an MPI_Allreduce
                // internally, so every rank ends up with the same global value
                // (same pattern as the virial reduction below via comm_backend_).
                force_calc->calculate_kinetic_energy(*system);
                Real global_kinetic_energy = force_calc->get_kinetic_energy() * mvv2e;
                barostat->set_kinetic_energy(global_kinetic_energy);

                Real local_virial = force_calc->get_virial();
                Real global_virial = comm_backend_->sum_reduce(local_virial);
                barostat->set_virial(global_virial);
            }

            Real old_lx = system->get_box().lx;
            Real old_ly = system->get_box().ly;
            Real old_lz = system->get_box().lz;

            if (cuda_calc) {
#ifdef USE_CUDA
                barostat->apply_device(cuda_calc->get_device_x(), cuda_calc->get_device_y(),
                                       cuda_calc->get_device_z(),
                                       cuda_calc->get_device_vx(), cuda_calc->get_device_vy(),
                                       cuda_calc->get_device_vz(), cuda_calc->get_device_mass(),
                                       system->get_box(), dt, system->get_local_particles(), 0);
#else
                barostat->apply(system->get_particles(), system->get_box(), dt, system->get_local_particles());
#endif
            } else {
                barostat->apply(system->get_particles(), system->get_box(), dt, system->get_local_particles());
            }

            Real new_lx = system->get_box().lx;
            Real new_ly = system->get_box().ly;
            Real new_lz = system->get_box().lz;
            bool box_changed = has_significant_box_change(
                old_lx, old_ly, old_lz,
                new_lx, new_ly, new_lz);

            if (box_changed) {
                handle_box_change_rebuild();
            }
        }
    });
}

bool MDSimulation::has_significant_box_change(Real old_lx, Real old_ly, Real old_lz,
                                              Real new_lx, Real new_ly, Real new_lz) const {
    // Only force the expensive box-change path (full neighbor rebuild, plus a
    // domain resync + halo exchange under MPI) when the barostat's scaling
    // could plausibly move a boundary atom past the Verlet skin margin that
    // the regular periodic/adaptive rebuild check already covers. Gating on
    // floating-point-noise-level changes (the previous 16*epsilon threshold)
    // meant nearly every NPT step's sub-percent scale factor tripped this,
    // forcing a full rebuild (and MPI domain resync) every single step.
    const Real skin = (verlet_skin > 0.0) ? verlet_skin : 2.0;
    const Real budget = neighbor_rebuild_hi_frac * skin;
    auto displaced_enough = [&](Real old_l, Real new_l) {
        // Isotropic scaling moves an atom at the box edge by ~|Δl|/2.
        return std::abs(new_l - old_l) * 0.5 > budget;
    };
    return displaced_enough(old_lx, new_lx)
        || displaced_enough(old_ly, new_ly)
        || displaced_enough(old_lz, new_lz);
}

void MDSimulation::handle_modify_phase(ForceCalculatorCUDA* cuda_calc,
                                       double& thermostat_time, double& ke_time,
                                       double& barostat_time, double& modify_time) {
    modify_time = measure_seconds([&]() {
        handle_thermostat_phase(cuda_calc, thermostat_time);
        handle_ke_output_phase(ke_time);
        handle_barostat_phase(cuda_calc, barostat_time);
    });
}

void MDSimulation::handle_box_change_rebuild() {
    ForceCalculatorCUDA* cuda_calc = cuda_calc_cached_;

    neighbor_rebuild_forced = true;
    if (comm_backend_->is_distributed()) {
        system->sync_decomposition_domain();
        system->invalidate_halo_cache();
    }

    if (comm_backend_->is_distributed()) {
#ifdef USE_CUDA
        if (cuda_calc) {
            system->exchange_halo_on_device(*cuda_calc, true);
        } else
#endif
        {
            system->perform_halo_exchange(true);
        }
    }

    if (cuda_calc) {
#ifdef USE_CUDA
        cuda_calc->build_neighbor_list_on_device(*system);
#else
        neighbor_list_->build(system->get_particles(), system->get_box(), system->get_local_particles());
#endif
    } else {
        neighbor_list_->build(system->get_particles(), system->get_box(), system->get_local_particles());
    }

    ++neighbor_list_builds;
    ++rebuild_reason_box_change_;
    neighbor_rebuild_forced = false;

    if (!cuda_calc) {
        zero_forces();
    }
    compute_forces_with_mode(true);
}

void MDSimulation::zero_forces() {
    ParticleData& particles = system->get_particles();
    size_t local_count = system->get_local_particles();

    for (size_t i = 0; i < local_count; i++) {
        particles.fx[i] = 0.0;
        particles.fy[i] = 0.0;
        particles.fz[i] = 0.0;
    }
}

void MDSimulation::zero_velocities() {
    ParticleData& particles = system->get_particles();
    size_t n = particles.vx.size();

    for (size_t i = 0; i < n; i++) {
        particles.vx[i] = 0.0;
        particles.vy[i] = 0.0;
        particles.vz[i] = 0.0;
    }
}

void MDSimulation::compute_fire_reduction_cpu(Real& power, Real& f_norm_sq,
                                              Real& v_norm_sq, Real& f_max_sq) const {
    const ParticleData& particles = system->get_particles();
    const size_t n = system->get_local_particles();

    power = 0.0; f_norm_sq = 0.0; v_norm_sq = 0.0; f_max_sq = 0.0;
    for (size_t i = 0; i < n; i++) {
        const Real fx = particles.fx[i], fy = particles.fy[i], fz = particles.fz[i];
        const Real vx = particles.vx[i], vy = particles.vy[i], vz = particles.vz[i];
        power += fx * vx + fy * vy + fz * vz;
        const Real f2 = fx * fx + fy * fy + fz * fz;
        f_norm_sq += f2;
        v_norm_sq += vx * vx + vy * vy + vz * vz;
        f_max_sq = std::max(f_max_sq, f2);
    }
}

void MDSimulation::apply_fire_update_cpu(const FireMinimizer::StepParams& params) {
    ParticleData& particles = system->get_particles();
    const SimulationBox& box = system->get_box();
    const size_t n = system->get_local_particles();
    const Real f_norm_inv = (params.mix_velocity && params.f_norm > 0.0)
        ? (1.0 / params.f_norm) : 0.0;

    for (size_t i = 0; i < n; i++) {
        Real vx = particles.vx[i], vy = particles.vy[i], vz = particles.vz[i];

        if (params.reset_velocity) {
            vx = vy = vz = 0.0;
        } else if (params.mix_velocity) {
            const Real fhx = particles.fx[i] * f_norm_inv;
            const Real fhy = particles.fy[i] * f_norm_inv;
            const Real fhz = particles.fz[i] * f_norm_inv;
            vx = (1.0 - params.alpha) * vx + params.alpha * params.v_norm * fhx;
            vy = (1.0 - params.alpha) * vy + params.alpha * params.v_norm * fhy;
            vz = (1.0 - params.alpha) * vz + params.alpha * params.v_norm * fhz;
        }

        const Real inv_m = 1.0 / particles.mass[i];
        vx += params.dt * particles.fx[i] * inv_m * FORCE_TO_ACCEL;
        vy += params.dt * particles.fy[i] * inv_m * FORCE_TO_ACCEL;
        vz += params.dt * particles.fz[i] * inv_m * FORCE_TO_ACCEL;

        particles.vx[i] = vx;
        particles.vy[i] = vy;
        particles.vz[i] = vz;
        particles.x[i] += params.dt * vx;
        particles.y[i] += params.dt * vy;
        particles.z[i] += params.dt * vz;
    }
}

void MDSimulation::run_energy_minimization() {
    if (minimizer_type_.empty() || minimizer_type_ == "none") {
        return;
    }
    if (minimizer_type_ != "fire") {
        throw std::runtime_error(
            "Unknown minimizer type: '" + minimizer_type_ + "' (supported: none, fire)");
    }

    FireConfig cfg;
    cfg.max_steps = minimizer_max_steps_;
    cfg.force_tolerance = minimizer_force_tolerance_;
    cfg.dt_start = (minimizer_dt_start_ > 0.0) ? minimizer_dt_start_ : (dt > 0.0 ? dt : 0.5);
    cfg.dt_max = (minimizer_dt_max_ > 0.0) ? minimizer_dt_max_ : (cfg.dt_start * 4.0);
    FireMinimizer fire(cfg);

    ForceCalculatorCUDA* cuda_calc = cuda_calc_cached_;

    if (system->get_rank() == 0) {
        std::cout << "Starting FIRE energy minimization (max_steps=" << cfg.max_steps
                  << ", force_tolerance=" << cfg.force_tolerance << " kcal/mol/A)" << std::endl;
    }

    zero_velocities();

    if (cuda_calc) {
#ifdef USE_CUDA
        cuda_calc->initial_upload_to_device(*system);
#endif
    }

    double neighbor_time = 0.0, comm_time = 0.0, migrate_time = 0.0, force_time = 0.0;
    neighbor_rebuild_forced = true;
    handle_neighbor_phase(neighbor_time, comm_time, migrate_time);
    handle_force_phase(cuda_calc, force_time);

    int iter = 0;
    bool converged = false;
    for (; iter < cfg.max_steps; ++iter) {
        Real local_power = 0.0, local_f2 = 0.0, local_v2 = 0.0, local_fmax2 = 0.0;

        if (cuda_calc) {
#ifdef USE_CUDA
            const auto local = cuda_calc->compute_fire_reduction_on_device(
                system->get_local_particles());
            local_power = local.power;
            local_f2 = local.f_norm_sq;
            local_v2 = local.v_norm_sq;
            local_fmax2 = local.f_max_sq;
#endif
        } else {
            compute_fire_reduction_cpu(local_power, local_f2, local_v2, local_fmax2);
        }

        const Real global_power = comm_backend_->sum_reduce(local_power);
        const Real global_f_norm = std::sqrt(std::max<Real>(comm_backend_->sum_reduce(local_f2), 0.0));
        const Real global_v_norm = std::sqrt(std::max<Real>(comm_backend_->sum_reduce(local_v2), 0.0));
        const Real fmax = std::sqrt(std::max<Real>(comm_backend_->max_reduce(local_fmax2), 0.0));

        if (!std::isfinite(global_power) || !std::isfinite(global_f_norm) ||
            !std::isfinite(global_v_norm) || !std::isfinite(fmax)) {
            throw std::runtime_error(
                "[minimize] FIRE diverged at step " + std::to_string(iter) +
                " (non-finite force/velocity norm). The starting geometry likely "
                "has severe atomic overlaps/clashes for the configured "
                "minimize.dt_start/minimize.dt_max; reduce the initial timestep "
                "or pre-relax the structure.");
        }

        if (system->get_rank() == 0 && (iter % minimizer_output_freq_ == 0)) {
            std::cout << "[minimize] step " << iter << "  PE=" << cached_potential_energy
                      << "  Fmax=" << fmax << std::endl;
        }

        if (fmax < cfg.force_tolerance) {
            converged = true;
            break;
        }

        const FireMinimizer::StepParams params =
            fire.advance(global_power, global_f_norm, global_v_norm);

        if (cuda_calc) {
#ifdef USE_CUDA
            cuda_calc->apply_fire_update_on_device(
                system->get_local_particles(), system->get_box(),
                params.reset_velocity, params.mix_velocity,
                params.alpha, params.v_norm, params.f_norm, params.dt);
#endif
        } else {
            apply_fire_update_cpu(params);
        }

        neighbor_rebuild_forced = true;
        handle_neighbor_phase(neighbor_time, comm_time, migrate_time);
        handle_force_phase(cuda_calc, force_time);
    }

    if (system->get_rank() == 0) {
        std::cout << "[minimize] " << (converged ? "converged" : "stopped at max_steps")
                  << " after " << iter << " step(s), PE=" << cached_potential_energy
                  << std::endl;
    }

    if (cuda_calc) {
#ifdef USE_CUDA
        cuda_calc->download_from_device(*system);
#endif
    }

    system->initialize_distributed_velocities(target_temperature);

    neighbor_rebuild_forced = true;
}


MDSimulation::TimingTraceMode MDSimulation::parse_timing_trace_mode() {
    const char* value = std::getenv("MD_TIMING_TRACE");
    if (!value) return TimingTraceMode::Off;
    const std::string v(value);
    if (v == "every") return TimingTraceMode::Every;
    if (v == "step") return TimingTraceMode::Step;
    return TimingTraceMode::Off;
}

void MDSimulation::run() {
    // Dedicated PME-group ranks (force_calculation.pme_ranks > 0) already ran
    // their entire simulation inside PmeServer::run(), driven from
    // initialize_impl's early return -- system/force_calc/thermostat/etc.
    // were never constructed on this rank, so there is nothing left to do.
#ifdef USE_CUDA
    if (pme_server_ || pme_server_cuda_) {
        return;
    }
#else
    if (pme_server_) {
        return;
    }
#endif
    timing_trace_mode_ = parse_timing_trace_mode();
    timing_trace_force_ = env_flag_enabled("MD_TIMING_TRACE_FORCE");
    if (force_calc) {
        force_calc->set_timing_breakdown_enabled(timing_trace_force_);
    }
    if (system->get_rank() == 0) {
        std::cout << "Starting MD simulation for " << total_steps << " steps" << std::endl;
        if (timing_trace_mode_ != TimingTraceMode::Off) {
            std::cout << "Per-step timing trace enabled (MD_TIMING_TRACE="
                      << (timing_trace_mode_ == TimingTraceMode::Every ? "every" : "step") << ")"
                      << (timing_trace_force_ ? ", with force-component breakdown" : "") << std::endl;
        } else if (timing_trace_force_) {
            std::cout << "Force-component timing enabled (MD_TIMING_TRACE_FORCE=1); "
                      << "will appear in the end-of-run report" << std::endl;
        }
    }

    current_step = 0;
    neighbor_rebuild_forced = false;
    adaptive_rebuild_armed_ = true;
    rebuild_reason_periodic_ = 0;
    rebuild_reason_adaptive_ = 0;
    rebuild_reason_forced_ = 0;
    rebuild_reason_mpi_sync_ = 0;
    rebuild_reason_box_change_ = 0;
    rebuild_reason_other_ = 0;

    const double minimize_time = measure_seconds([&]() {
        run_energy_minimization();
    });
    cumulative_timing.minimize_time += minimize_time;

    prepare_first_step();

    output_results(0);

    for (int step = 1; step <= total_steps; step++) {
        single_step();

        if (timing_trace_mode_ == TimingTraceMode::Every && should_print_timing_for_step(step)) {
            print_step_timing(step);
        }

        handle_step_output(step);
    }

    if (system->get_rank() == 0) {
        std::cout << "Neighbor list builds = " << neighbor_list_builds << std::endl;
        if (nonbonded_kernel_ == "graph") {
            std::cout << "Active graph incremental prunes = " << active_graph_prunes_
                      << ", safe reuses = " << active_graph_reuses_ << std::endl;
        }
        std::cout << "Neighbor rebuild reasons (overlapping counts):"
                  << " periodic=" << rebuild_reason_periodic_
                  << " adaptive=" << rebuild_reason_adaptive_
                  << " forced=" << rebuild_reason_forced_
                  << " mpi_sync=" << rebuild_reason_mpi_sync_
                  << " box_change=" << rebuild_reason_box_change_
                  << " other=" << rebuild_reason_other_
                  << std::endl;
    }
}

void MDSimulation::print_step_timing(int step) {
    if (timing_trace_mode_ == TimingTraceMode::Off || system->get_rank() != 0) {
        return;
    }

    std::cout << "[TIMING] step=" << step
              << " total_ms=" << last_timing.total_ms
              << " integrate_ms=" << last_timing.integrate_ms
              << " nlist_build_ms=" << last_timing.nlist_build_ms
              << " comm_ms=" << last_timing.comm_ms
              << " migrate_ms=" << last_timing.migrate_ms
              << " force_ms=" << last_timing.force_ms
              << " thermo_ms=" << last_timing.thermostat_ms
              << " ke_ms=" << last_timing.ke_ms
              << " baro_ms=" << last_timing.barostat_ms
              << " io_ms=" << last_timing.io_ms;
    for (const auto& [name, ms] : last_timing.force_components_ms) {
        std::cout << " force." << name << "_ms=" << ms;
    }
    std::cout << std::endl;
}

bool MDSimulation::should_output_step(int step) const {
    return (output_freq > 0) && (step % output_freq == 0);
}

bool MDSimulation::needs_ke_update_for_step(int step) const {
    return should_output_step(step) || should_write_trajectory_step(step) ||
           (step == total_steps);
}

bool MDSimulation::should_write_trajectory_step(int step) const {
    return (trajectory_output_freq > 0) && (step % trajectory_output_freq == 0);
}

bool MDSimulation::should_print_timing_for_step(int step) const {
    if (timing_trace_mode_ == TimingTraceMode::Off) {
        return false;
    }
    if (timing_trace_mode_ == TimingTraceMode::Every) {
        return true;
    }
    return should_output_step(step);
}

void MDSimulation::handle_step_output(int step) {
    const bool emit_text = should_output_step(step);
    const bool write_frame = should_write_trajectory_step(step);
    if (!emit_text && !write_frame) {
        return;
    }

    if (emit_text && should_print_timing_for_step(step) && timing_trace_mode_ != TimingTraceMode::Every) {
        print_step_timing(step);
    }

    output_results(step, emit_text);

    if (emit_text && system->get_rank() == 0) {
        std::cout << "Neighbor rebuild reasons through step " << step
                  << " (overlapping counts):"
                  << " periodic=" << rebuild_reason_periodic_
                  << " adaptive=" << rebuild_reason_adaptive_
                  << " forced=" << rebuild_reason_forced_
                  << " mpi_sync=" << rebuild_reason_mpi_sync_
                  << " box_change=" << rebuild_reason_box_change_
                  << " other=" << rebuild_reason_other_
                  << std::endl;
    }
}

void MDSimulation::compute_output_energies(const ParticleData& particles, OutputMetrics& metrics) {
    if (force_calc) {
        metrics.total_ke = force_calc->get_kinetic_energy() * mvv2e;
        metrics.total_pe = cached_potential_energy;
        return;
    }

    for (size_t i = 0; i < particles.num_particles; i++) {
        Real v2 = particles.vx[i] * particles.vx[i]
                + particles.vy[i] * particles.vy[i]
                + particles.vz[i] * particles.vz[i];
        metrics.total_ke += 0.5 * particles.mass[i] * v2;
    }
    metrics.total_ke *= mvv2e;
    metrics.total_pe = cached_potential_energy;
}

void MDSimulation::compute_output_force_speed_stats(const ParticleData& particles, OutputMetrics& metrics) {
    size_t num_particles_for_stats = system->get_local_particles();
    Real total_speed = 0.0;

    metrics.max_force_particle_id = 0;
    for (size_t i = 0; i < num_particles_for_stats; i++) {
        Real f_mag = std::sqrt(particles.fx[i] * particles.fx[i]
                             + particles.fy[i] * particles.fy[i]
                             + particles.fz[i] * particles.fz[i]);
        if (f_mag > metrics.max_force) {
            metrics.max_force = f_mag;
            metrics.max_force_particle_id =
                static_cast<size_t>(particles.global_id[i]);
        }

        Real speed = std::sqrt(particles.vx[i] * particles.vx[i]
                             + particles.vy[i] * particles.vy[i]
                             + particles.vz[i] * particles.vz[i]);
        if (speed > metrics.max_speed) {
            metrics.max_speed = speed;
        }
        total_speed += speed;
    }

    metrics.avg_speed = (particles.num_particles > 0) ? (total_speed / particles.num_particles) : 0.0;
}

void MDSimulation::compute_output_thermodynamics(const ParticleData& particles, OutputMetrics& metrics) {
    int local_particle_count = static_cast<int>(system->get_local_particles());
    int global_particles = comm_backend_->sum_int(local_particle_count);
    comm_backend_->reduce_max_force_particle(
        metrics.max_force, metrics.max_force_particle_id, system->get_rank(),
        metrics.max_force, metrics.max_force_particle_id);

    Real global_virial_for_press = 0.0;
    if (force_calc) {
        Real local_virial = force_calc->get_virial();
        global_virial_for_press = comm_backend_->sum_reduce(local_virial);
    }

    if (particles.num_particles > 0 && system->get_rank() == 0) {
        metrics.temperature = (2.0 * metrics.total_ke) / (3.0 * global_particles * kB);

        Real volume = system->get_box().lx * system->get_box().ly * system->get_box().lz;
        if (volume > 0.0) {
            metrics.pressure = (2.0 * metrics.total_ke + global_virial_for_press) / (3.0 * volume) * nktv2p;
        }
    }
}

void MDSimulation::emit_rank0_output(int step, const OutputMetrics& metrics) {
    if (system->get_rank() != 0) {
        return;
    }

    std::cout << format_output_line(step, metrics) << std::endl;
}

void MDSimulation::sync_output_state_to_host_if_needed() {
    ForceCalculatorCUDA* cuda_calc = cuda_calc_cached_;
    if (cuda_calc) {
#ifdef USE_CUDA
        cuda_calc->download_from_device(*system);
#endif
    }
}

std::string MDSimulation::format_output_line(int step, const OutputMetrics& metrics) const {
    std::stringstream output_line;
    output_line << std::fixed << std::setprecision(6);

    static const std::vector<OutputField> kDefaultOutputFields = {
        OutputField::STEP, OutputField::T, OutputField::KE
    };
    const std::vector<OutputField>& fields =
        output_fields_cache.empty() ? kDefaultOutputFields : output_fields_cache;

    for (OutputField field : fields) {
        switch (field) {
        case OutputField::STEP:
            output_line << "Step: " << step;
            break;
        case OutputField::T:
            output_line << ", T: " << metrics.temperature;
            break;
        case OutputField::KE:
            output_line << ", KE: " << metrics.total_ke;
            break;
        case OutputField::PE:
            output_line << ", PE: " << metrics.total_pe;
            break;
        case OutputField::ETOTAL:
            output_line << ", Etot: " << metrics.total_ke + metrics.total_pe;
            break;
        case OutputField::PRESS:
            output_line << ", P: " << metrics.pressure;
            break;
        case OutputField::MAX_SPEED:
            output_line << ", MaxSpd: " << metrics.max_speed;
            break;
        case OutputField::AVG_SPEED:
            output_line << ", AvgSpd: " << metrics.avg_speed;
            break;
        case OutputField::FMAX:
            output_line << ", Fmax: " << metrics.max_force
                        << " (particle " << metrics.max_force_particle_id << ")";
            break;
        }
    }

    return output_line.str();
}

void MDSimulation::output_results(int step, bool emit_text) {
    double output_time = measure_seconds([&]() {
        sync_output_state_to_host_if_needed();

        OutputMetrics metrics;
        ParticleData& particles = system->get_particles();

        compute_output_energies(particles, metrics);
        compute_output_force_speed_stats(particles, metrics);
        compute_output_thermodynamics(particles, metrics);

        if (emit_text) {
            emit_rank0_output(step, metrics);
        }

        if (should_write_trajectory_step(step)) {
            write_trajectory(step);
        }
    });

    last_timing.io_ms = output_time * 1000.0;
    cumulative_timing.output_time += output_time;
}

void MDSimulation::finalize(double program_wallclock_ms) {
#ifdef USE_CUDA
    if (pme_server_ || pme_server_cuda_) {
        return;
    }
#else
    if (pme_server_) {
        return;
    }
#endif
    // This is a PP rank (or PmePpSplit is inactive): tell the dedicated PME
    // group's PmeServer::run() receive loop to stop. Every prior call to
    // pme_reciprocal_begin() sent a nonzero "continue" flag on this same
    // intercommunicator (see its comment); this is the matching final zero,
    // sent exactly once, after the last such call for the whole run.
    if (PmePpSplit::instance().active()) {
        int pp_rank = 0;
        MPI_Comm_rank(PmePpSplit::instance().pp_comm(), &pp_rank);
        const int pp_leader_root = (pp_rank == 0) ? MPI_ROOT : MPI_PROC_NULL;
        int stop_flag = 0;
        MPI_Bcast(&stop_flag, 1, MPI_INT, pp_leader_root, PmePpSplit::instance().inter_comm());
    }
    if (cumulative_timing.step_count > 0) {
        print_timing_report(program_wallclock_ms);
    }

    if (force_calc) {
        ForceCalculatorCUDA* cuda_fin = cuda_calc_cached_;
        if (cuda_fin && force_calculation_type == "mace") {
#ifdef USE_CUDA
            cuda_fin->unregister_ml_pinned_buffers(system->get_particles());
#endif
        }
        force_calc->finalize();
    }
    
    if (python_interface) {
        python_interface->finalize_python();
    }
    
    finalize_trajectory_writers();

    if (system->get_rank() == 0) {
        std::cout << "MD simulation finalized" << std::endl;
    }

}

void MDSimulation::parse_simulation_config(const std::string& config_file) {
    std::ifstream file(config_file);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + config_file);
    }
    
    bool found_coordinate_file = false;
    bool found_target_temperature = false;
    bool found_timestep = false;
    bool found_total_steps = false;
    std::string input_system;
    
    std::string line;
    std::string current_section = "";
    

    while (std::getline(file, line)) {
        size_t comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line = line.substr(0, comment_pos);
        }
        
        if (line.empty()) continue;
        
        const bool is_indented = !line.empty() && (line[0] == ' ' || line[0] == '\t');

        bool is_section_header = false;
        if (!line.empty() && line[0] != ' ' && line[0] != '\t') {
            size_t colon_at = line.find(':');
            if (colon_at != std::string::npos) {
                std::string after_colon = line.substr(colon_at + 1);
                after_colon.erase(0, after_colon.find_first_not_of(" \t\r\n"));
                if (after_colon.empty()) {
                    is_section_header = true;
                    current_section = line.substr(0, colon_at);
                    current_section.erase(0, current_section.find_first_not_of(" \t"));
                    current_section.erase(current_section.find_last_not_of(" \t") + 1);
                }
            }
        }

        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);
        
        if (line.empty()) continue;
        
        if (is_section_header) continue;

        if (!is_indented) {
            current_section.clear();
        }
        
        size_t colon_pos = line.find(':');
        if (colon_pos == std::string::npos) continue;
        
        std::string key = line.substr(0, colon_pos);
        std::string value = line.substr(colon_pos + 1);
        
        const char* ws = " \t\n\r";
        
        size_t key_start = key.find_first_not_of(ws);
        if (key_start != std::string::npos) {
            size_t key_end = key.find_last_not_of(ws);
            key = key.substr(key_start, key_end - key_start + 1);
        } else {
            key.clear();
        }
        
        size_t value_start = value.find_first_not_of(ws);
        if (value_start != std::string::npos) {
            size_t value_end = value.find_last_not_of(ws);
            value = value.substr(value_start, value_end - value_start + 1);
        } else {
            value.clear();
        }
        
        if (!value.empty() && value.front() == '"' && value.back() == '"') {
            value = value.substr(1, value.length() - 2);
            size_t start = value.find_first_not_of(ws);
            if (start != std::string::npos) {
                size_t end = value.find_last_not_of(ws);
                value = value.substr(start, end - start + 1);
            } else {
                value.clear();
            }
        }
        
        if (current_section == "force_calculation") {
            if (key == "mode" || key == "force_calculation_mode") {
                force_calculation_mode = trim_copy(value);
            } else if (key == "type" || key == "force_calculation_type") {
                force_calculation_type = value;
            } else if (key == "spec_file" || key == "mlip_spec_file") {
                mlip_spec_file_ = value;
            } else if (key == "precision") {
                force_precision_ = trim_copy(value);
            } else if (key == "nonbonded_kernel") {
                nonbonded_kernel_ = trim_copy(value);
            } else if (key == "pme_cluster_spread") {
                std::string v = trim_copy(value);
                pme_cluster_spread_ = (v == "true" || v == "1" || v == "on");
            } else if (key == "electrostatics_solver") {
                electrostatics_solver_ = trim_copy(value);
            } else if (key == "pme_ranks") {
                pme_ranks_ = std::stoi(trim_copy(value));
            }
        } else if (current_section == "minimize") {
            if (key == "method") {
                minimizer_type_ = value;
            } else if (key == "max_steps") {
                minimizer_max_steps_ = std::stoi(value);
            } else if (key == "force_tolerance") {
                minimizer_force_tolerance_ = std::stod(value);
            } else if (key == "dt_start") {
                minimizer_dt_start_ = std::stod(value);
            } else if (key == "dt_max") {
                minimizer_dt_max_ = std::stod(value);
            } else if (key == "output_freq") {
                minimizer_output_freq_ = std::stoi(value);
            }
        } else if (current_section == "mlip_config" || current_section == "mace_config") {
            if (key == "model_path") {
                mlip_model_path = value;
            } else if (key == "model_type") {
                mlip_model_type = value;
            } else if (key == "model_size") {
                mlip_model_size = value;
            } else if (key == "device") {
                mlip_device = value;
            } else if (key == "dtype") {
                mlip_dtype = value;
            } else if (!key.empty()) {
                mlip_extra_kwargs[key] = value;
            }
        } else if (key == "input_system") {
            input_system = value;
        } else if (key == "coordinate_file") {
            coordinate_file = value;
            found_coordinate_file = true;
        } else if (key == "topology_file") {
            topology_file = value;
        } else if (key == "parameter_file") {
            parameter_file = value;
        } else if (key == "target_temperature") {
            target_temperature = std::stod(value);
            found_target_temperature = true;
        } else if (key == "timestep") {
            dt = std::stod(value);
            found_timestep = true;
        } else if (key == "total_steps") {
            total_steps = std::stoi(value);
            found_total_steps = true;
        } else if (key == "output_prefix") {
            output_prefix = value;
        } else if (key == "trajectory_output_freq") {
            trajectory_output_freq = std::stoi(value);
        } else if (key == "trajectory_format") {
            trajectory_format = value;
        } else if (key == "trajectory_include_velocities") {
            trajectory_include_velocities = (value == "true");
        } else if (key == "trajectory_include_forces") {
            trajectory_include_forces = (value == "true");
        } else if (key == "energy_output_freq") {
            energy_output_freq = std::stoi(value);
        } else if (key == "output_freq") {
            output_freq = std::stoi(value);
        } else if (key == "output_info") {
            output_info = value;
        } else if (key == "cutoff_distance") {
            cutoff_distance = std::stod(value);
        } else if (key == "cutoff") {
            cutoff_distance = std::stod(value);
        } else if (key == "lj_switch_distance" || key == "switchdist") {
            lj_switch_distance = std::stod(value);
        } else if (key == "neighbor_list_update_freq") {
            neighbor_list_update_freq = std::stoi(value);
        } else if (key == "verlet_skin") {
            verlet_skin = std::stod(value);
        } else if (key == "neighbor_rebuild_hi_frac") {
            neighbor_rebuild_hi_frac = std::stod(value);
        } else if (key == "neighbor_rebuild_lo_frac") {
            neighbor_rebuild_lo_frac = std::stod(value);
        } else if (key == "neighbor_displacement_check_stride") {
            neighbor_displacement_check_stride = std::stoi(value);
        } else if (key == "integrator") {
            integrator_type = value;
        } else if (key == "thermostat") {
            thermostat_type = value;
        } else if (key == "tau_t") {
            tau_t = std::stod(value);
        } else if (key == "thermostat_damping") {
            tau_t = std::stod(value);  // thermostat_damping is alias for tau_t
        } else if (key == "barostat") {
            barostat_type = value;
        } else if (key == "target_pressure") {
            target_pressure = std::stod(value);
        } else if (key == "tau_p") {
            tau_p = std::stod(value);
        } else if (key == "barostat_damping") {
            tau_p = std::stod(value);  // barostat_damping is alias for tau_p
        } else if (key == "tau_p_decay" || key == "langevin_piston_decay") {
            tau_p_decay_ = std::stod(value);
        } else if (key == "langevin_piston_temp") {
            langevin_piston_temp_ = std::stod(value);
        } else if (key == "python_force_path") {
            python_force_path = value;
        } else if (key == "python_force_enabled") {
            python_force_enabled = (value == "true");
        } else if (key == "mlip_spec_file") {
            mlip_spec_file_ = value;
        } else if (key == "box_x") {
            system->get_box().lx = std::stod(value);
        } else if (key == "box_y") {
            system->get_box().ly = std::stod(value);
        } else if (key == "box_z") {
            system->get_box().lz = std::stod(value);
        } else if (key == "periodic_x") {
            system->get_box().periodic_x = (value == "true");
        } else if (key == "periodic_y") {
            system->get_box().periodic_y = (value == "true");
        } else if (key == "periodic_z") {
            system->get_box().periodic_z = (value == "true");
        }
    }
    
    if (!input_system.empty()) {
        const auto manifest = read_dsmd_input_manifest(input_system);
        const std::filesystem::path base =
            std::filesystem::absolute(std::filesystem::path(input_system)).parent_path();
        auto resolve = [&](const char* key) -> std::string {
            auto it = manifest.find(key);
            if (it == manifest.end()) return "";
            const std::filesystem::path path(it->second);
            return (path.is_absolute() ? path : base / path).lexically_normal().string();
        };
        if (coordinate_file.empty()) coordinate_file = resolve("coordinate_file");
        if (topology_file.empty()) topology_file = resolve("topology_file");
        if (parameter_file.empty()) parameter_file = resolve("parameter_file");
        found_coordinate_file = !coordinate_file.empty();
    }

    std::vector<std::string> missing_params;
    if (!found_coordinate_file) missing_params.push_back("coordinate_file");
    if (!input_system.empty() && topology_file.empty()) missing_params.push_back("topology_file");
    if (!input_system.empty() && parameter_file.empty()) missing_params.push_back("parameter_file");
    if (!found_target_temperature) missing_params.push_back("target_temperature");
    if (!found_timestep) missing_params.push_back("timestep");
    if (!found_total_steps) missing_params.push_back("total_steps");
    
    if (!missing_params.empty()) {
        std::string error_msg = "Missing required configuration parameters: ";
        for (size_t i = 0; i < missing_params.size(); ++i) {
            if (i > 0) error_msg += ", ";
            error_msg += missing_params[i];
        }
        throw std::runtime_error(error_msg);
    }
    
    force_calculation_mode = trim_copy(force_calculation_mode);
    const auto force_components = parse_force_components(force_calculation_mode);
    for (const auto& component : force_components) {
        if (component != "builtin" && component != "python" && component != "libtorch") {
            throw std::runtime_error(
                "Invalid force_calculation.mode component: '" + component
                + "'. Supported components are: builtin, python, libtorch");
        }
    }
    if (force_components.empty()) {
        throw std::runtime_error(
            "Invalid force_calculation.mode: '" + force_calculation_mode
            + "'. Specify one or more of: builtin, python, libtorch");
    }
    enable_builtin_forces_  = force_components.count("builtin") != 0;
    enable_python_forces_   = force_components.count("python") != 0;
    enable_libtorch_forces_ = force_components.count("libtorch") != 0;

    force_precision_ = trim_copy(force_precision_);
    if (force_precision_.empty()) {
        force_precision_ = "double";
    }
    if (force_precision_ != "double" && force_precision_ != "mixed") {
        throw std::runtime_error(
            "Invalid force_calculation.precision: '" + force_precision_
            + "'. Supported values are: double, mixed");
    }

    nonbonded_kernel_ = trim_copy(nonbonded_kernel_);
    if (nonbonded_kernel_.empty()) {
        nonbonded_kernel_ = "tiled";
    }
    std::transform(nonbonded_kernel_.begin(), nonbonded_kernel_.end(),
                   nonbonded_kernel_.begin(), [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    if (nonbonded_kernel_ != "tiled" && nonbonded_kernel_ != "graph") {
        throw std::runtime_error(
            "Invalid force_calculation.nonbonded_kernel: '" + nonbonded_kernel_
            + "'. Supported values are: tiled, graph");
    }

    electrostatics_solver_ = trim_copy(electrostatics_solver_);
    if (electrostatics_solver_.empty()) {
        electrostatics_solver_ = "pme";
    }
    if (electrostatics_solver_ != "pme" && electrostatics_solver_ != "fmm") {
        throw std::runtime_error(
            "Invalid force_calculation.electrostatics_solver: '" + electrostatics_solver_
            + "'. Supported values are: pme, fmm");
    }
    if (electrostatics_solver_ == "fmm") {
        throw std::runtime_error(
            "force_calculation.electrostatics_solver = fmm is a placeholder for a "
            "future Fast Multipole Method solver and is not implemented yet. Use "
            "'pme' (the default).");
    }

    if (pme_ranks_ < 0) {
        throw std::runtime_error(
            "Invalid force_calculation.pme_ranks: must be >= 0.");
    }
#ifndef DEEPSCALEMD_HAVE_FFTW_MPI
    if (pme_ranks_ > 1) {
        throw std::runtime_error(
            "force_calculation.pme_ranks > 1 requires a build with "
            "libfftw3_mpi available (CMake did not find it -- see the "
            "\"DeepScaleMD: libfftw3_mpi NOT found\" configure message). "
            "Use 0 (no split, default) or 1, or rebuild with libfftw3_mpi "
            "installed. (This applies even on CUDA builds: a PME group of "
            "more than one rank runs on the CPU FFTW-MPI engine, not the "
            "GPU -- see MDSimulation::initialize_impl.)");
    }
#endif
    if (pme_ranks_ > 0) {
        int world_size = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        if (pme_ranks_ >= world_size) {
            throw std::runtime_error(
                "Invalid force_calculation.pme_ranks: must leave at least one "
                "PP rank (pme_ranks must be < total MPI rank count).");
        }
#ifdef USE_SYCL
        throw std::runtime_error(
            "force_calculation.pme_ranks > 0 is not supported on the SYCL "
            "build: the SYCL force backend has no PME/reciprocal-space "
            "implementation at all (real-space Coulomb only), so a "
            "dedicated PME rank group would have nothing to talk to. Use "
            "pme_ranks: 0 (the default) on this build.");
#endif
    }


    {
        std::string normalized = trajectory_format;
        if (trim_copy(normalized) == "both") normalized = "xyz,dump";
        trajectory_formats_ = parse_force_components(normalized);
        for (const auto& fmt : trajectory_formats_) {
            if (fmt != "xyz" && fmt != "dump" && fmt != "dcd" && fmt != "bp" && fmt != "h5md") {
                throw std::runtime_error(
                    "Invalid trajectory_format component: '" + fmt
                    + "'. Supported components are: xyz, dump, dcd, bp, h5md (or legacy 'both')");
            }
        }
#ifndef USE_ADIOS2
        if (trajectory_formats_.count("bp")) {
            throw std::runtime_error(
                "trajectory_format 'bp' requires a build with ADIOS2 enabled "
                "(-DENABLE_ADIOS2=ON at CMake configure time).");
        }
#endif
#ifndef USE_HDF5
        if (trajectory_formats_.count("h5md")) {
            throw std::runtime_error(
                "trajectory_format 'h5md' requires a build with parallel HDF5 enabled "
                "(-DENABLE_HDF5=ON at CMake configure time).");
        }
#endif
    }

    if (dt <= 0) {
        throw std::runtime_error("Invalid timestep: must be positive");
    }
    if (total_steps <= 0) {
        throw std::runtime_error("Invalid total_steps: must be positive");
    }
    if (target_temperature < 0) {
        throw std::runtime_error("Invalid target_temperature: must be non-negative");
    }
    if (thermostat_type != "none" && !thermostat_type.empty() && tau_t <= 0.0) {
        throw std::runtime_error("Invalid thermostat damping time: must be positive");
    }
    if (minimizer_type_ != "none" && !minimizer_type_.empty()) {
        if (minimizer_type_ != "fire") {
            throw std::runtime_error(
                "Invalid minimize.method: '" + minimizer_type_ + "' (supported: none, fire)");
        }
        if (minimizer_max_steps_ <= 0) {
            throw std::runtime_error("Invalid minimize.max_steps: must be positive");
        }
        if (minimizer_force_tolerance_ <= 0.0) {
            throw std::runtime_error("Invalid minimize.force_tolerance: must be positive");
        }
    }
    if (neighbor_list_update_freq <= 0) {
        throw std::runtime_error("Invalid neighbor_list_update_freq: must be positive");
    }
    if (!(neighbor_rebuild_lo_frac > 0.0 && neighbor_rebuild_lo_frac < 1.0)) {
        throw std::runtime_error("Invalid neighbor_rebuild_lo_frac: must be in (0, 1)");
    }
    if (!(neighbor_rebuild_hi_frac > 0.0 && neighbor_rebuild_hi_frac < 1.0)) {
        throw std::runtime_error("Invalid neighbor_rebuild_hi_frac: must be in (0, 1)");
    }
    if (neighbor_rebuild_lo_frac >= neighbor_rebuild_hi_frac) {
        throw std::runtime_error("Invalid adaptive rebuild thresholds: neighbor_rebuild_lo_frac must be smaller than neighbor_rebuild_hi_frac");
    }
    if (neighbor_rebuild_hi_frac > 0.5) {
        throw std::runtime_error(
            "Invalid neighbor_rebuild_hi_frac: must be <= 0.5 to preserve the Verlet-skin "
            "safety invariant (two particles closing head-on must not together cover more "
            "than the skin distance before a rebuild is forced); got " +
            std::to_string(neighbor_rebuild_hi_frac));
    }
    if (neighbor_displacement_check_stride < 0) {
        throw std::runtime_error("Invalid neighbor_displacement_check_stride: must be non-negative (0 means auto)");
    }
    if (lj_switch_distance >= 0.0 &&
        (lj_switch_distance <= 0.0 || lj_switch_distance >= cutoff_distance)) {
        throw std::runtime_error("Invalid lj_switch_distance: must be positive and smaller than cutoff_distance");
    }
    
    system->set_physical_cutoff(cutoff_distance);
    system->set_verlet_skin(verlet_skin);
    system->get_ff_params().lj_switch_distance = lj_switch_distance;
    rebuild_output_info_items_cache();
    if (system->get_rank() == 0) {
        std::cout << "Configuration loaded successfully:" << std::endl;
        std::cout << "  Coordinate file: " << coordinate_file << std::endl;
        if (!parameter_file.empty()) {
            std::cout << "  Parameter file: " << parameter_file << std::endl;
        }
        std::cout << "  Target temperature: " << target_temperature << " K" << std::endl;
        std::cout << "  Timestep: " << dt << " fs" << std::endl;
        std::cout << "  Total steps: " << total_steps << std::endl;
        std::cout << "  Integrator: " << integrator_type << std::endl;
        std::cout << "  Thermostat: " << thermostat_type << std::endl;
        std::cout << "  Barostat: " << barostat_type << std::endl;
        if (minimizer_type_ != "none" && !minimizer_type_.empty()) {
            std::cout << "  Minimizer: " << minimizer_type_
                      << " (max_steps=" << minimizer_max_steps_
                      << ", force_tolerance=" << minimizer_force_tolerance_
                      << " kcal/mol/A)" << std::endl;
        }
        {
            std::string fmt_list;
            for (const auto& fmt : trajectory_formats_) {
                if (!fmt_list.empty()) fmt_list += ", ";
                fmt_list += fmt;
            }
            std::cout << "  Trajectory formats: " << fmt_list << std::endl;
        }
        std::cout << "  Box size: " << system->get_box().lx << " x " << system->get_box().ly << " x " << system->get_box().lz << std::endl;
        std::cout << "  Periodic BC: " << (system->get_box().periodic_x ? "T" : "F") 
                  << " " << (system->get_box().periodic_y ? "T" : "F") 
                  << " " << (system->get_box().periodic_z ? "T" : "F") << std::endl;
        std::cout << "  Cutoff distance: " << cutoff_distance << std::endl;
        if (lj_switch_distance > 0.0) {
            std::cout << "  LJ switch distance: " << lj_switch_distance << std::endl;
        }
        std::cout << "  Neighbor list update frequency: " << neighbor_list_update_freq << " steps" << std::endl;
        std::cout << "  Neighbor adaptive rebuild: hi=" << neighbor_rebuild_hi_frac
              << "*skin, lo=" << neighbor_rebuild_lo_frac
              << "*skin, check_stride="
              << (neighbor_displacement_check_stride > 0
                  ? std::to_string(neighbor_displacement_check_stride)
                  : std::string("auto(update_freq/4)"))
              << std::endl;
        std::cout << "  Force calculation mode: " << force_calculation_mode << std::endl;
        if (enable_python_forces_) {
            std::cout << "  Python force path: " << python_force_path << std::endl;
        }
        std::cout << "  Python force interface: " << (python_force_enabled ? "enabled" : "disabled") << std::endl;
        if (python_force_enabled) {
            std::cout << "  Python force path: " << python_force_path << std::endl;
        }
    }
}

void MDSimulation::initialize_integrator() {
    const char* label = backend_label();

    if (integrator_type == "velocity_verlet") {
        integrator = create_velocity_verlet_integrator_for_build(dt);
        if (system->get_rank() == 0) {
            std::cout << "Initialized " << label
                      << " Velocity-Verlet integrator" << std::endl;
        }

    } else if (integrator_type == "r_respa") {
        throw std::runtime_error(
            "r_respa requires explicit fast/slow force components and is not supported by the current force API");

    } else if (integrator_type == "leap_frog" || integrator_type == "leapfrog") {
        integrator = create_leap_frog_integrator_for_build(dt);
        if (system->get_rank() == 0) {
            std::cout << "Initialized " << label
                      << " Leap-frog integrator" << std::endl;
        }
    } else {
        integrator = create_velocity_verlet_integrator_for_build(dt);
        if (system->get_rank() == 0) {
            std::cout << "Unknown integrator type '" << integrator_type 
                      << "', using Velocity-Verlet as default" << std::endl;
        }
    }
}

void MDSimulation::initialize_thermostat() {
    const char* label = backend_label();

    if (thermostat_type == "none" || thermostat_type.empty()) {
        thermostat = nullptr;
        if (system->get_rank() == 0) {
            std::cout << "No thermostat initialized" << std::endl;
        }
        return;
    }
    
    if (thermostat_type == "nose_hoover") {
        Real Q = tau_t;
        thermostat = create_nose_hoover_thermostat_for_build(target_temperature, Q);
        if (system->get_rank() == 0) {
            std::cout << "Initialized " << label
                      << " Nose-Hoover thermostat (T=" << target_temperature << " K)" << std::endl;
        }
    } else if (thermostat_type == "langevin") {
        Real gamma = 1.0 / tau_t; // friction coefficient from time constant
        thermostat = create_langevin_thermostat_for_build(target_temperature, gamma);
        if (system->get_rank() == 0) {
            std::cout << "Initialized " << label
                      << " Langevin thermostat (T=" << target_temperature 
                      << " K, tau=" << tau_t << " ps)" << std::endl;
        }
    } else if (thermostat_type == "andersen") {
        Real collision_freq = 1.0 / tau_t; // collision frequency from time constant
        thermostat = create_andersen_thermostat_for_build(target_temperature, collision_freq);
        if (system->get_rank() == 0) {
            std::cout << "Initialized " << label
                      << " Andersen thermostat (T=" << target_temperature 
                      << " K, freq=" << collision_freq << " ps^-1)" << std::endl;
        }
    } else {
        if (system->get_rank() == 0) {
            std::cout << "Unknown thermostat type '" << thermostat_type 
                      << "', no thermostat initialized" << std::endl;
        }
    }
}

void MDSimulation::initialize_barostat() {
    const char* label = backend_label();

    if (barostat_type == "none" || barostat_type.empty()) {
        barostat = nullptr;
        if (system->get_rank() == 0 ) {
            std::cout << "No barostat initialized" << std::endl;
        }
        return;
    }
    
    bool ok = true;

    ok = (dt > 0.0) && (tau_p > 0.0) && std::isfinite(target_pressure);
    ok = comm_backend_->all_true(ok);
    if (!ok) {
        if (system->get_rank() == 0) {
            std::cerr << "Invalid NPT parameters; disabling barostat on every rank." << std::endl;
        }
        barostat_type = "none";
        barostat.reset();
        return;
    }
    
    if (barostat_type == "parrinello_rahman") {
        int local_particles = static_cast<int>(system->get_local_particles());
        int global_particles = comm_backend_->sum_int(local_particles);
        Real thermal_energy = std::max<Real>(target_temperature, 1.0) * kB;
        Real W = (3.0 * global_particles + 3.0) * thermal_energy * tau_p * tau_p;

        // tau_p alone only supplies the piston's inertia (NAMD's
        // langevinPistonPeriod). Without a friction term the piston is a
        // lossless oscillator that never settles on the target pressure --
        // it just rings, which is what NAMD's langevinPistonDecay (and the
        // matching thermal kick from langevinPistonTemp) is for. Default the
        // decay to tau_p/2 when unset, matching NAMD's own 200fs/100fs ratio.
        const Real tau_p_decay = (tau_p_decay_ > 0.0) ? tau_p_decay_ : (0.5 * tau_p);
        const Real piston_gamma = 1.0 / tau_p_decay;
        const Real piston_temp = (langevin_piston_temp_ > 0.0)
            ? langevin_piston_temp_ : target_temperature;

        if (system->get_rank() == 0) {
            std::cout << "[DEBUG] Isotropic Parrinello-Rahman: tau_p=" << tau_p
                      << " W=" << W << " tau_p_decay=" << tau_p_decay
                      << " piston_temp=" << piston_temp << std::endl;
        }
        barostat = create_parrinello_rahman_barostat_for_build(target_pressure, W,
                                                                 piston_gamma, piston_temp);
        if (system->get_rank() == 0) {
            std::cout << "Initialized " << label
                  << " Parrinello-Rahman barostat (P=" << target_pressure
                      << " bar)" << std::endl;
        }
    } else if (barostat_type == "andersen") {
        Real compressibility = 1e-5 / tau_p;
        barostat = create_andersen_barostat_for_build(target_pressure, compressibility);
        if (system->get_rank() == 0) {
            std::cout << "Initialized " << label
                  << " Andersen barostat (P=" << target_pressure 
                      << " bar, tau=" << tau_p << " ps)" << std::endl;
        }
    } else if (barostat_type == "berendsen") {
        barostat = create_berendsen_barostat_for_build(target_pressure, tau_p);
        if (system->get_rank() == 0) {
            std::cout << "Initialized " << label
                  << " Berendsen barostat (P=" << target_pressure 
                      << " bar, tau=" << tau_p << " fs)" << std::endl;
        }
    } else {
        if (system->get_rank() == 0) {
            std::cout << "Unknown barostat type '" << barostat_type 
                      << "', no barostat initialized" << std::endl;
        }
    }
}

void MDSimulation::check_energy_stability() {
    static Real last_total_energy = 0.0;
    static bool first_check = true;
    
    if (!force_calc) return;
    
    Real kinetic_energy = force_calc->get_kinetic_energy();
    Real potential_energy = force_calc->get_potential_energy();
    Real total_energy = kinetic_energy + potential_energy;
    
    if (first_check) {
        last_total_energy = total_energy;
        first_check = false;
        return;
    }
    
    Real energy_change = std::abs(total_energy - last_total_energy);
    Real relative_change = energy_change / std::abs(last_total_energy + 1e-10);
    
    if (relative_change > 0.1) {
        if (system->get_rank() == 0) {
            std::cout << "Warning: Large energy change detected at step " << current_step 
                      << " (ΔE/E = " << relative_change << ")" << std::endl;
            std::cout << "  Previous total energy: " << last_total_energy << " kcal/mol" << std::endl;
            std::cout << "  Current total energy:  " << total_energy << " kcal/mol" << std::endl;
            std::cout << "  Kinetic energy:       " << kinetic_energy << " kcal/mol" << std::endl;
            std::cout << "  Potential energy:     " << potential_energy << " kcal/mol" << std::endl;
        }
        
        if (total_energy > last_total_energy * 1.05) { // More sensitive threshold
            dt *= 0.5; // Reduce timestep by 50%
            if (system->get_rank() == 0) {
                std::cout << "  Reducing timestep to " << dt << " ps" << std::endl;
            }
            
            if (dt < 1e-6) {
                if (system->get_rank() == 0) {
                    std::cout << "ERROR: Timestep became too small (dt < 1e-6 ps). Simulation unstable!" << std::endl;
                }
                throw std::runtime_error("Simulation became numerically unstable");
            }
        }
    }
    
    last_total_energy = total_energy;
}

void MDSimulation::gather_particles_for_serial_output(SerialTrajectoryFrame& frame) const {
    const int rank = system->get_rank();
    const int size = system->get_size();
    const size_t local_n = system->get_local_particles();
    ParticleData& particles = system->get_particles();
    const bool need_v = trajectory_include_velocities;
    const bool need_f = trajectory_include_forces;

    if (size == 1) {
        frame.x.resize(local_n); frame.y.resize(local_n); frame.z.resize(local_n);
        frame.type.resize(local_n); frame.element.resize(local_n);
        if (need_v) { frame.vx.resize(local_n); frame.vy.resize(local_n); frame.vz.resize(local_n); }
        if (need_f) { frame.fx.resize(local_n); frame.fy.resize(local_n); frame.fz.resize(local_n); }
        for (size_t k = 0; k < local_n; ++k) {
            const size_t gid = static_cast<size_t>(particles.global_id[k]);
            if (gid >= local_n) {
                throw std::runtime_error("Invalid global_id while gathering serial trajectory output");
            }
            frame.x[gid] = particles.x[k]; frame.y[gid] = particles.y[k]; frame.z[gid] = particles.z[k];
            frame.type[gid] = particles.type[k];
            frame.element[gid] = system->has_topology()
                ? system->get_topology().get_atom_element(static_cast<int>(gid)) : "C";
            if (need_v) { frame.vx[gid] = particles.vx[k]; frame.vy[gid] = particles.vy[k]; frame.vz[gid] = particles.vz[k]; }
            if (need_f) { frame.fx[gid] = particles.fx[k]; frame.fy[gid] = particles.fy[k]; frame.fz[gid] = particles.fz[k]; }
        }
        return;
    }

    const int local_count = static_cast<int>(local_n);
    std::vector<int> counts, displs;
    int total = 0;
    if (rank == 0) counts.resize(size);
    MPI_Gather(&local_count, 1, MPI_INT, rank == 0 ? counts.data() : nullptr, 1, MPI_INT,
               0, PmePpSplit::instance().pp_comm());
    if (rank == 0) {
        displs.resize(size);
        for (int r = 0; r < size; ++r) { displs[r] = total; total += counts[r]; }
    }

    std::vector<int> local_gid(particles.global_id.begin(), particles.global_id.begin() + local_n);
    std::vector<int> local_type(particles.type.begin(), particles.type.begin() + local_n);
    std::vector<int> gathered_gid(rank == 0 ? total : 0);
    std::vector<int> gathered_type(rank == 0 ? total : 0);
    MPI_Gatherv(local_gid.data(), local_count, MPI_INT,
                rank == 0 ? gathered_gid.data() : nullptr,
                rank == 0 ? counts.data() : nullptr, rank == 0 ? displs.data() : nullptr,
                MPI_INT, 0, PmePpSplit::instance().pp_comm());
    MPI_Gatherv(local_type.data(), local_count, MPI_INT,
                rank == 0 ? gathered_type.data() : nullptr,
                rank == 0 ? counts.data() : nullptr, rank == 0 ? displs.data() : nullptr,
                MPI_INT, 0, PmePpSplit::instance().pp_comm());

    auto gather_field = [&](const Real* local_ptr, std::vector<Real>& scratch) {
        scratch.resize(rank == 0 ? total : 0);
        MPI_Gatherv(local_ptr, local_count, MPI_DOUBLE,
                    rank == 0 ? scratch.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr, rank == 0 ? displs.data() : nullptr,
                    MPI_DOUBLE, 0, PmePpSplit::instance().pp_comm());
    };

    std::vector<Real> tmp_x, tmp_y, tmp_z;
    gather_field(particles.x.data(), tmp_x);
    gather_field(particles.y.data(), tmp_y);
    gather_field(particles.z.data(), tmp_z);

    std::vector<Real> tmp_vx, tmp_vy, tmp_vz, tmp_fx, tmp_fy, tmp_fz;
    if (need_v) {
        gather_field(particles.vx.data(), tmp_vx);
        gather_field(particles.vy.data(), tmp_vy);
        gather_field(particles.vz.data(), tmp_vz);
    }
    if (need_f) {
        gather_field(particles.fx.data(), tmp_fx);
        gather_field(particles.fy.data(), tmp_fy);
        gather_field(particles.fz.data(), tmp_fz);
    }

    if (rank != 0) return;

    frame.x.resize(total); frame.y.resize(total); frame.z.resize(total);
    frame.type.resize(total); frame.element.resize(total);
    if (need_v) { frame.vx.resize(total); frame.vy.resize(total); frame.vz.resize(total); }
    if (need_f) { frame.fx.resize(total); frame.fy.resize(total); frame.fz.resize(total); }

    for (int k = 0; k < total; ++k) {
        const int gid = gathered_gid[k];
        if (gid < 0 || gid >= total) {
            throw std::runtime_error("Invalid global_id while gathering serial trajectory output");
        }
        frame.x[gid] = tmp_x[k]; frame.y[gid] = tmp_y[k]; frame.z[gid] = tmp_z[k];
        frame.type[gid] = gathered_type[k];
        if (need_v) { frame.vx[gid] = tmp_vx[k]; frame.vy[gid] = tmp_vy[k]; frame.vz[gid] = tmp_vz[k]; }
        if (need_f) { frame.fx[gid] = tmp_fx[k]; frame.fy[gid] = tmp_fy[k]; frame.fz[gid] = tmp_fz[k]; }
    }
    for (int gid = 0; gid < total; ++gid) {
        frame.element[gid] = system->has_topology()
            ? system->get_topology().get_atom_element(gid) : "C";
    }
}

void MDSimulation::write_trajectory(int step) {
    if (trajectory_formats_.empty()) return;

    const bool need_serial = trajectory_formats_.count("xyz") || trajectory_formats_.count("dump")
        || trajectory_formats_.count("dcd");

    if (need_serial) {
        SerialTrajectoryFrame frame;
        gather_particles_for_serial_output(frame);

        if (system->get_rank() == 0) {
            if (trajectory_formats_.count("xyz"))  write_xyz_trajectory(step, frame);
            if (trajectory_formats_.count("dump")) write_lammps_dump(step, frame);
            if (trajectory_formats_.count("dcd"))  write_dcd_frame(step, frame);
        }
    }

#ifdef USE_ADIOS2
    if (trajectory_formats_.count("bp")) {
        write_adios2_frame(step);
    }
#endif

#ifdef USE_HDF5
    if (trajectory_formats_.count("h5md")) {
        write_h5md_frame(step);
    }
#endif
}

void MDSimulation::write_xyz_trajectory(int step, const SerialTrajectoryFrame& frame) {
    std::string filename = output_prefix + "_" + std::to_string(step) + ".xyz";
    std::ofstream traj_file(filename);

    if (!traj_file.is_open()) {
        std::cerr << "Warning: Cannot open trajectory file " << filename << std::endl;
        return;
    }

    const size_t n = frame.x.size();

    traj_file << n << "\n";
    traj_file << "Step " << step << " Time " << std::fixed << std::setprecision(3)
              << step * dt << " fs";

    if (force_calc) {
        Real ke = force_calc->get_kinetic_energy() * mvv2e;
        Real pe = cached_potential_energy;
        traj_file << " KE=" << std::setprecision(6) << ke
                  << " PE=" << pe << " Total=" << (ke + pe);
    }
    traj_file << "\n";

    for (size_t i = 0; i < n; i++) {
        traj_file << frame.element[i] << " "
                  << std::fixed << std::setprecision(6)
                  << frame.x[i] << " "
                  << frame.y[i] << " "
                  << frame.z[i];

        if (trajectory_include_velocities) {
            traj_file << " " << frame.vx[i] << " "
                      << frame.vy[i] << " " << frame.vz[i];
        }

        if (trajectory_include_forces) {
            traj_file << " " << frame.fx[i] << " "
                      << frame.fy[i] << " " << frame.fz[i];
        }

        traj_file << "\n";
    }

    traj_file.close();
}

void MDSimulation::write_lammps_dump(int step, const SerialTrajectoryFrame& frame) {
    std::string filename = output_prefix + "_dump." + std::to_string(step);
    std::ofstream dump_file(filename);

    if (!dump_file.is_open()) {
        std::cerr << "Warning: Cannot open dump file " << filename << std::endl;
        return;
    }

    const size_t n = frame.x.size();
    SimulationBox& box = system->get_box();

    dump_file << "ITEM: TIMESTEP\n" << step << "\n";
    dump_file << "ITEM: NUMBER OF ATOMS\n" << n << "\n";

    dump_file << "ITEM: BOX BOUNDS pp pp pp\n";
    dump_file << "0.0 " << std::fixed << std::setprecision(6) << box.lx << "\n";
    dump_file << "0.0 " << box.ly << "\n";
    dump_file << "0.0 " << box.lz << "\n";

    dump_file << "ITEM: ATOMS id type x y z";
    if (trajectory_include_velocities) {
        dump_file << " vx vy vz";
    }
    if (trajectory_include_forces) {
        dump_file << " fx fy fz";
    }
    dump_file << "\n";

    for (size_t i = 0; i < n; i++) {
        dump_file << (i + 1) << " " << frame.type[i] << " "
                  << std::fixed << std::setprecision(6)
                  << frame.x[i] << " "
                  << frame.y[i] << " "
                  << frame.z[i];

        if (trajectory_include_velocities) {
            dump_file << " " << frame.vx[i] << " "
                      << frame.vy[i] << " " << frame.vz[i];
        }

        if (trajectory_include_forces) {
            dump_file << " " << frame.fx[i] << " "
                      << frame.fy[i] << " " << frame.fz[i];
        }

        dump_file << "\n";
    }

    dump_file.close();
}

void MDSimulation::write_dcd_header(int32_t natom_global) {
    const int32_t reclen1 = 84;
    dcd_file_.write(reinterpret_cast<const char*>(&reclen1), 4);
    dcd_file_.write("CORD", 4);

    int32_t icntrl[20] = {0};
    icntrl[2] = std::max(1, trajectory_output_freq);  // NSAVC
    icntrl[3] = total_steps;                          // NSTEP (cosmetic)
    const float delta_akma = static_cast<float>(dt / 48.888211);
    std::memcpy(&icntrl[9], &delta_akma, sizeof(float));
    icntrl[10] = 1;                                   // unit cell record present
    icntrl[19] = 24;                                  // VERSION (charmm format)

    dcd_nset_patch_pos_ = dcd_file_.tellp();  // icntrl[0] == NSET, patched at close
    dcd_file_.write(reinterpret_cast<const char*>(icntrl), sizeof(icntrl));
    dcd_file_.write(reinterpret_cast<const char*>(&reclen1), 4);

    std::string title = "Created by DeepScaleMD";
    title.resize(80, ' ');
    const int32_t ntitle = 1;
    const int32_t title_reclen = static_cast<int32_t>(sizeof(int32_t) + 80);
    dcd_file_.write(reinterpret_cast<const char*>(&title_reclen), 4);
    dcd_file_.write(reinterpret_cast<const char*>(&ntitle), 4);
    dcd_file_.write(title.data(), 80);
    dcd_file_.write(reinterpret_cast<const char*>(&title_reclen), 4);

    dcd_write_record(dcd_file_, reinterpret_cast<const char*>(&natom_global),
                      static_cast<int32_t>(sizeof(int32_t)));
}

void MDSimulation::write_dcd_frame(int step, const SerialTrajectoryFrame& frame) {
    (void)step;
    if (!dcd_file_.is_open()) return;

    const SimulationBox& box = system->get_box();
    double cell[6] = {box.lx, 0.0, box.ly, 0.0, 0.0, box.lz};
    dcd_write_record(dcd_file_, reinterpret_cast<const char*>(cell), static_cast<int32_t>(sizeof(cell)));

    const int32_t n = static_cast<int32_t>(frame.x.size());
    std::vector<float> buf(n);

    for (int32_t i = 0; i < n; ++i) buf[i] = static_cast<float>(frame.x[i]);
    dcd_write_record(dcd_file_, reinterpret_cast<const char*>(buf.data()), n * static_cast<int32_t>(sizeof(float)));
    for (int32_t i = 0; i < n; ++i) buf[i] = static_cast<float>(frame.y[i]);
    dcd_write_record(dcd_file_, reinterpret_cast<const char*>(buf.data()), n * static_cast<int32_t>(sizeof(float)));
    for (int32_t i = 0; i < n; ++i) buf[i] = static_cast<float>(frame.z[i]);
    dcd_write_record(dcd_file_, reinterpret_cast<const char*>(buf.data()), n * static_cast<int32_t>(sizeof(float)));

    ++dcd_frames_written_;
}

void MDSimulation::initialize_trajectory_writers() {
    if (trajectory_formats_.empty()) return;

    if (trajectory_formats_.count("dcd") || trajectory_formats_.count("bp")
        || trajectory_formats_.count("h5md")) {
        int local_n = static_cast<int>(system->get_local_particles());
        int global_n = 0;
        MPI_Allreduce(&local_n, &global_n, 1, MPI_INT, MPI_SUM, PmePpSplit::instance().pp_comm());
        trajectory_natoms_global_cached_ = global_n;
    }

    if (trajectory_formats_.count("dcd") && system->get_rank() == 0) {
        const std::string filename = output_prefix + ".dcd";
        dcd_file_.open(filename, std::ios::binary | std::ios::trunc);
        if (!dcd_file_.is_open()) {
            throw std::runtime_error("Cannot open DCD trajectory file: " + filename);
        }
        write_dcd_header(trajectory_natoms_global_cached_);
        dcd_frames_written_ = 0;
    }

#ifdef USE_ADIOS2
    if (trajectory_formats_.count("bp")) {
        initialize_adios2_writer();
    }
#endif

#ifdef USE_HDF5
    if (trajectory_formats_.count("h5md")) {
        initialize_h5md_writer();
    }
#endif

    trajectory_writers_initialized_ = true;
}

void MDSimulation::finalize_trajectory_writers() {
    if (!trajectory_writers_initialized_) return;

    if (dcd_file_.is_open()) {
        dcd_file_.seekp(dcd_nset_patch_pos_);
        const int32_t nset = dcd_frames_written_;
        dcd_file_.write(reinterpret_cast<const char*>(&nset), sizeof(nset));
        dcd_file_.close();
    }

#ifdef USE_ADIOS2
    if (trajectory_formats_.count("bp")) {
        finalize_adios2_writer();
    }
#endif

#ifdef USE_HDF5
    if (trajectory_formats_.count("h5md")) {
        finalize_h5md_writer();
    }
#endif
}

#ifdef USE_ADIOS2
void MDSimulation::initialize_adios2_writer() {
    adios2_instance_ = adios2::ADIOS(PmePpSplit::instance().pp_comm());
    adios2_io_ = adios2_instance_.DeclareIO("DeepScaleMD_Trajectory");
    adios2_io_.SetEngine("BP5");

    const size_t local_n = system->get_local_particles();

    adios2_var_global_id_ = adios2_io_.DefineVariable<int>("global_id", {}, {}, {local_n});
    adios2_var_x_ = adios2_io_.DefineVariable<Real>("x", {}, {}, {local_n});
    adios2_var_y_ = adios2_io_.DefineVariable<Real>("y", {}, {}, {local_n});
    adios2_var_z_ = adios2_io_.DefineVariable<Real>("z", {}, {}, {local_n});
    if (trajectory_include_velocities) {
        adios2_var_vx_ = adios2_io_.DefineVariable<Real>("vx", {}, {}, {local_n});
        adios2_var_vy_ = adios2_io_.DefineVariable<Real>("vy", {}, {}, {local_n});
        adios2_var_vz_ = adios2_io_.DefineVariable<Real>("vz", {}, {}, {local_n});
    }
    if (trajectory_include_forces) {
        adios2_var_fx_ = adios2_io_.DefineVariable<Real>("fx", {}, {}, {local_n});
        adios2_var_fy_ = adios2_io_.DefineVariable<Real>("fy", {}, {}, {local_n});
        adios2_var_fz_ = adios2_io_.DefineVariable<Real>("fz", {}, {}, {local_n});
    }

    adios2_var_step_ = adios2_io_.DefineVariable<int>("step");
    adios2_var_time_ = adios2_io_.DefineVariable<Real>("time_fs");
    adios2_var_box_ = adios2_io_.DefineVariable<Real>("box", {}, {}, {3});

    adios2_io_.DefineAttribute<int>("natoms_global", trajectory_natoms_global_cached_);

    adios2_engine_ = adios2_io_.Open(output_prefix + ".bp", adios2::Mode::Write);
    adios2_active_ = true;
}

void MDSimulation::write_adios2_frame(int step) {
    const size_t local_n = system->get_local_particles();
    ParticleData& particles = system->get_particles();

    adios2_engine_.BeginStep();

    adios2_var_global_id_.SetSelection({{}, {local_n}});
    adios2_engine_.Put(adios2_var_global_id_, particles.global_id.data());

    adios2_var_x_.SetSelection({{}, {local_n}});
    adios2_engine_.Put(adios2_var_x_, particles.x.data());
    adios2_var_y_.SetSelection({{}, {local_n}});
    adios2_engine_.Put(adios2_var_y_, particles.y.data());
    adios2_var_z_.SetSelection({{}, {local_n}});
    adios2_engine_.Put(adios2_var_z_, particles.z.data());

    if (trajectory_include_velocities) {
        adios2_var_vx_.SetSelection({{}, {local_n}});
        adios2_engine_.Put(adios2_var_vx_, particles.vx.data());
        adios2_var_vy_.SetSelection({{}, {local_n}});
        adios2_engine_.Put(adios2_var_vy_, particles.vy.data());
        adios2_var_vz_.SetSelection({{}, {local_n}});
        adios2_engine_.Put(adios2_var_vz_, particles.vz.data());
    }
    if (trajectory_include_forces) {
        adios2_var_fx_.SetSelection({{}, {local_n}});
        adios2_engine_.Put(adios2_var_fx_, particles.fx.data());
        adios2_var_fy_.SetSelection({{}, {local_n}});
        adios2_engine_.Put(adios2_var_fy_, particles.fy.data());
        adios2_var_fz_.SetSelection({{}, {local_n}});
        adios2_engine_.Put(adios2_var_fz_, particles.fz.data());
    }

    adios2_engine_.Put(adios2_var_step_, step);
    adios2_engine_.Put(adios2_var_time_, static_cast<Real>(step) * dt);

    std::array<Real, 3> box_vals = {system->get_box().lx, system->get_box().ly, system->get_box().lz};
    adios2_var_box_.SetSelection({{}, {3}});
    adios2_engine_.Put(adios2_var_box_, box_vals.data());

    adios2_engine_.EndStep();
}

void MDSimulation::finalize_adios2_writer() {
    if (adios2_active_) {
        adios2_engine_.Close();
        adios2_active_ = false;
    }
}
#endif // USE_ADIOS2

#ifdef USE_HDF5

namespace {

hid_t h5md_create_extendible_dataset(hid_t loc, const char* name, hid_t type,
                                      int rank, const hsize_t* dims, const hsize_t* chunk_dims) {
    std::vector<hsize_t> maxdims(dims, dims + rank);
    maxdims[0] = H5S_UNLIMITED;
    hid_t space = H5Screate_simple(rank, dims, maxdims.data());
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(dcpl, rank, chunk_dims);
    hid_t dset = H5Dcreate2(loc, name, type, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Pclose(dcpl);
    H5Sclose(space);
    return dset;
}

void h5md_write_string_attr(hid_t loc, const char* name, const std::string& value) {
    hid_t strtype = H5Tcopy(H5T_C_S1);
    H5Tset_size(strtype, value.size() + 1);
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(loc, name, strtype, space, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(attr, strtype, value.c_str());
    H5Aclose(attr);
    H5Sclose(space);
    H5Tclose(strtype);
}

void h5md_extend(hid_t dset, std::initializer_list<hsize_t> dims) {
    std::vector<hsize_t> d(dims);
    H5Dset_extent(dset, d.data());
}

} // namespace

void MDSimulation::initialize_h5md_writer() {
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(fapl, PmePpSplit::instance().pp_comm(), MPI_INFO_NULL);
    const std::string filename = output_prefix + ".h5";
    h5md_file_ = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    H5Pclose(fapl);
    if (h5md_file_ < 0) {
        throw std::runtime_error("Cannot open H5MD trajectory file: " + filename);
    }

    hid_t h5md_group = H5Gcreate2(h5md_file_, "/h5md", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    {
        int version[2] = {1, 1};
        hsize_t dims2[1] = {2};
        hid_t space = H5Screate_simple(1, dims2, nullptr);
        hid_t attr = H5Acreate2(h5md_group, "version", H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, H5T_NATIVE_INT, version);
        H5Aclose(attr);
        H5Sclose(space);
    }
    hid_t creator_group = H5Gcreate2(h5md_group, "creator", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    h5md_write_string_attr(creator_group, "name", "DeepScaleMD");
    H5Gclose(creator_group);
    hid_t author_group = H5Gcreate2(h5md_group, "author", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    const char* user_env = std::getenv("USER");
    if (!user_env) user_env = std::getenv("USERNAME");
    h5md_write_string_attr(author_group, "name", user_env ? user_env : "unknown");
    H5Gclose(author_group);
    H5Gclose(h5md_group);

    hid_t particles_group = H5Gcreate2(h5md_file_, "/particles", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t all_group = H5Gcreate2(particles_group, "all", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    hid_t box_group = H5Gcreate2(all_group, "box", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    {
        int dim = 3;
        hid_t space = H5Screate(H5S_SCALAR);
        hid_t attr = H5Acreate2(box_group, "dimension", H5T_NATIVE_INT, space, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, H5T_NATIVE_INT, &dim);
        H5Aclose(attr);
        H5Sclose(space);
    }
    {
        hid_t strtype = H5Tcopy(H5T_C_S1);
        H5Tset_size(strtype, 9);  // "periodic" + NUL
        hsize_t dims3[1] = {3};
        hid_t space = H5Screate_simple(1, dims3, nullptr);
        hid_t attr = H5Acreate2(box_group, "boundary", strtype, space, H5P_DEFAULT, H5P_DEFAULT);
        char boundary[3][9] = {"periodic", "periodic", "periodic"};
        H5Awrite(attr, strtype, boundary);
        H5Aclose(attr);
        H5Sclose(space);
        H5Tclose(strtype);
    }
    {
        hid_t edges_group = H5Gcreate2(box_group, "edges", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hsize_t dims[2] = {0, 3};
        hsize_t chunk[2] = {1, 3};
        h5md_box_edges_ = h5md_create_extendible_dataset(edges_group, "value", H5T_NATIVE_DOUBLE, 2, dims, chunk);
        H5Gclose(edges_group);
    }
    H5Gclose(box_group);

    const hsize_t n_global = static_cast<hsize_t>(trajectory_natoms_global_cached_);

    {
        hid_t id_group = H5Gcreate2(all_group, "id", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hsize_t dims[2] = {0, n_global};
        hsize_t chunk[2] = {1, n_global};
        h5md_id_value_ = h5md_create_extendible_dataset(id_group, "value", H5T_NATIVE_INT, 2, dims, chunk);
        H5Gclose(id_group);
    }
    {
        hid_t species_group = H5Gcreate2(all_group, "species", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hsize_t dims[2] = {0, n_global};
        hsize_t chunk[2] = {1, n_global};
        h5md_species_value_ = h5md_create_extendible_dataset(species_group, "value", H5T_NATIVE_INT, 2, dims, chunk);
        H5Gclose(species_group);
    }
    {
        hid_t position_group = H5Gcreate2(all_group, "position", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hsize_t dims1[1] = {0};
        hsize_t chunk1[1] = {1};
        h5md_position_step_ = h5md_create_extendible_dataset(position_group, "step", H5T_NATIVE_INT, 1, dims1, chunk1);
        h5md_position_time_ = h5md_create_extendible_dataset(position_group, "time", H5T_NATIVE_DOUBLE, 1, dims1, chunk1);
        hsize_t dims3[3] = {0, n_global, 3};
        hsize_t chunk3[3] = {1, n_global, 3};
        h5md_position_value_ = h5md_create_extendible_dataset(position_group, "value", H5T_NATIVE_DOUBLE, 3, dims3, chunk3);
        H5Gclose(position_group);
    }
    if (trajectory_include_velocities) {
        hid_t velocity_group = H5Gcreate2(all_group, "velocity", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hsize_t dims1[1] = {0};
        hsize_t chunk1[1] = {1};
        h5md_velocity_step_ = h5md_create_extendible_dataset(velocity_group, "step", H5T_NATIVE_INT, 1, dims1, chunk1);
        h5md_velocity_time_ = h5md_create_extendible_dataset(velocity_group, "time", H5T_NATIVE_DOUBLE, 1, dims1, chunk1);
        hsize_t dims3[3] = {0, n_global, 3};
        hsize_t chunk3[3] = {1, n_global, 3};
        h5md_velocity_value_ = h5md_create_extendible_dataset(velocity_group, "value", H5T_NATIVE_DOUBLE, 3, dims3, chunk3);
        H5Gclose(velocity_group);
    }
    if (trajectory_include_forces) {
        hid_t force_group = H5Gcreate2(all_group, "force", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hsize_t dims1[1] = {0};
        hsize_t chunk1[1] = {1};
        h5md_force_step_ = h5md_create_extendible_dataset(force_group, "step", H5T_NATIVE_INT, 1, dims1, chunk1);
        h5md_force_time_ = h5md_create_extendible_dataset(force_group, "time", H5T_NATIVE_DOUBLE, 1, dims1, chunk1);
        hsize_t dims3[3] = {0, n_global, 3};
        hsize_t chunk3[3] = {1, n_global, 3};
        h5md_force_value_ = h5md_create_extendible_dataset(force_group, "value", H5T_NATIVE_DOUBLE, 3, dims3, chunk3);
        H5Gclose(force_group);
    }

    H5Gclose(all_group);
    H5Gclose(particles_group);

    h5md_frame_count_ = 0;
    h5md_active_ = true;
}

void MDSimulation::write_h5md_frame(int step) {
    if (!h5md_active_) return;

    int local_n = static_cast<int>(system->get_local_particles());
    ParticleData& particles = system->get_particles();
    const hsize_t n_global = static_cast<hsize_t>(trajectory_natoms_global_cached_);
    const hsize_t frame = static_cast<hsize_t>(h5md_frame_count_);
    const int rank = system->get_rank();

    int local_offset_int = 0;
    MPI_Exscan(&local_n, &local_offset_int, 1, MPI_INT, MPI_SUM, PmePpSplit::instance().pp_comm());
    if (rank == 0) local_offset_int = 0;
    const hsize_t local_offset = static_cast<hsize_t>(local_offset_int);
    const hsize_t local_count = static_cast<hsize_t>(local_n);

    h5md_extend(h5md_position_step_, {frame + 1});
    h5md_extend(h5md_position_time_, {frame + 1});
    h5md_extend(h5md_position_value_, {frame + 1, n_global, 3});
    h5md_extend(h5md_id_value_, {frame + 1, n_global});
    h5md_extend(h5md_species_value_, {frame + 1, n_global});
    h5md_extend(h5md_box_edges_, {frame + 1, 3});
    if (trajectory_include_velocities) {
        h5md_extend(h5md_velocity_step_, {frame + 1});
        h5md_extend(h5md_velocity_time_, {frame + 1});
        h5md_extend(h5md_velocity_value_, {frame + 1, n_global, 3});
    }
    if (trajectory_include_forces) {
        h5md_extend(h5md_force_step_, {frame + 1});
        h5md_extend(h5md_force_time_, {frame + 1});
        h5md_extend(h5md_force_value_, {frame + 1, n_global, 3});
    }

    hid_t dxpl_collective = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(dxpl_collective, H5FD_MPIO_COLLECTIVE);

    auto write_vec3 = [&](hid_t dset, const std::vector<Real>& buf) {
        hid_t filespace = H5Dget_space(dset);
        if (local_count > 0) {
            hsize_t offset[3] = {frame, local_offset, 0};
            hsize_t count[3] = {1, local_count, 3};
            H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, nullptr, count, nullptr);
        } else {
            H5Sselect_none(filespace);
        }
        hsize_t mem_dims[3] = {1, local_count, 3};
        hid_t memspace = H5Screate_simple(3, mem_dims, nullptr);
        if (local_count == 0) H5Sselect_none(memspace);
        H5Dwrite(dset, H5T_NATIVE_DOUBLE, memspace, filespace, dxpl_collective, buf.data());
        H5Sclose(memspace);
        H5Sclose(filespace);
    };
    auto write_ids = [&](hid_t dset, const int* data) {
        hid_t filespace = H5Dget_space(dset);
        if (local_count > 0) {
            hsize_t offset[2] = {frame, local_offset};
            hsize_t count[2] = {1, local_count};
            H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, nullptr, count, nullptr);
        } else {
            H5Sselect_none(filespace);
        }
        hsize_t mem_dims[2] = {1, local_count};
        hid_t memspace = H5Screate_simple(2, mem_dims, nullptr);
        if (local_count == 0) H5Sselect_none(memspace);
        H5Dwrite(dset, H5T_NATIVE_INT, memspace, filespace, dxpl_collective, data);
        H5Sclose(memspace);
        H5Sclose(filespace);
    };
    auto pack_xyz = [&](const Real* a, const Real* b, const Real* c) {
        std::vector<Real> buf(static_cast<size_t>(local_n) * 3);
        for (int i = 0; i < local_n; ++i) {
            buf[3 * i + 0] = a[i]; buf[3 * i + 1] = b[i]; buf[3 * i + 2] = c[i];
        }
        return buf;
    };

    write_vec3(h5md_position_value_, pack_xyz(particles.x.data(), particles.y.data(), particles.z.data()));
    write_ids(h5md_id_value_, particles.global_id.data());
    write_ids(h5md_species_value_, particles.type.data());
    if (trajectory_include_velocities) {
        write_vec3(h5md_velocity_value_, pack_xyz(particles.vx.data(), particles.vy.data(), particles.vz.data()));
    }
    if (trajectory_include_forces) {
        write_vec3(h5md_force_value_, pack_xyz(particles.fx.data(), particles.fy.data(), particles.fz.data()));
    }

    H5Pclose(dxpl_collective);

    if (rank == 0) {
        auto write_scalar = [&](hid_t dset, hid_t type, const void* value) {
            hid_t filespace = H5Dget_space(dset);
            hsize_t offset[1] = {frame};
            hsize_t count[1] = {1};
            H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, nullptr, count, nullptr);
            hid_t memspace = H5Screate_simple(1, count, nullptr);
            H5Dwrite(dset, type, memspace, filespace, H5P_DEFAULT, value);
            H5Sclose(memspace);
            H5Sclose(filespace);
        };
        write_scalar(h5md_position_step_, H5T_NATIVE_INT, &step);
        const Real time_fs = static_cast<Real>(step) * dt;
        write_scalar(h5md_position_time_, H5T_NATIVE_DOUBLE, &time_fs);
        if (trajectory_include_velocities) {
            write_scalar(h5md_velocity_step_, H5T_NATIVE_INT, &step);
            write_scalar(h5md_velocity_time_, H5T_NATIVE_DOUBLE, &time_fs);
        }
        if (trajectory_include_forces) {
            write_scalar(h5md_force_step_, H5T_NATIVE_INT, &step);
            write_scalar(h5md_force_time_, H5T_NATIVE_DOUBLE, &time_fs);
        }

        Real box_vals[3] = {system->get_box().lx, system->get_box().ly, system->get_box().lz};
        hid_t filespace = H5Dget_space(h5md_box_edges_);
        hsize_t offset2[2] = {frame, 0};
        hsize_t count2[2] = {1, 3};
        H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset2, nullptr, count2, nullptr);
        hid_t memspace = H5Screate_simple(2, count2, nullptr);
        H5Dwrite(h5md_box_edges_, H5T_NATIVE_DOUBLE, memspace, filespace, H5P_DEFAULT, box_vals);
        H5Sclose(memspace);
        H5Sclose(filespace);
    }

    ++h5md_frame_count_;
}

void MDSimulation::finalize_h5md_writer() {
    if (!h5md_active_) return;

    H5Dclose(h5md_position_step_);
    H5Dclose(h5md_position_time_);
    H5Dclose(h5md_position_value_);
    H5Dclose(h5md_id_value_);
    H5Dclose(h5md_species_value_);
    H5Dclose(h5md_box_edges_);
    if (trajectory_include_velocities) {
        H5Dclose(h5md_velocity_step_);
        H5Dclose(h5md_velocity_time_);
        H5Dclose(h5md_velocity_value_);
    }
    if (trajectory_include_forces) {
        H5Dclose(h5md_force_step_);
        H5Dclose(h5md_force_time_);
        H5Dclose(h5md_force_value_);
    }
    H5Fclose(h5md_file_);
    h5md_active_ = false;
}
#endif // USE_HDF5
