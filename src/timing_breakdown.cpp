// End-of-run timing report: a small fixed tree (init / minimize / md_loop ->
// its per-phase buckets -> optional per-force-component sub-buckets),
// printed once. Percentages are relative to each row's immediate parent
// (top-level rows against the grand total, md_loop's children against
// md_loop, force's children against force), not a single flat denominator.
//
// USE_MPI is unconditionally defined for every build of this target (see
// CMakeLists.txt), so there is no meaningful single-process code path to
// special-case; MPI_COMM_WORLD with size 1 already covers that. The extra
// [min/max/var] columns are only printed when mpi_size > 1, so a single-GPU
// run (the common case) gets one clean column instead of three identical
// ones plus a permanently-zero variance figure.
#include "md_simulation.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <mpi.h>

namespace {

struct TimingRow {
    std::string label;
    int indent;       // 0, 1, or 2
    double seconds;   // this rank's local accumulated time
};

void print_row(const TimingRow& row, double denom_seconds, int mpi_size,
               double min_v, double avg_v, double max_v) {
    std::string label(static_cast<size_t>(row.indent) * 2, ' ');
    label += row.label;

    std::cout << std::left << std::setw(26) << label << std::right
              << std::fixed << std::setprecision(4) << std::setw(11) << avg_v << " s";

    double pct = (denom_seconds > 1e-12) ? (avg_v / denom_seconds * 100.0) : 0.0;
    std::cout << "  (" << std::setprecision(1) << std::setw(5) << pct << "%)\n";
}

}  // namespace

void MDSimulation::print_timing_report(double program_wallclock_ms) const {
    int mpi_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    const int rank = system->get_rank();

    // Fixed layout. Indices of the three top-level rows and "force" are
    // relied on below (they're always present, regardless of how many
    // force sub-components get appended between "force" and "comm").
    std::vector<TimingRow> rows;
    const int idx_init     = static_cast<int>(rows.size());
    rows.push_back({"init", 0, cumulative_timing.init_time});
    const int idx_minimize  = static_cast<int>(rows.size());
    rows.push_back({"minimize", 0, cumulative_timing.minimize_time});
    const int idx_md_loop   = static_cast<int>(rows.size());
    rows.push_back({"md_loop", 0, cumulative_timing.total_time});
    rows.push_back({"neighbor", 1, cumulative_timing.neighbor_time});
    rows.push_back({"integrate", 1, cumulative_timing.integration_time});
    const int idx_force     = static_cast<int>(rows.size());
    rows.push_back({"force", 1, cumulative_timing.force_time});
    // Only present when MD_TIMING_TRACE_FORCE=1 was set for this run; empty
    // otherwise, so these rows just don't appear.
    for (const auto& component : cumulative_timing.force_component_time) {
        rows.push_back({component.first, 2, component.second});
    }
    rows.push_back({"comm", 1, cumulative_timing.comm_time});
    rows.push_back({"migrate", 1, cumulative_timing.migration_time});
    rows.push_back({"thermostat", 1, cumulative_timing.thermostat_time});
    rows.push_back({"barostat", 1, cumulative_timing.barostat_time});
    rows.push_back({"ke", 1, cumulative_timing.ke_time});
    rows.push_back({"output", 1, cumulative_timing.output_time});

    const int n = static_cast<int>(rows.size());
    std::vector<double> local(n), min_v(n), max_v(n), sum_v(n);
    for (int i = 0; i < n; ++i) local[i] = rows[i].seconds;

    MPI_Reduce(local.data(), min_v.data(), n, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(local.data(), max_v.data(), n, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(local.data(), sum_v.data(), n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        return;
    }

    std::vector<double> avg_v(n);
    for (int i = 0; i < n; ++i) avg_v[i] = sum_v[i] / mpi_size;

    const double grand_total_avg = avg_v[idx_init] + avg_v[idx_minimize] + avg_v[idx_md_loop];
    const double md_loop_avg = avg_v[idx_md_loop];
    const double force_avg = avg_v[idx_force];

    std::cout << "\n=== DeepScaleMD Timing Report";
    if (mpi_size > 1) std::cout << " (" << mpi_size << " ranks)";
    std::cout << " ===\n";

    for (int i = 0; i < n; ++i) {
        double denom;
        if (rows[i].indent == 0) denom = grand_total_avg;
        else if (rows[i].indent == 1) denom = md_loop_avg;
        else denom = force_avg;
        print_row(rows[i], denom, mpi_size, min_v[i], avg_v[i], max_v[i]);
    }

    std::cout << "------------------------------------------------------------\n";
    std::cout << std::left << std::setw(26) << "total (init+minimize+loop)" << std::right
               << std::fixed << std::setprecision(4) << std::setw(11) << grand_total_avg << " s\n";

    // program_wallclock_ms comes from main.cpp's own chrono span around the
    // whole run (the only place that can see time spent before this object
    // existed, and after finalize() starts). Comparing it against the sum of
    // the buckets above surfaces exactly the kind of "where did the other
    // N seconds go" gap that used to require manual arithmetic across three
    // separately-printed numbers.
    if (program_wallclock_ms >= 0.0) {
        const double program_total_s = program_wallclock_ms / 1000.0;
        std::cout << std::left << std::setw(26) << "total (wall clock)" << std::right
                   << std::setw(11) << program_total_s << " s\n";
        const double unaccounted = program_total_s - grand_total_avg;
        if (unaccounted > 0.01) {
            std::cout << std::left << std::setw(26) << "  unaccounted" << std::right
                       << std::setw(11) << unaccounted << " s\n";
        }
    }

    std::cout << "\nNeighbor list builds = " << neighbor_list_builds << "\n";

    const double nsteps = cumulative_timing.step_count;
    const double natoms = system->get_num_particles();
    const double time_per_step = (nsteps > 0.0) ? (md_loop_avg / nsteps) : 0.0;
    const double atom_steps_per_sec = (md_loop_avg > 0.0) ? (natoms * nsteps / md_loop_avg) : 0.0;
    // ns/day, the standard MD throughput figure (NAMD/GROMACS/LAMMPS all
    // report it): how much simulated time (at this run's timestep, dt in
    // fs) a full day of wall clock buys, extrapolated from the measured
    // md_loop steady-state rate. Comparable across codes/systems/timesteps
    // in a way ms/step alone isn't.
    const double ns_per_day = (time_per_step > 0.0)
        ? (dt * 1e-6) * (86400.0 / time_per_step)
        : 0.0;
    std::cout << "Performance: " << std::fixed << std::setprecision(3)
              << atom_steps_per_sec / 1e6 << " Matom-step/s, "
              << time_per_step * 1e3 << " ms/step, "
              << ns_per_day << " ns/day (md_loop only, dt=" << dt << " fs)\n\n";
}
