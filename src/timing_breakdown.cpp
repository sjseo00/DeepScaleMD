// LAMMPS-style timing breakdown implementation
// This file contains the print_timing_breakdown() function

#include "md_simulation.h"
#include <iostream>
#include <iomanip>
#include <array>
#include <mpi.h>

void MDSimulation::print_timing_breakdown() {
#ifdef USE_MPI
    int mpi_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    constexpr int kTimingBuckets = 10;
    // Gather timing statistics from all ranks
    double local_times[kTimingBuckets] = {
        cumulative_timing.neighbor_time,
        cumulative_timing.force_time,
        cumulative_timing.integration_time,
        cumulative_timing.thermostat_time,
        cumulative_timing.barostat_time,
        cumulative_timing.ke_time,
        cumulative_timing.comm_time,
        cumulative_timing.migration_time,
        cumulative_timing.output_time,
        cumulative_timing.total_time
    };

    std::array<double, kTimingBuckets> min_times, max_times, sum_times;

    MPI_Reduce(local_times, min_times.data(), kTimingBuckets, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_times, max_times.data(), kTimingBuckets, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_times, sum_times.data(), kTimingBuckets, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (system->get_rank() == 0) {
        std::array<double, kTimingBuckets> avg_times;
        for (int i = 0; i < kTimingBuckets; i++) {
            avg_times[i] = sum_times[i] / mpi_size;
        }

        // Use reduced average total time so section percentages sum to 100%
        double total_avg = avg_times[9];
        if (total_avg <= 0.0) {
            total_avg = 1.0;
        }

        std::cout << "\n";
        std::cout << "Loop time of " << std::fixed << std::setprecision(5) 
                  << avg_times[9] << " on " << mpi_size
                  << " procs for " << cumulative_timing.step_count << " steps\n";
        std::cout << "\n";
        std::cout << "MPI task timing breakdown:\n";
        std::cout << "Section |  min time  |  avg time  |  max time  |%varavg| %total\n";
        std::cout << "---------------------------------------------------------------\n";
        
        auto print_section = [&](const char* name, int idx) {
            double pct = (avg_times[idx] / total_avg) * 100.0;
            double variance = 0.0;
            if (avg_times[idx] > 1e-10) {
                variance = ((max_times[idx] - min_times[idx]) / avg_times[idx]) * 100.0;
            }
            
            std::cout << std::left << std::setw(8) << name << "| "
                      << std::right << std::fixed << std::setprecision(4) << std::setw(10) << min_times[idx] << " | "
                      << std::setw(10) << avg_times[idx] << " | "
                      << std::setw(10) << max_times[idx] << " | "
                      << std::setw(5) << std::setprecision(1) << variance << " | "
                      << std::setw(6) << std::setprecision(2) << pct << "\n";
        };

        print_section("Neighbor", 0);
        print_section("Force", 1);
        print_section("Integrate", 2);
        print_section("Thermo", 3);
        print_section("Barostat", 4);
        print_section("KE", 5);
        print_section("Comm", 6);
        print_section("Migrate", 7);
        print_section("Output", 8);

        std::cout << "\n";

        // Performance metrics
        double nsteps = cumulative_timing.step_count;
        double natoms = system->get_num_particles();
        double total_time = avg_times[9];
        double time_per_step = (nsteps > 0.0) ? (total_time / nsteps) : 0.0;
        double atom_steps_per_sec = (total_time > 0.0) ? (natoms * nsteps / total_time) : 0.0;

        std::cout << "Performance: " << std::fixed << std::setprecision(3)
                  << atom_steps_per_sec / 1e6 << " Matom-step/s, "
                  << time_per_step * 1e3 << " ms/step\n";
        std::cout << "\n";
    }
#else
    // Single processor version
    double raw_total = cumulative_timing.total_time;
    double total = (raw_total > 0.0) ? raw_total : 1.0;
    
    std::cout << "\n";
    std::cout << "Loop time of " << std::fixed << std::setprecision(5) 
              << raw_total << " on 1 procs for " << cumulative_timing.step_count << " steps\n";
    std::cout << "\n";
    std::cout << "Section timing breakdown:\n";
    std::cout << "Section |    time    | %total\n";
    std::cout << "--------------------------------\n";
    
    auto print_section = [&](const char* name, double time) {
        double pct = (time / total) * 100.0;
        std::cout << std::left << std::setw(8) << name << "| "
                  << std::right << std::fixed << std::setprecision(4) << std::setw(10) << time << " | "
                  << std::setw(6) << std::setprecision(2) << pct << "\n";
    };
    
    print_section("Neighbor", cumulative_timing.neighbor_time);
    print_section("Force", cumulative_timing.force_time);
    print_section("Integrate", cumulative_timing.integration_time);
    print_section("Thermo", cumulative_timing.thermostat_time);
    print_section("Barostat", cumulative_timing.barostat_time);
    print_section("KE", cumulative_timing.ke_time);
    print_section("Comm", cumulative_timing.comm_time);
    print_section("Migrate", cumulative_timing.migration_time);
    print_section("Output", cumulative_timing.output_time);
    
    std::cout << "\n";
    
    // Performance metrics
    double nsteps = cumulative_timing.step_count;
    double natoms = system->get_num_particles();
    
    std::cout << "Neighbor list builds = " << neighbor_list_builds << "\n";
    std::cout << "Performance: " << std::fixed << std::setprecision(3) 
              << ((total > 0.0) ? (nsteps * natoms / total / 1e6) : 0.0) << " million atom-steps/sec\n";
    std::cout << "\n";
#endif
}
