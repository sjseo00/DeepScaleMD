// LAMMPS-style timing breakdown implementation
// This file contains the print_timing_breakdown() function

#include "md_simulation.h"
#include <iostream>
#include <iomanip>
#include <mpi.h>

void MDSimulation::print_timing_breakdown() {
#ifdef USE_MPI
    int mpi_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    
    // Gather timing statistics from all ranks
    double local_times[6] = {
        cumulative_timing.pair_time,
        cumulative_timing.neigh_time,
        cumulative_timing.comm_time,
        cumulative_timing.output_time,
        cumulative_timing.modify_time,
        cumulative_timing.other_time
    };
    
    double min_times[6], max_times[6], sum_times[6];
    
    MPI_Reduce(local_times, min_times, 6, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_times, max_times, 6, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_times, sum_times, 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (system->get_rank() == 0) {
        double avg_times[6];
        for (int i = 0; i < 6; i++) {
            avg_times[i] = sum_times[i] / mpi_size;
        }
        
        double total_avg = cumulative_timing.total_time / mpi_size;
        
        std::cout << "\n";
        std::cout << "Loop time of " << std::fixed << std::setprecision(5) 
                  << cumulative_timing.total_time << " on " << mpi_size 
                  << " procs for " << cumulative_timing.step_count << " steps\n";
        std::cout << "\n";
        std::cout << "MPI task timing breakdown:\n";
        std::cout << "Section |  min time  |  avg time  |  max time  |%varavg| %total\n";
        std::cout << "---------------------------------------------------------------\n";
        
        auto print_section = [&](const char* name, int idx) {
            double pct = (avg_times[idx] / cumulative_timing.total_time) * 100.0;
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
        
        print_section("Pair", 0);
        print_section("Neigh", 1);
        print_section("Comm", 2);
        print_section("Output", 3);
        print_section("Modify", 4);
        
        // Calculate "Other" time
        double other_time = cumulative_timing.total_time - 
                           (avg_times[0] + avg_times[1] + avg_times[2] + avg_times[3] + avg_times[4]);
        double other_pct = (other_time / cumulative_timing.total_time) * 100.0;
        
        std::cout << std::left << std::setw(8) << "Other" << "|            | "
                  << std::right << std::fixed << std::setprecision(4) << std::setw(10) << other_time << " |"
                  << "            |       | "
                  << std::setw(6) << std::setprecision(2) << other_pct << "\n";
        
        std::cout << "\n";
        
        // Performance metrics
        double nsteps = cumulative_timing.step_count;
        double natoms = system->get_num_particles();
        double time_per_step = cumulative_timing.total_time / nsteps;
        double atom_steps_per_sec = natoms * nsteps / cumulative_timing.total_time;
        
        std::cout << "Nlocal:    " << std::fixed << std::setprecision(1) << natoms 
                  << " ave " << natoms << " max " << natoms << " min\n";
        std::cout << "Histogram: " << mpi_size << " 0 0 0 0 0 0 0 0 0\n";
        std::cout << "Nghost:    0 ave 0 max 0 min\n";
        std::cout << "Histogram: " << mpi_size << " 0 0 0 0 0 0 0 0 0\n";
        std::cout << "Neighs:    0 ave 0 max 0 min\n";
        std::cout << "Histogram: " << mpi_size << " 0 0 0 0 0 0 0 0 0\n";
        std::cout << "\n";
        std::cout << "Total # of neighbors = " << 0 << "\n";
        std::cout << "Ave neighs/atom = " << std::setprecision(2) << 0.0 << "\n";
        std::cout << "Neighbor list builds = " << neighbor_list_builds << "\n";
        std::cout << "Dangerous builds = 0\n";
        std::cout << "\n";
    }
#else
    // Single processor version
    double total = cumulative_timing.total_time;
    
    std::cout << "\n";
    std::cout << "Loop time of " << std::fixed << std::setprecision(5) 
              << total << " on 1 procs for " << cumulative_timing.step_count << " steps\n";
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
    
    print_section("Pair", cumulative_timing.pair_time);
    print_section("Neigh", cumulative_timing.neigh_time);
    print_section("Comm", cumulative_timing.comm_time);
    print_section("Output", cumulative_timing.output_time);
    print_section("Modify", cumulative_timing.modify_time);
    
    double other_time = total - (cumulative_timing.pair_time + cumulative_timing.neigh_time + 
                                 cumulative_timing.comm_time + cumulative_timing.output_time + 
                                 cumulative_timing.modify_time);
    print_section("Other", other_time);
    
    std::cout << "\n";
    
    // Performance metrics
    double nsteps = cumulative_timing.step_count;
    double natoms = system->get_num_particles();
    
    std::cout << "Neighbor list builds = " << neighbor_list_builds << "\n";
    std::cout << "Performance: " << std::fixed << std::setprecision(3) 
              << (nsteps * natoms / total / 1e6) << " million atom-steps/sec\n";
    std::cout << "\n";
#endif
}
