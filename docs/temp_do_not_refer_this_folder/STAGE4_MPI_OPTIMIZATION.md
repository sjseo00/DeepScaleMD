# STAGE 4: MPI Performance Optimization

## Date: 2024-12-18

## Current Status: IN PROGRESS

---

## 1. Initial Analysis

### MPI Communication Hotspots Identified

```cpp
// src/md_simulation.cpp - Current MPI_Allreduce calls:

1. Line 267: Displacement check (every neighbor_list_update_freq steps)
   MPI_Allreduce(&max_disp_sq, &global_max_disp_sq, 1, MPI_DOUBLE, MPI_MAX, ...)

2. Line 423: Virial for barostat (every step if barostat active)
   MPI_Allreduce(&local_virial, &global_virial, 1, MPI_DOUBLE, MPI_SUM, ...)

3. Line 647: Potential energy (called from output_results)
   MPI_Allreduce(&local_pe, &cached_potential_energy, 1, MPI_DOUBLE, MPI_SUM, ...)

4. Line 755: Particle count (every output step)
   MPI_Allreduce(&local_particle_count, &global_particles, 1, MPI_INT, MPI_SUM, ...)

5. Line 766: Max force across ranks (every output step)
   MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE_INT, MPI_MAXLOC, ...)
```

### Problem: Synchronous Collective Communication Overhead

- **NVE ensemble (no barostat):** ~2-3 MPI_Allreduce calls per output step
- **NPT ensemble (with barostat):** ~1 extra MPI_Allreduce every step
- Each `MPI_Allreduce` causes global synchronization barrier
- With 8+ processes, communication latency dominates

---

## 2. Optimization Strategy

### Phase 1: Minimize MPI Communication Frequency ✅

**Target:** Reduce MPI_Allreduce calls that are NOT critical for correctness

#### Changes Applied:

1. **Displacement check (Line 267):** ✅ KEPT
   - **Reasoning:** REQUIRED for correctness - must check global max displacement
   - **Frequency:** Only every `neighbor_list_update_freq` steps (e.g., every 20 steps)
   - **Impact:** Minimal overhead (already infrequent)

2. **Virial for barostat (Line 423):** ✅ KEPT (conditional)
   - **Reasoning:** REQUIRED for NPT ensemble pressure calculation
   - **Frequency:** Every step IF barostat is active, otherwise NEVER
   - **Impact:** Only affects NPT simulations (NVE/NVT unaffected)

3. **Potential energy (Line 647):** 🔄 OPTIMIZE
   - **Current:** Called from `calculate_potential_energy()` which is invoked by `get_potential_energy()`
   - **Problem:** May be called multiple times per step
   - **Solution:** Cache PE after force calculation, only MPI_Allreduce during output

4. **Particle count (Line 755):** 🔄 OPTIMIZE
   - **Current:** MPI_Allreduce every output step
   - **Problem:** Particle count rarely changes (only during migration)
   - **Solution:** Cache global particle count, update only after domain decomposition

5. **Max force (Line 766):** 🔄 OPTIMIZE
   - **Current:** MPI_Allreduce every output step
   - **Problem:** Only needed for output/diagnostics
   - **Solution:** Make optional, or combine with other reductions

### Phase 2: Batch MPI Communications 🚧

**Target:** Combine multiple MPI_Allreduce into single call

```cpp
// Instead of:
MPI_Allreduce(&local_pe, &global_pe, 1, MPI_DOUBLE, MPI_SUM, ...);
MPI_Allreduce(&local_ke, &global_ke, 1, MPI_DOUBLE, MPI_SUM, ...);
MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, ...);

// Do:
struct OutputData {
    double pe;
    double ke;
    int count;
} local_data, global_data;

MPI_Allreduce(&local_data, &global_data, 3, MPI_DOUBLE, MPI_SUM, ...);
```

### Phase 3: Async Communication (Advanced) 🔮

**Target:** Use non-blocking MPI_Iallreduce for diagnostics

```cpp
// Start async reduction while doing other work
MPI_Request req;
MPI_Iallreduce(&local_pe, &global_pe, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &req);

// Do other computations...
force_calc->calculate_forces(...);

// Wait for result only when needed
MPI_Wait(&req, MPI_STATUS_IGNORE);
```

---

## 3. Implementation Plan

### Step 1: Profile Current MPI Overhead ✅

```bash
# Strong scaling test
for NP in 1 2 4 8; do
    mpirun -np $NP ./bin/md_simulation_cpu examples/lj_32000/input.yaml
done
```

**Expected Results:**
- NP=1: 40-50 seconds (baseline, no MPI overhead)
- NP=2: 20-30 seconds (ideal: 50% of NP=1)
- NP=4: 10-20 seconds (ideal: 25% of NP=1)
- NP=8: 5-15 seconds (ideal: 12.5% of NP=1)

**Scaling Efficiency Formula:**
```
Efficiency = T(1) / (N × T(N)) × 100%

Target: ≥ 80% for 2-4 processes
        ≥ 60% for 8 processes
```

### Step 2: Implement PE Caching 🔄

**Current Issue:**
```cpp
// src/md_simulation.cpp:645
void MDSimulation::calculate_potential_energy() {
    // ... calculate local_pe ...
    MPI_Allreduce(&local_pe, &cached_potential_energy, 1, ...);  // ← Called frequently!
}
```

**Optimization:**
```cpp
// Add flag to control when MPI reduction is needed
void MDSimulation::calculate_potential_energy(bool do_mpi_reduce = false) {
    // ... calculate local_pe ...
    
#ifdef USE_MPI
    if (do_mpi_reduce) {
        MPI_Allreduce(&local_pe, &cached_potential_energy, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    } else {
        cached_potential_energy = local_pe;  // Store local value only
    }
#else
    cached_potential_energy = local_pe;
#endif
}

// In output_results():
void MDSimulation::output_results(int step) {
    // ... download from GPU if needed ...
    
    // Calculate and reduce PE ONLY during output
    calculate_potential_energy(true);  // ← Force MPI reduction
    
    // ... rest of output logic ...
}
```

### Step 3: Batch Output Reductions 🚧

**Target:** Single MPI_Allreduce for all output data

```cpp
void MDSimulation::output_results(int step) {
    // Gather all local output data
    struct OutputBuffer {
        double pe;
        double ke;
        double virial;
        double max_force;
        int particle_count;
        int max_force_rank;
    } local_buf, global_buf;
    
    local_buf.pe = calculate_local_pe();
    local_buf.ke = calculate_local_ke();
    local_buf.virial = force_calc->get_virial();
    local_buf.particle_count = system->get_local_particles();
    
    // Single MPI_Allreduce for all scalar quantities
    MPI_Allreduce(&local_buf.pe, &global_buf.pe, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_buf.particle_count, &global_buf.particle_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    // Use global_buf for output
    if (rank == 0) {
        std::cout << "PE: " << global_buf.pe << ", KE: " << global_buf.ke << std::endl;
    }
}
```

---

## 4. Benchmarking Results

### Test Configuration
- **System:** 32,000 LJ particles
- **Box:** 320³ Å³
- **Timesteps:** 10,000 (10 ps)
- **Output freq:** 1000 (10 outputs total)
- **Cutoff:** 12 Å
- **Neighbor update:** every 20 steps

### Baseline (Before Optimization)

| NP | Time (s) | Speedup | Efficiency (%) | Status |
|----|----------|---------|----------------|--------|
| 1  | TBD      | 1.00x   | 100%           | —      |
| 2  | TBD      | TBD     | TBD            | —      |
| 4  | TBD      | TBD     | TBD            | —      |
| 8  | TBD      | TBD     | TBD            | —      |

### After Optimization

| NP | Time (s) | Speedup | Efficiency (%) | Improvement |
|----|----------|---------|----------------|-------------|
| 1  | TBD      | 1.00x   | 100%           | —           |
| 2  | TBD      | TBD     | TBD            | TBD         |
| 4  | TBD      | TBD     | TBD            | TBD         |
| 8  | TBD      | TBD     | TBD            | TBD         |

---

## 5. Completion Criteria

### Must Pass:
- [x] Phase 1: Minimize MPI communication frequency (comments added)
- [ ] Phase 2: Batch output MPI reductions (implementation pending)
- [ ] NP=2 efficiency ≥ 80%
- [ ] NP=4 efficiency ≥ 70%
- [ ] NP=8 efficiency ≥ 50%

### Nice to Have:
- [ ] Phase 3: Async MPI_Iallreduce for diagnostics
- [ ] NP=8 efficiency ≥ 60%
- [ ] Load balancing analysis (particle distribution)
- [ ] Communication vs computation ratio breakdown

---

## 6. Next Steps

1. ✅ Add optimization comments to current MPI_Allreduce calls
2. 🔄 Run baseline MPI scaling test (in progress)
3. ⏳ Implement PE caching optimization
4. ⏳ Batch output MPI reductions
5. ⏳ Re-run scaling test and measure improvement
6. ⏳ Document final results in STAGE4_RESULTS.md

---

## Notes

- **Critical for correctness:** Displacement check (global max needed)
- **Critical for NPT:** Virial reduction (every step if barostat active)
- **Not critical:** PE/KE/max_force reductions (only for output/diagnostics)

**Optimization Philosophy:**
- Keep synchronization points minimal and intentional
- Batch communications when possible
- Cache values that don't change frequently
- Only reduce global quantities when actually needed for output

**MPI Best Practices Applied:**
- Minimize MPI_Allreduce frequency
- Use MPI_Allreduce only for truly global operations
- Avoid unnecessary synchronization barriers
- Batch multiple reductions into single call where possible
