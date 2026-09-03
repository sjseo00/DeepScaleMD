# DeepScaleMD Architecture & Code Flow Diagrams

학회 발표용 동작 과정 다이어그램 모음

---

## 1. 전체 시스템 아키텍처 (System Architecture)

```mermaid
graph TB
    subgraph INPUT["📄 Input Layer"]
        YAML["YAML Config\n(timestep, T, P, cutoff...)"]
        XYZ["XYZ Coordinates\n(init_*.xyz)"]
        TOP["Topology\n(topology.top)"]
    end

    subgraph MAIN["🚀 Main Entry (main.cpp)"]
        MPIINIT["MPI_Init_thread()\nOpenMP auto-tune"]
        SIM["MDSimulation Object"]
    end

    subgraph CORE["🧠 Core Orchestration (md_simulation.cpp)"]
        INIT["initialize()"]
        RUN["run() — Main MD Loop"]
        STEP["single_step()"]
        OUT["output_results()"]
    end

    subgraph SYSTEM["⚛️ MDSystem (md_system.cpp)"]
        PARTICLES["ParticleData\n(x, y, z, vx, vy, vz, fx, fy, fz)"]
        DOMAIN["DomainDecomposition\nCartesian MPI Grid"]
        HALO["Halo Exchange\n(Ghost Particles)"]
        MIGRATE["Particle Migration"]
    end

    subgraph FORCES["⚡ Force Calculation Layer"]
        FM["ForceManager\n(Factory Pattern)"]
        CPU["ForceCalculatorCPU\n(OpenMP)"]
        GPU["ForceCalculatorCUDA\n(CUDA Kernels)"]
        PME["PMECalculator\n(Long-range Electrostatics)"]
        ML["ForceCalculatorML\n(MACE Python)"]
    end

    subgraph NBLIST["🔍 Neighbor List (neighbor_list.cpp)"]
        VERLET["VerletCellList\nCSR Format"]
        MORTON["MortonZOrder\nSpace-filling Curve"]
    end

    subgraph DYNAMICS["🔄 Dynamics Components"]
        INT["Integrator\nVerlet / LeapFrog / r-RESPA"]
        THERM["Thermostat\nNose-Hoover / Langevin / Andersen"]
        BARO["Barostat\nParrinello-Rahman / Berendsen"]
    end

    subgraph OUTPUT["📊 Output"]
        XYZ_OUT["XYZ Trajectory"]
        LAMMPS["LAMMPS Dump"]
        THERMO["Thermodynamic Data\n(E, T, P)"]
    end

    YAML --> INIT
    XYZ --> INIT
    TOP --> INIT
    MPIINIT --> SIM
    SIM --> INIT
    INIT --> RUN
    RUN --> STEP
    STEP --> OUT

    STEP --> SYSTEM
    STEP --> FORCES
    STEP --> NBLIST
    STEP --> DYNAMICS

    SYSTEM --> DOMAIN
    SYSTEM --> HALO
    SYSTEM --> MIGRATE
    SYSTEM --> PARTICLES

    FORCES --> FM
    FM --> CPU
    FM --> GPU
    FM --> ML
    GPU --> PME

    NBLIST --> VERLET
    NBLIST --> MORTON

    DYNAMICS --> INT
    DYNAMICS --> THERM
    DYNAMICS --> BARO

    OUT --> XYZ_OUT
    OUT --> LAMMPS
    OUT --> THERMO

    style INPUT fill:#e8f4f8,stroke:#2980b9
    style MAIN fill:#fef9e7,stroke:#f39c12
    style CORE fill:#eaf9ea,stroke:#27ae60
    style SYSTEM fill:#fdf2f8,stroke:#8e44ad
    style FORCES fill:#fff3cd,stroke:#d68910
    style NBLIST fill:#e8f8f5,stroke:#1abc9c
    style DYNAMICS fill:#f9ebea,stroke:#e74c3c
    style OUTPUT fill:#eaf2ff,stroke:#3498db
```

---

## 2. 단일 타임스텝 실행 흐름 (Single Timestep Flow)

```mermaid
flowchart TD
    START(["🟢 START\nt = n·Δt"])

    subgraph FORCE1["Force Calculation — at x(n)"]
        CF["clear_forces_kernel()"]
        LJ["compute_lj_forces_kernel()\nLennard-Jones"]
        ELEC["compute_electrostatic_kernel()\nCoulomb / PME"]
        BOND["compute_bond_forces_kernel()\nHarmonic Springs"]
        ANGLE["compute_angle_forces_kernel()"]
        VIRIAL["Accumulate virial &\npotential_energy → device"]
        CF --> LJ --> ELEC --> BOND --> ANGLE --> VIRIAL
    end

    subgraph NBCHECK["Neighbor List Check (every N steps)"]
        DISP["max_displacement > skin/2 ?"]
        MPIALL1["MPI_Allreduce()\nglobal decision"]
        REBUILD["NeighborListBuilder::build()\nCell-list → CSR format"]
        HALO_FULL["perform_halo_exchange()\nfull ghost assembly"]
        REFRESH["update_ghost_positions()\nfast in-place refresh"]
        DISP -->|"MPI sync"| MPIALL1
        MPIALL1 -->|"YES"| REBUILD --> HALO_FULL
        MPIALL1 -->|"NO"| REFRESH
    end

    subgraph HALO_COMM["MPI Communication"]
        REV_HALO["perform_reverse_halo_exchange()\nghost forces → owner ranks"]
        MIGR["migrate_particles()\nredistribute escaped particles"]
    end

    subgraph INTEG1["Integration Stage 1 — Velocity Verlet"]
        VHD["v(t+Δt/2) = v(n) + ½·a(n)·Δt"]
        XNEW["x(n+1) = x(n) + v(t+Δt/2)·Δt"]
        PBC["apply_pbc()\nperiodic boundary"]
        VHD --> XNEW --> PBC
    end

    subgraph FORCE2["Force Calculation — at x(n+1)"]
        CF2["clear_forces_kernel()"]
        LJ2["compute_lj_forces_kernel()"]
        ELEC2["compute_electrostatic_kernel()"]
        BOND2["compute_bond_forces_kernel()"]
        VIRIAL2["Accumulate virial & energy"]
        CF2 --> LJ2 --> ELEC2 --> BOND2 --> VIRIAL2
    end

    subgraph INTEG2["Integration Stage 2 — Velocity Completion"]
        VFULL["v(n+1) = v(t+Δt/2) + ½·a(n+1)·Δt"]
        CONS["✅ Consistent state:\nx(n+1), v(n+1), f(n+1)"]
        VFULL --> CONS
    end

    subgraph ENSEMBLE["Ensemble Control"]
        THERM_APPLY["Thermostat::apply()\nT_current = 2·KE / (3·N·k_B)\nscale velocities or add noise"]
        MPIALL2["MPI_Allreduce(KE)"]
        BARO_APPLY["Barostat::apply()\nP = (2·KE + virial) / (3·V)\nscale box & positions"]
        MPIALL3["MPI_Allreduce(KE,virial)\nMPI_Bcast(box dims)"]
        BARO_FLAG["Set neighbor_rebuild_forced"]
        MPIALL2 --> THERM_APPLY
        MPIALL3 --> BARO_APPLY --> BARO_FLAG
    end

    subgraph OUTBLOCK["Output (every M steps)"]
        EOUT["write_xyz_trajectory()"]
        DOUT["write_lammps_dump()"]
        POUT["Print: E_tot, T, P, step"]
    end

    END_STEP(["🔵 END\nt = (n+1)·Δt"])

    START --> FORCE1
    FORCE1 --> NBCHECK
    NBCHECK --> HALO_COMM
    HALO_COMM --> INTEG1
    INTEG1 --> FORCE2
    FORCE2 --> INTEG2
    INTEG2 --> ENSEMBLE
    ENSEMBLE --> OUTBLOCK
    OUTBLOCK --> END_STEP

    style FORCE1 fill:#fff3cd,stroke:#e67e22,color:#000
    style NBCHECK fill:#e8f8f5,stroke:#16a085,color:#000
    style HALO_COMM fill:#fdf2f8,stroke:#8e44ad,color:#000
    style INTEG1 fill:#eaf9ea,stroke:#27ae60,color:#000
    style FORCE2 fill:#fff3cd,stroke:#e67e22,color:#000
    style INTEG2 fill:#eaf9ea,stroke:#27ae60,color:#000
    style ENSEMBLE fill:#f9ebea,stroke:#e74c3c,color:#000
    style OUTBLOCK fill:#eaf2ff,stroke:#3498db,color:#000
```

---

## 3. MPI 병렬화 구조 (MPI Parallelization)

```mermaid
graph TB
    subgraph GLOBAL["🌐 Global Domain (Full System)"]
        GB["Global Box\nL_x × L_y × L_z\nN_total particles"]
    end

    subgraph DECOMP["Domain Decomposition (n_x × n_y × n_z grid)"]
        R0["Rank 0\n(0,0,0)\nOwns subdomain"]
        R1["Rank 1\n(1,0,0)"]
        R2["Rank 2\n(0,1,0)"]
        R3["Rank 3\n(1,1,0)"]
    end

    subgraph RANK_DETAIL["Single Rank Detail"]
        OWN["Owned Particles\n(local_particles)"]
        GHOST["Ghost Particles\nfrom ≤26 neighbors"]
        NBUF["halo_send_buffers[26]\nhalo_recv_buffers[26]"]
        OWN -.->|"MPI_Isend/Irecv"| GHOST
    end

    subgraph SYNC_POINTS["🔄 MPI Synchronization Points"]
        S1["MPI_Allreduce(rebuild_flag)\nNeighbor list rebuild decision"]
        S2["MPI_Allreduce(KE)\nTemperature calculation"]
        S3["MPI_Allreduce(KE + virial)\nPressure calculation"]
        S4["MPI_Bcast(box dims)\nNPT box synchronization"]
        S5["MPI_Isend/Irecv\nHalo exchange (non-blocking)"]
        S6["MPI_Isend/Irecv\nParticle migration"]
    end

    GB -->|"Partition"| DECOMP
    R0 <-->|"Halo\nExchange"| R1
    R0 <-->|"Halo\nExchange"| R2
    R1 <-->|"Halo\nExchange"| R3
    R2 <-->|"Halo\nExchange"| R3

    DECOMP --> RANK_DETAIL
    RANK_DETAIL --> SYNC_POINTS

    style GLOBAL fill:#e8f4f8,stroke:#2980b9
    style DECOMP fill:#eaf9ea,stroke:#27ae60
    style RANK_DETAIL fill:#fff3cd,stroke:#e67e22
    style SYNC_POINTS fill:#fdf2f8,stroke:#8e44ad
```

---

## 4. GPU (CUDA) 데이터 흐름 (GPU Data Flow)

```mermaid
graph LR
    subgraph HOST["🖥️ CPU Host (RAM)"]
        H_POS["positions\nx, y, z"]
        H_VEL["velocities\nvx, vy, vz"]
        H_FOR["forces\nfx, fy, fz"]
        H_NB["neighbor list\n(CSR format)"]
        H_PARAM["FF params\nε, σ, q, m"]
        H_OUT["output buffers\nenergy, virial"]
    end

    subgraph PCIE["PCIe / NVLink"]
        TRANSFER["cudaMemcpy\n(H2D / D2H)"]
    end

    subgraph DEVICE["🎮 GPU Device (VRAM)"]
        D_POS["d_x, d_y, d_z"]
        D_VEL["d_vx, d_vy, d_vz"]
        D_FOR["d_fx, d_fy, d_fz"]
        D_NB["d_neighbors[]\nd_num_neighbors[]\nd_neighbor_offset[]"]
        D_PARAM["d_epsilon_mixed[]\nd_sigma_mixed[]"]
        D_ENERGY["d_potential_energy\nd_virial"]

        subgraph KERNELS["CUDA Kernels"]
            K1["clear_forces_kernel\n<<<blocks, threads>>>"]
            K2["compute_lj_forces_kernel\n<<<N/256, 256>>>"]
            K3["compute_electrostatic_kernel"]
            K4["compute_bond_forces_kernel"]
            K5["calculate_kinetic_energy_kernel"]
        end
    end

    subgraph CUDA_AWARE["CUDA-Aware MPI (optional)"]
        CAMPI["Direct GPU ↔ GPU\ncommunication\nno host staging"]
    end

    H_POS --> TRANSFER --> D_POS
    H_VEL --> TRANSFER --> D_VEL
    H_NB --> TRANSFER --> D_NB
    H_PARAM --> TRANSFER --> D_PARAM

    D_POS --> K2
    D_NB --> K2
    D_PARAM --> K2
    K2 --> D_FOR
    K2 --> D_ENERGY

    D_VEL --> K5
    K5 --> D_ENERGY

    D_FOR --> TRANSFER --> H_FOR
    D_ENERGY --> TRANSFER --> H_OUT

    D_FOR --> CAMPI
    D_POS --> CAMPI

    style HOST fill:#e8f4f8,stroke:#2980b9
    style DEVICE fill:#2c3e50,stroke:#f39c12,color:#fff
    style KERNELS fill:#1a252f,stroke:#f39c12,color:#fff
    style CUDA_AWARE fill:#fdf2f8,stroke:#8e44ad
    style PCIE fill:#eaf9ea,stroke:#27ae60
```

---

## 5. 컴포넌트 클래스 다이어그램 (Class Diagram)

```mermaid
classDiagram
    class MDSimulation {
        -MDSystem* system
        -ForceCalculator* force_calc
        -Integrator* integrator
        -Thermostat* thermostat
        -Barostat* barostat
        -NeighborListBuilder* nb_builder
        -double dt, cutoff, verlet_skin
        -int total_steps, output_freq
        +initialize(config_file)
        +run()
        +single_step()
        +output_results(step)
    }

    class MDSystem {
        -ParticleData particles
        -DomainDecomposition domain
        -Topology topology
        -int num_local, num_ghost
        +load_coordinates(xyz)
        +load_topology(top)
        +setup_domain_decomposition()
        +perform_halo_exchange()
        +update_ghost_positions()
        +perform_reverse_halo_exchange()
        +migrate_particles()
    }

    class ParticleData {
        +vector x, y, z
        +vector vx, vy, vz
        +vector fx, fy, fz
        +vector mass, charge, type
        +vector global_id
        +size_t num_particles
    }

    class ForceCalculatorCUDA {
        -float* d_x, d_y, d_z
        -float* d_fx, d_fy, d_fz
        -int* d_neighbors, d_num_neighbors
        -float* d_epsilon_mixed, d_sigma_mixed
        +compute_forces()
        +compute_lj_forces()
        +compute_electrostatic_forces()
        +compute_bond_forces()
        +calculate_kinetic_energy()
    }

    class NeighborListBuilder {
        <<abstract>>
        +build(particles, box, cutoff, skin)
        +needs_rebuild()
    }

    class VerletCellListBuilder {
        -vector neighbors_csr
        -double max_displacement
        +build()
        +needs_rebuild()
    }

    class Integrator {
        <<abstract>>
        +integrate(particles, forces, dt)
        +complete_velocity_update()
    }

    class VelocityVerletIntegrator {
        +integrate()
        +complete_velocity_update()
    }

    class Thermostat {
        <<abstract>>
        +apply(particles, T_target)
    }

    class NoseHooverThermostat {
        -double Q
        +apply()
    }

    class LangevinThermostat {
        -double gamma, sigma
        +apply()
    }

    class Barostat {
        <<abstract>>
        +apply(system, P_target)
    }

    class ParrinelloRahmanBarostat {
        -double W
        -matrix3x3 h, h_dot
        +apply()
    }

    MDSimulation --> MDSystem
    MDSimulation --> ForceCalculatorCUDA
    MDSimulation --> NeighborListBuilder
    MDSimulation --> Integrator
    MDSimulation --> Thermostat
    MDSimulation --> Barostat
    MDSystem --> ParticleData
    NeighborListBuilder <|-- VerletCellListBuilder
    Integrator <|-- VelocityVerletIntegrator
    Thermostat <|-- NoseHooverThermostat
    Thermostat <|-- LangevinThermostat
    Barostat <|-- ParrinelloRahmanBarostat
```

---

## 6. 앙상블별 실행 경로 (Ensemble Execution Paths)

```mermaid
graph TD
    CONFIG["Simulation Config\n(YAML)"] --> ETYPE{"Ensemble\nType?"}

    ETYPE -->|"NVE\n(microcanonical)"| NVE_PATH
    ETYPE -->|"NVT\n(canonical)"| NVT_PATH
    ETYPE -->|"NPT\n(isothermal-isobaric)"| NPT_PATH

    subgraph NVE_PATH["NVE Path"]
        NVE1["Forces"]
        NVE2["Verlet Integration\nv, x update"]
        NVE3["Energy conserved\n(no T/P control)"]
        NVE1 --> NVE2 --> NVE3
    end

    subgraph NVT_PATH["NVT Path"]
        NVT1["Forces"]
        NVT2["Verlet Integration"]
        NVT3["Thermostat::apply()\n↓\nNose-Hoover λ scaling\nor Langevin stochastic\nor Andersen collision"]
        NVT1 --> NVT2 --> NVT3
    end

    subgraph NPT_PATH["NPT Path"]
        NPT1["Forces"]
        NPT2["Verlet Integration"]
        NPT3["Thermostat::apply()"]
        NPT4["Barostat::apply()\n↓\nP = (2KE + virial)/(3V)\nScale box & positions\nMPI_Bcast(box dims)"]
        NPT5["Force neighbor\nrebuild forced"]
        NPT1 --> NPT2 --> NPT3 --> NPT4 --> NPT5
    end

    style NVE_PATH fill:#eaf9ea,stroke:#27ae60
    style NVT_PATH fill:#fff3cd,stroke:#e67e22
    style NPT_PATH fill:#f9ebea,stroke:#e74c3c
```

---

## 7. 이웃 목록 알고리즘 (Neighbor List Algorithm)

```mermaid
flowchart TD
    A["모든 입자 (local + ghost)"] --> B["셀 격자 생성\ncell_size = cutoff + skin"]
    B --> C["입자를 셀에 배치\n(binning)"]
    C --> D["각 local 입자 i에 대해"]
    D --> E["27개 인접 셀 탐색\n(3×3×3 stencil)"]
    E --> F{"r_ij < cutoff + skin?"}
    F -->|"YES"| G["j를 i의 이웃 목록에 추가\nCSR 형식으로 저장"]
    F -->|"NO"| H["skip"]
    G --> I["다음 셀의 입자"]
    H --> I
    I --> J{"모든 셀\n처리 완료?"}
    J -->|"NO"| E
    J -->|"YES"| K["GPU 메모리로 전송\nd_neighbors[], d_num_neighbors[]\nd_neighbor_offset[]"]

    subgraph REBUILD_CHECK["재구성 판단 (매 N 스텝)"]
        RC1["max_displacement = max_i |x_i - x_i_old|"]
        RC2["MPI_Allreduce(max_displacement)"]
        RC3{"max_displacement\n> skin / 2 ?"}
        RC4["목록 재구성 필요"]
        RC5["기존 목록 재사용\n(fast path)"]
        RC1 --> RC2 --> RC3
        RC3 -->|"YES"| RC4
        RC3 -->|"NO"| RC5
    end

    K --> REBUILD_CHECK

    style REBUILD_CHECK fill:#e8f4f8,stroke:#2980b9
```

---

## 8. 성능 최적화 기법 요약 (Optimizations Summary)

```mermaid
mindmap
  root((DeepScaleMD\nOptimizations))
    GPU Acceleration
      CUDA Force Kernels
        LJ forces
        Electrostatics
        Bond forces
      CUDA-Aware MPI
        Direct GPU↔GPU comm
      Shared Memory
        Tile-based force calc
    MPI Parallelization
      Domain Decomposition
        Cartesian grid
        26-neighbor halo
      Non-blocking Comms
        MPI_Isend/Irecv
      Load Balancing
        Uniform subdomain
    Neighbor List
      Verlet Skin Method
        Avoids rebuild every step
      Cell List Algorithm
        O(N) instead of O(N²)
      Morton Z-Order
        Cache-friendly layout
    Memory Optimization
      CSR Neighbor Format
        Compact storage
      Mixed Precision
        Float32 on GPU
      Precomputed FF Params
        Mixing rules cached
    Algorithm Choices
      Velocity Verlet
        Time-reversible
        Symplectic
      r-RESPA Multi-timescale
        Fast+slow forces
      PME Electrostatics
        O(N log N) long-range
```

---

## 단위계 (Unit System — LAMMPS 'real' units)

| 물리량 | 단위 |
|--------|------|
| 길이 | Å (옹스트롬) |
| 시간 | fs (펨토초) |
| 에너지 | kcal/mol |
| 온도 | K |
| 압력 | bar |
| 질량 | g/mol |
| 힘 | kcal/(mol·Å) |

**핵심 변환 상수:**
- `mvv2e = 2390.057` — (g/mol)·(Å/fs)² → kcal/mol
- `nktv2p = 68568.415` — bar·Å³/kcal
- `FORCE_TO_ACCEL = 4.184×10⁻⁴` — kcal/(mol·Å) → Å/fs²

---

*Generated for DeepScaleMD — GPU-accelerated MPI Molecular Dynamics Simulation*
