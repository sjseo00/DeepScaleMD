// Parameters of so3lr::pair_energy and so3lr::edge_geometry, shared by so3lr_fused_ops.cpp and so3lr_fused_kernels.cu (the operator's float[] params in order).
#pragma once

struct So3lrPairParams {
    double lr_cut;        // long-range cutoff (A)
    double sigma;         // electrostatics damping erf(r / sigma) (A)
    double cuton;         // electrostatics switching starts here (0.45 x cutoff)
    double coul_shift;    // erf(rc / sigma) / rc
    double coul_fshift;   // derivative of erf(r / sigma) / r at rc
    double ke;            // Coulomb constant (eV A / e^2)
    double disp_on;       // dispersion switching starts here (cutoff - damping)
    double disp_scale;    // so3lr dispersion_energy_scale
    double bohr;          // A
    double hartree;       // eV
    double vdw_factor;    // fine_structure^(-4/21)
    int use_elec;
    int use_disp;
    // With --kspace pme (the real-space parts of an Ewald split; 0 = off, the 13-value parameter list of older folders):
    double pme_smear;     // electrostatics: erf(r / pme_smear) / r is left to the k-space sum (pme_smear = sqrt(2) x smearing, A)
    double disp_smear;    // dispersion: sqrt(c6_i c6_j) (1 - exp(-x)(1 + x + x^2/2)) / rb^6 added, x = rb^2 / (2 disp_smear^2) (bohr)
};

// so3lr::edge_geometry (the operator's float[] params in order)
struct So3lrGeometryParams {
    double sr_cutoff;          // model cutoff (A): PhysNet cutoff of the short-range edges
    double floor_width;        // NLH distance floor width (A)
    double ke;                 // Coulomb constant (eV A / e^2), NLH repulsion prefactor
    double repulsion_switch;   // NLH repulsion switched off from 0 to here (A)
    int use_repulsion;
};
