#ifndef MINIMIZER_H
#define MINIMIZER_H

#include "md_types.h"

struct FireConfig {
    int  max_steps       = 1000;
    Real force_tolerance = 1.0;   // kcal/mol/Angstrom; stop when global max |F_i| < this
    Real dt_start         = 0.5;  // fs
    Real dt_max           = 2.0;  // fs
    int  n_min            = 5;    // steps of P>0 required before dt/alpha adapt
    Real f_inc            = 1.1;
    Real f_dec            = 0.5;
    Real alpha_start      = 0.1;
    Real f_alpha          = 0.99;
};

class FireMinimizer {
public:
    explicit FireMinimizer(const FireConfig& cfg)
        : cfg_(cfg), dt_(cfg.dt_start), alpha_(cfg.alpha_start) {}

    struct StepParams {
        bool reset_velocity = false;  // P <= 0 branch: zero v before the kick
        bool mix_velocity   = false;  // P > 0 branch: blend v toward F direction
        Real alpha = 0.0;             // mixing coefficient to use this iteration
        Real dt    = 0.0;             // pseudo-timestep to use this iteration
        Real v_norm = 0.0;            // |v| from the END of the previous iteration
        Real f_norm = 0.0;            // |F| at the current position
    };

    StepParams advance(Real global_power, Real global_f_norm, Real global_v_norm);

    const FireConfig& config() const { return cfg_; }

private:
    FireConfig cfg_;
    Real dt_;
    Real alpha_;
    int n_pos_steps_ = 0;
};

#endif // MINIMIZER_H
