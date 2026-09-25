#include "minimizer.h"

#include <algorithm>

FireMinimizer::StepParams FireMinimizer::advance(Real global_power, Real global_f_norm,
                                                 Real global_v_norm) {
    StepParams p;
    p.f_norm = global_f_norm;

    if (global_power > 0.0) {
        p.mix_velocity = true;
        p.alpha = alpha_;
        p.v_norm = global_v_norm;
        ++n_pos_steps_;
        if (n_pos_steps_ > cfg_.n_min) {
            dt_ = std::min(dt_ * cfg_.f_inc, cfg_.dt_max);
            alpha_ *= cfg_.f_alpha;
        }
    } else {
        p.reset_velocity = true;
        p.v_norm = 0.0;
        n_pos_steps_ = 0;
        alpha_ = cfg_.alpha_start;
        dt_ *= cfg_.f_dec;
        p.alpha = alpha_;
    }

    p.dt = dt_;
    return p;
}
