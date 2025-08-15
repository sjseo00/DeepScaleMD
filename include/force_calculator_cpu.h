#ifndef FORCE_CALCULATOR_CPU_H
#define FORCE_CALCULATOR_CPU_H

#include "force_calculator.h"

class ForceCalculatorCPU : public ForceCalculator {
public:
    ForceCalculatorCPU() = default;
    virtual ~ForceCalculatorCPU() = default;
    
    virtual void initialize() override;
    virtual void finalize() override;
    virtual void compute_forces(MDSystem& system) override;
    virtual void calculate_kinetic_energy(MDSystem& system) override;

private:
    void compute_bond_forces(MDSystem& system);
    void compute_angle_forces(MDSystem& system);
    void compute_dihedral_forces(MDSystem& system);
    void compute_lj_forces(MDSystem& system);
    void compute_electrostatic_forces(MDSystem& system);
};

#endif // FORCE_CALCULATOR_CPU_H
