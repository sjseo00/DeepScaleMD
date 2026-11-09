#ifndef MLIP_KSPACE_HOST_H
#define MLIP_KSPACE_HOST_H

// The MD side of the k-space sums of an MLIP (MLIPKspace.h): checks that the model and the input file agree, sets the box, grid and
// grid sum of the operator's context for every model call, and adds the operator's virial correction. Created only when the model
// has metadata.json "kspace" or the input says mlip_kspace: pme (and then only if both do); nothing else uses it.

#include "MLIPKspace.h"

#include <memory>
#include <string>

class MLIPKspaceHost {
public:
    // nullptr when neither the model (`model`: its metadata settings, nullptr or disabled = none) nor the input asks for k-space sums.
    // Throws when they disagree (a model converted with --kspace pme without mlip_kspace: pme, or the other way round), when the
    // box is not periodic along every axis, or when a grid setting is invalid. `verbose`: print the settings (rank 0).
    static std::unique_ptr<MLIPKspaceHost> create(const MLIPKspaceSettings* model, const MLIPKspaceInput& input, bool periodic_all,
                                                  bool verbose);

    // The context for the next model call: box (A); sum_over_ranks: the ranks hold different atoms (domain decomposition) and sum
    // their grids over the MD ranks; ranks: how many. Resets the virial correction.
    MLIPKspaceContext* prepare(const double box[3], bool sum_over_ranks, int ranks);

    // The operator's virial correction of the last call in the model's energy unit, W_aa (their sum is the trace), and that unit in
    // kcal/mol (the MD's).
    const double* virial_correction() const { return context_.virial_correction; }
    double energy_to_md() const { return settings_.energy_to_md; }
    int calls() const { return context_.calls; }

    ~MLIPKspaceHost() = default;

private:
    MLIPKspaceHost() = default;
    MLIPKspaceSettings settings_;
    MLIPKspaceContext context_;
};

#endif // MLIP_KSPACE_HOST_H
