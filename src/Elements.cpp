#include "Elements.h"

#include <array>

namespace {

constexpr std::array<const char*, 118> kSymbols = {
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"};

// standard atomic masses (u) of the elements 1..94
constexpr std::array<double, 94> kMasses = {
    1.008,   4.0026,  6.94,    9.0122,  10.81,   12.011,  14.007,  15.999,  18.998,  20.180,  22.990,  24.305,  26.982,  28.085,  30.974,
    32.06,   35.45,   39.948,  39.098,  40.078,  44.956,  47.867,  50.942,  51.996,  54.938,  55.845,  58.933,  58.693,  63.546,  65.38,
    69.723,  72.630,  74.922,  78.971,  79.904,  83.798,  85.468,  87.62,   88.906,  91.224,  92.906,  95.95,   97.907,  101.07,  102.91,
    106.42,  107.87,  112.41,  114.82,  118.71,  121.76,  127.60,  126.90,  131.29,  132.91,  137.33,  138.91,  140.12,  140.91,  144.24,
    144.913, 150.36,  151.96,  157.25,  158.93,  162.50,  164.93,  167.26,  168.93,  173.05,  174.97,  178.49,  180.95,  183.84,  186.21,
    190.23,  192.22,  195.08,  196.97,  200.59,  204.38,  207.2,   208.98,  208.982, 209.987, 222.018, 223.020, 226.025, 227.028, 232.04,
    231.04,  238.03,  237.048, 244.064};

}  // namespace

int atomic_number_from_symbol(const std::string& symbol) {
    for (size_t i = 0; i < kSymbols.size(); ++i) {
        if (symbol == kSymbols[i]) return static_cast<int>(i) + 1;
    }
    return 0;
}

std::string element_symbol(int z) {
    if (z < 1 || z > static_cast<int>(kSymbols.size())) return "";
    return kSymbols[static_cast<size_t>(z) - 1];
}

int atomic_number_from_mass(double mass, double tolerance) {
    int found = 0;
    for (size_t i = 0; i < kMasses.size(); ++i) {
        const double difference = kMasses[i] > mass ? kMasses[i] - mass : mass - kMasses[i];
        if (difference <= tolerance) {
            if (found != 0) return 0;       // two elements are that close: not a safe answer
            found = static_cast<int>(i) + 1;
        }
    }
    return found;
}
