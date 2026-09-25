
#include "topology.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#ifdef USE_MPI
#include <mpi.h>
#endif

Topology::Topology() : rank(0) {}

Topology::~Topology() {}

namespace {

std::string trim(const std::string& input) {
    const size_t first = input.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const size_t last = input.find_last_not_of(" \t\r\n");
    return input.substr(first, last - first + 1);
}

std::vector<std::string> tokenize_data_line(std::string line) {
    const size_t semicolon = line.find(';');
    const size_t hash = line.find('#');
    size_t comment = std::min(semicolon, hash);
    if (semicolon == std::string::npos) comment = hash;
    if (hash == std::string::npos) comment = semicolon;
    if (comment != std::string::npos) line.erase(comment);
    std::istringstream stream(trim(line));
    std::vector<std::string> tokens;
    std::string token;
    while (stream >> token) tokens.push_back(token);
    return tokens;
}

std::uint64_t pair_key(int atom1, int atom2) {
    const std::uint32_t lo = static_cast<std::uint32_t>(std::min(atom1, atom2));
    const std::uint32_t hi = static_cast<std::uint32_t>(std::max(atom1, atom2));
    return (static_cast<std::uint64_t>(lo) << 32) | hi;
}

void broadcast_string(std::string& value, MPI_Comm comm, int root, int rank) {
    std::uint64_t size = (rank == root) ? static_cast<std::uint64_t>(value.size()) : 0;
    MPI_Bcast(&size, 1, MPI_UINT64_T, root, comm);
    if (rank != root) value.resize(static_cast<size_t>(size));
    if (size != 0) {
        MPI_Bcast(value.data(), static_cast<int>(size), MPI_CHAR, root, comm);
    }
}

} // namespace

void Topology::broadcast(MPI_Comm comm, int root) {
    MPI_Comm_rank(comm, &rank);

    auto broadcast_count = [&](size_t local_size) {
        std::uint64_t count = (rank == root) ? static_cast<std::uint64_t>(local_size) : 0;
        MPI_Bcast(&count, 1, MPI_UINT64_T, root, comm);
        return static_cast<size_t>(count);
    };

    const size_t atom_type_count = broadcast_count(atom_types.size());
    if (rank != root) atom_types.clear();
    if (rank == root) {
        for (auto& entry : atom_types) {
            std::string name = entry.first;
            broadcast_string(name, comm, root, rank);
            Real values[4] = {entry.second.mass, entry.second.charge,
                              entry.second.sigma, entry.second.epsilon};
            MPI_Bcast(values, 4, MPI_DOUBLE, root, comm);
        }
    } else {
        for (size_t i = 0; i < atom_type_count; ++i) {
            std::string name;
            broadcast_string(name, comm, root, rank);
            Real values[4]{};
            MPI_Bcast(values, 4, MPI_DOUBLE, root, comm);
            atom_types[name] = AtomType{name, values[0], values[1], values[2], values[3]};
        }
    }

    const size_t bond_type_count = broadcast_count(bond_types.size());
    if (rank != root) bond_types.clear();
    if (rank == root) {
        for (auto& entry : bond_types) {
            std::string name = entry.first;
            broadcast_string(name, comm, root, rank);
            Real values[2] = {entry.second.k, entry.second.r0};
            MPI_Bcast(values, 2, MPI_DOUBLE, root, comm);
        }
    } else {
        for (size_t i = 0; i < bond_type_count; ++i) {
            std::string name;
            broadcast_string(name, comm, root, rank);
            Real values[2]{};
            MPI_Bcast(values, 2, MPI_DOUBLE, root, comm);
            bond_types[name] = BondType{name, values[0], values[1]};
        }
    }

    const size_t angle_type_count = broadcast_count(angle_types.size());
    if (rank != root) angle_types.clear();
    if (rank == root) {
        for (auto& entry : angle_types) {
            std::string name = entry.first;
            broadcast_string(name, comm, root, rank);
            Real values[2] = {entry.second.k, entry.second.theta0};
            MPI_Bcast(values, 2, MPI_DOUBLE, root, comm);
        }
    } else {
        for (size_t i = 0; i < angle_type_count; ++i) {
            std::string name;
            broadcast_string(name, comm, root, rank);
            Real values[2]{};
            MPI_Bcast(values, 2, MPI_DOUBLE, root, comm);
            angle_types[name] = AngleType{name, values[0], values[1]};
        }
    }

    const size_t dihedral_type_count = broadcast_count(dihedral_types.size());
    if (rank != root) dihedral_types.clear();
    if (rank == root) {
        for (auto& entry : dihedral_types) {
            std::string name = entry.first;
            broadcast_string(name, comm, root, rank);
            Real values[2] = {entry.second.k, entry.second.phi0};
            int multiplicity = entry.second.n;
            MPI_Bcast(values, 2, MPI_DOUBLE, root, comm);
            MPI_Bcast(&multiplicity, 1, MPI_INT, root, comm);
        }
    } else {
        for (size_t i = 0; i < dihedral_type_count; ++i) {
            std::string name;
            broadcast_string(name, comm, root, rank);
            Real values[2]{};
            int multiplicity = 0;
            MPI_Bcast(values, 2, MPI_DOUBLE, root, comm);
            MPI_Bcast(&multiplicity, 1, MPI_INT, root, comm);
            dihedral_types[name] = DihedralType{name, values[0], values[1], multiplicity};
        }
    }

    const size_t atom_count = broadcast_count(atom_type_names.size());
    if (rank != root) {
        atom_type_names.resize(atom_count);
        atom_elements.resize(atom_count);
        atom_type_ids.resize(atom_count);
        atom_masses.resize(atom_count);
        atom_charges.resize(atom_count);
    }
    for (size_t i = 0; i < atom_count; ++i) {
        broadcast_string(atom_type_names[i], comm, root, rank);
        broadcast_string(atom_elements[i], comm, root, rank);
    }
    if (atom_count != 0) {
        MPI_Bcast(atom_type_ids.data(), static_cast<int>(atom_count), MPI_INT, root, comm);
        MPI_Bcast(atom_masses.data(), static_cast<int>(atom_count), MPI_DOUBLE, root, comm);
        MPI_Bcast(atom_charges.data(), static_cast<int>(atom_count), MPI_DOUBLE, root, comm);
    }

    const size_t bond_count = broadcast_count(bonds.size());
    if (rank != root) bonds.resize(bond_count);
    for (size_t i = 0; i < bond_count; ++i) {
        int atoms[2]{};
        if (rank == root) { atoms[0] = bonds[i].atom1; atoms[1] = bonds[i].atom2; }
        MPI_Bcast(atoms, 2, MPI_INT, root, comm);
        if (rank != root) { bonds[i].atom1 = atoms[0]; bonds[i].atom2 = atoms[1]; }
        broadcast_string(bonds[i].type, comm, root, rank);
    }

    const size_t angle_count = broadcast_count(angles.size());
    if (rank != root) angles.resize(angle_count);
    for (size_t i = 0; i < angle_count; ++i) {
        int atoms[3]{};
        if (rank == root) {
            atoms[0] = angles[i].atom1; atoms[1] = angles[i].atom2; atoms[2] = angles[i].atom3;
        }
        MPI_Bcast(atoms, 3, MPI_INT, root, comm);
        if (rank != root) {
            angles[i].atom1 = atoms[0]; angles[i].atom2 = atoms[1]; angles[i].atom3 = atoms[2];
        }
        broadcast_string(angles[i].type, comm, root, rank);
    }

    const size_t dihedral_count = broadcast_count(dihedrals.size());
    if (rank != root) dihedrals.resize(dihedral_count);
    for (size_t i = 0; i < dihedral_count; ++i) {
        int atoms[4]{};
        if (rank == root) {
            atoms[0] = dihedrals[i].atom1; atoms[1] = dihedrals[i].atom2;
            atoms[2] = dihedrals[i].atom3; atoms[3] = dihedrals[i].atom4;
        }
        MPI_Bcast(atoms, 4, MPI_INT, root, comm);
        if (rank != root) {
            dihedrals[i].atom1 = atoms[0]; dihedrals[i].atom2 = atoms[1];
            dihedrals[i].atom3 = atoms[2]; dihedrals[i].atom4 = atoms[3];
        }
        broadcast_string(dihedrals[i].type, comm, root, rank);
    }

    const size_t improper_count = broadcast_count(impropers.size());
    if (rank != root) impropers.resize(improper_count);
    for (size_t i = 0; i < improper_count; ++i) {
        int atoms[4]{};
        if (rank == root) {
            atoms[0] = impropers[i].atom1; atoms[1] = impropers[i].atom2;
            atoms[2] = impropers[i].atom3; atoms[3] = impropers[i].atom4;
        }
        MPI_Bcast(atoms, 4, MPI_INT, root, comm);
        if (rank != root) {
            impropers[i].atom1 = atoms[0]; impropers[i].atom2 = atoms[1];
            impropers[i].atom3 = atoms[2]; impropers[i].atom4 = atoms[3];
        }
        broadcast_string(impropers[i].type, comm, root, rank);
    }

    const size_t special_pair_count = broadcast_count(special_pairs.size());
    if (rank != root) special_pairs.resize(special_pair_count);
    for (size_t i = 0; i < special_pair_count; ++i) {
        int atoms[2]{};
        Real scales[4]{};
        if (rank == root) {
            atoms[0] = special_pairs[i].atom1; atoms[1] = special_pairs[i].atom2;
            scales[0] = special_pairs[i].lj_scale; scales[1] = special_pairs[i].coulomb_scale;
            scales[2] = special_pairs[i].sigma; scales[3] = special_pairs[i].epsilon;
        }
        MPI_Bcast(atoms, 2, MPI_INT, root, comm);
        MPI_Bcast(scales, 4, MPI_DOUBLE, root, comm);
        if (rank != root) {
            special_pairs[i].atom1 = atoms[0]; special_pairs[i].atom2 = atoms[1];
            special_pairs[i].lj_scale = scales[0]; special_pairs[i].coulomb_scale = scales[1];
            special_pairs[i].sigma = scales[2]; special_pairs[i].epsilon = scales[3];
        }
    }

    const size_t constraint_count = broadcast_count(constraints.size());
    if (rank != root) constraints.resize(constraint_count);
    for (size_t i = 0; i < constraint_count; ++i) {
        int atoms[2]{};
        Real distance = 0.0;
        if (rank == root) {
            atoms[0] = constraints[i].atom1; atoms[1] = constraints[i].atom2;
            distance = constraints[i].distance;
        }
        MPI_Bcast(atoms, 2, MPI_INT, root, comm);
        MPI_Bcast(&distance, 1, MPI_DOUBLE, root, comm);
        if (rank != root) {
            constraints[i].atom1 = atoms[0]; constraints[i].atom2 = atoms[1];
            constraints[i].distance = distance;
        }
    }

    int canonical = canonical_format ? 1 : 0;
    MPI_Bcast(&canonical, 1, MPI_INT, root, comm);
    canonical_format = canonical != 0;
    int mie_exponents[2] = {mie_repulsive_exponent, mie_attractive_exponent};
    MPI_Bcast(mie_exponents, 2, MPI_INT, root, comm);
    mie_repulsive_exponent = mie_exponents[0];
    mie_attractive_exponent = mie_exponents[1];

    const size_t pair_type_count = broadcast_count(pair_sigma.size());
    if (rank != root) {
        pair_sigma.assign(pair_type_count, std::vector<Real>(pair_type_count));
        pair_epsilon.assign(pair_type_count, std::vector<Real>(pair_type_count));
    }
    for (size_t i = 0; i < pair_type_count; ++i) {
        MPI_Bcast(pair_sigma[i].data(), static_cast<int>(pair_type_count), MPI_DOUBLE, root, comm);
        MPI_Bcast(pair_epsilon[i].data(), static_cast<int>(pair_type_count), MPI_DOUBLE, root, comm);
    }

    const size_t exclusion_count = broadcast_count(exclusion_keys.size());
    if (rank != root) exclusion_keys.resize(exclusion_count);
    if (exclusion_count != 0) {
        MPI_Bcast(exclusion_keys.data(), static_cast<int>(exclusion_count),
                  MPI_UINT64_T, root, comm);
    }
}

bool Topology::load_topology(const std::string& filename, int mpi_rank) {
    rank = mpi_rank;  // Store the rank for use throughout the class
    atom_types.clear(); bond_types.clear(); angle_types.clear(); dihedral_types.clear();
    atom_type_names.clear(); atom_elements.clear(); atom_type_ids.clear(); atom_masses.clear(); atom_charges.clear();
    bonds.clear(); angles.clear(); dihedrals.clear(); impropers.clear(); special_pairs.clear();
    constraints.clear();
    pair_sigma.clear(); pair_epsilon.clear(); exclusion_keys.clear();
    canonical_format = false;
    mie_repulsive_exponent = 12;
    mie_attractive_exponent = 6;
    std::string clean_filename = filename;
    
    auto start = clean_filename.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        clean_filename = "";
    } else {
        auto end = clean_filename.find_last_not_of(" \t\r\n");
        clean_filename = clean_filename.substr(start, end - start + 1);
    }
    
    if (!clean_filename.empty() && clean_filename.front() == '"' && clean_filename.back() == '"') {
        clean_filename = clean_filename.substr(1, clean_filename.length() - 2);
    }
    
    if (rank == 0) {
        std::cout << "Opening topology file: " << clean_filename << std::endl;
    }
    
    std::ifstream file(clean_filename);
    if (!file.is_open()) {
        return false;
    }
    
    std::string line;
    std::string current_section = "";
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#' || line[0] == ';') continue;
        
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);
        if (line.empty()) continue;
        
        if (line[0] == '[' && line.back() == ']') {
            current_section = line.substr(1, line.length() - 2);
            
            current_section.erase(0, current_section.find_first_not_of(" \t"));
            current_section.erase(current_section.find_last_not_of(" \t") + 1);
            
            std::transform(current_section.begin(), current_section.end(), 
                         current_section.begin(), ::tolower);
            continue;
        }
        
        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string token;
        while (iss >> token) {
            tokens.push_back(token);
        }
        
        if (tokens.empty()) continue;
        
        if (current_section == "atomtypes") {
            if (tokens.size() >= 5) {
                AtomType atom_type;
                atom_type.name = tokens[0];
                atom_type.mass = std::stod(tokens[1]);
                atom_type.charge = std::stod(tokens[2]);
                atom_type.sigma = std::stod(tokens[3]);
                atom_type.epsilon = std::stod(tokens[4]);
                atom_types[atom_type.name] = atom_type;
            }
        }
        else if (current_section == "bondtypes") {
            if (tokens.size() >= 4) {
                BondType bond_type;
                bond_type.name = tokens[0] + "-" + tokens[1];  // e.g., "C-C"
                bond_type.k = std::stod(tokens[2]);
                bond_type.r0 = std::stod(tokens[3]);
                bond_types[bond_type.name] = bond_type;
            }
        }
        else if (current_section == "angletypes") {
            if (tokens.size() >= 5) {
                AngleType angle_type;
                angle_type.name = tokens[0] + "-" + tokens[1] + "-" + tokens[2];  // e.g., "C-C-H"
                angle_type.k = std::stod(tokens[3]);
                angle_type.theta0 = std::stod(tokens[4]) * M_PI / 180.0;  // Convert degrees to radians
                angle_types[angle_type.name] = angle_type;
            }
        }
        else if (current_section == "dihedraltypes") {
            if (tokens.size() >= 6) {
                DihedralType dihedral_type;
                dihedral_type.name = tokens[0] + "-" + tokens[1] + "-" + tokens[2] + "-" + tokens[3];
                dihedral_type.k = std::stod(tokens[4]);
                dihedral_type.phi0 = std::stod(tokens[5]) * M_PI / 180.0;  // Convert degrees to radians
                dihedral_type.n = (tokens.size() > 6) ? std::stoi(tokens[6]) : 1;
                dihedral_types[dihedral_type.name] = dihedral_type;
            }
        }
        else if (current_section == "atoms") {
            if (tokens.size() >= 2) {
                int atom_id = std::stoi(tokens[0]) - 1;  // Convert to 0-based indexing
                std::string type_name = tokens[1];
                
                if (static_cast<size_t>(atom_id) >= atom_type_names.size()) {
                    atom_type_names.resize(atom_id + 1);
                }
                atom_type_names[atom_id] = type_name;
            }
        }
        else if (current_section == "bonds") {
            if (tokens.size() >= 3) {
                Bond bond;
                bond.atom1 = std::stoi(tokens[0]) - 1;  // Convert to 0-based indexing
                bond.atom2 = std::stoi(tokens[1]) - 1;
                bond.type = tokens[2];
                bonds.push_back(bond);
            }
        }
        else if (current_section == "angles") {
            if (tokens.size() >= 4) {
                Angle angle;
                angle.atom1 = std::stoi(tokens[0]) - 1;  // Convert to 0-based indexing
                angle.atom2 = std::stoi(tokens[1]) - 1;
                angle.atom3 = std::stoi(tokens[2]) - 1;
                angle.type = tokens[3];
                angles.push_back(angle);
            }
        }
        else if (current_section == "dihedrals") {
            if (tokens.size() >= 5) {
                Dihedral dihedral;
                dihedral.atom1 = std::stoi(tokens[0]) - 1;  // Convert to 0-based indexing
                dihedral.atom2 = std::stoi(tokens[1]) - 1;
                dihedral.atom3 = std::stoi(tokens[2]) - 1;
                dihedral.atom4 = std::stoi(tokens[3]) - 1;
                dihedral.type = tokens[4];
                dihedrals.push_back(dihedral);
            }
        }
    }

    std::map<std::string, int> legacy_type_ids;
    for (size_t i = 0; i < atom_type_names.size(); ++i) {
        const std::string& name = atom_type_names[i];
        auto inserted = legacy_type_ids.emplace(name, static_cast<int>(legacy_type_ids.size()));
        atom_type_ids.push_back(inserted.first->second);
        atom_elements.push_back("");
        const AtomType& type = get_atom_type(name);
        atom_masses.push_back(type.mass);
        atom_charges.push_back(type.charge);
    }
    
    if (rank == 0) {
        std::cout << "Topology loaded successfully:" << std::endl;
        std::cout << "  Atom types: " << atom_types.size() << std::endl;
        std::cout << "  Atoms: " << atom_type_names.size() << std::endl;
        std::cout << "  Bond types: " << bond_types.size() << std::endl;
        std::cout << "  Bonds: " << bonds.size() << std::endl;
        std::cout << "  Angle types: " << angle_types.size() << std::endl;
        std::cout << "  Angles: " << angles.size() << std::endl;
        std::cout << "  Dihedral types: " << dihedral_types.size() << std::endl;
        std::cout << "  Dihedrals: " << dihedrals.size() << std::endl;
    }
        
    return true;
}

bool Topology::load_canonical(const std::string& topology_filename,
                              const std::string& parameter_filename,
                              int mpi_rank) {
    rank = mpi_rank;
    atom_types.clear(); bond_types.clear(); angle_types.clear(); dihedral_types.clear();
    atom_type_names.clear(); atom_elements.clear(); atom_type_ids.clear(); atom_masses.clear(); atom_charges.clear();
    bonds.clear(); angles.clear(); dihedrals.clear(); impropers.clear(); special_pairs.clear();
    constraints.clear();
    pair_sigma.clear(); pair_epsilon.clear(); exclusion_keys.clear();
    canonical_format = true;
    mie_repulsive_exponent = 12;
    mie_attractive_exponent = 6;

    try {
        std::ifstream parameter_file(parameter_filename);
        if (!parameter_file) {
            throw std::runtime_error("cannot open parameter file: " + parameter_filename);
        }

        std::map<int, std::string> type_names_by_id;
        struct PairRow { int i, j; Real sigma, epsilon; };
        std::vector<PairRow> pair_rows;
        std::map<int, std::pair<Real, Real>> special_pair_params;
        bool mie_component_seen = false;
        std::string section;
        std::string line;
        size_t line_number = 0;
        while (std::getline(parameter_file, line)) {
            ++line_number;
            const std::string cleaned = trim(line);
            if (cleaned.empty() || cleaned[0] == ';' || cleaned[0] == '#') continue;
            if (cleaned.front() == '[' && cleaned.back() == ']') {
                section = trim(cleaned.substr(1, cleaned.size() - 2));
                std::transform(section.begin(), section.end(), section.begin(), ::tolower);
                continue;
            }
            const auto tokens = tokenize_data_line(line);
            if (tokens.empty()) continue;

            auto fail = [&](const std::string& message) {
                throw std::runtime_error(parameter_filename + ":" +
                                         std::to_string(line_number) + ": " + message);
            };

            if (section == "meta") {
                if (tokens.size() < 2) fail("metadata requires key and value");
                if (tokens[0] == "schema_version" && tokens[1] != "1") {
                    fail("unsupported schema_version '" + tokens[1] + "'");
                }
                if (tokens[0] == "length_unit" && tokens[1] != "angstrom") {
                    fail("length_unit must be angstrom");
                }
                if (tokens[0] == "energy_unit" && tokens[1] != "kcal_per_mol") {
                    fail("energy_unit must be kcal_per_mol");
                }
                if (tokens[0] == "charge_unit" && tokens[1] != "elementary_charge") {
                    fail("charge_unit must be elementary_charge");
                }
            } else if (section == "nonbonded") {
                if (tokens[0] == "component") {
                    bool mie = false;
                    int n = -1, m = -1;
                    for (const auto& token : tokens) {
                        mie = mie || token == "mie";
                        if (token.rfind("n=", 0) == 0) n = std::stoi(token.substr(2));
                        if (token.rfind("m=", 0) == 0) m = std::stoi(token.substr(2));
                    }
                    if (mie) {
                        if (mie_component_seen) fail("only one Mie nonbonded component is supported");
                        if (!((n == 12 && m == 6) || (n == 9 && m == 6))) {
                            fail("supported Mie exponents are (12,6) and SPICA (9,6)");
                        }
                        mie_repulsive_exponent = n;
                        mie_attractive_exponent = m;
                        mie_component_seen = true;
                    }
                }
            } else if (section == "atom_types") {
                if (tokens.size() < 2) fail("atom type requires: type_id name");
                const int id = std::stoi(tokens[0]);
                if (id < 0 || !type_names_by_id.emplace(id, tokens[1]).second) {
                    fail("invalid or duplicate atom type ID");
                }
            } else if (section == "pair_parameters") {
                if (tokens.size() < 5) fail("pair parameter requires: component type_i type_j sigma epsilon");
                pair_rows.push_back({std::stoi(tokens[1]), std::stoi(tokens[2]),
                                     std::stod(tokens[3]), std::stod(tokens[4])});
            } else if (section == "bond_parameters") {
                if (tokens.size() < 4 || tokens[1] != "harmonic") {
                    fail("only 'param_id harmonic k r0' bond parameters are supported");
                }
                const std::string name = "p" + tokens[0];
                const Real k = std::stod(tokens[2]);
                const Real r0 = std::stod(tokens[3]);
                if (!std::isfinite(k) || !std::isfinite(r0) || k < 0.0 || r0 <= 0.0) {
                    fail("bond k must be finite and non-negative, and r0 must be finite and positive");
                }
                if (!bond_types.emplace(name, BondType{name, k, r0}).second) {
                    fail("duplicate bond parameter ID '" + tokens[0] + "'");
                }
            } else if (section == "angle_parameters") {
                if (tokens.size() < 4 || tokens[1] != "harmonic") {
                    fail("only 'param_id harmonic k theta0_deg' angle parameters are supported");
                }
                const std::string name = "p" + tokens[0];
                const Real k = std::stod(tokens[2]);
                const Real theta = std::stod(tokens[3]) * M_PI / 180.0;
                if (!std::isfinite(k) || !std::isfinite(theta) || k < 0.0) {
                    fail("angle k and theta0 must be finite, and k must be non-negative");
                }
                if (!angle_types.emplace(name, AngleType{name, k, theta}).second) {
                    fail("duplicate angle parameter ID '" + tokens[0] + "'");
                }
            } else if (section == "torsion_parameters") {
                if (tokens.size() < 5 || tokens[1] != "periodic") {
                    fail("only 'param_id periodic k phase_deg multiplicity' torsions are supported");
                }
                const std::string name = "p" + tokens[0];
                const Real k = std::stod(tokens[2]);
                const Real phase = std::stod(tokens[3]) * M_PI / 180.0;
                const int multiplicity = std::stoi(tokens[4]);
                if (!std::isfinite(k) || !std::isfinite(phase) || multiplicity <= 0) {
                    fail("torsion k and phase must be finite, and multiplicity must be positive");
                }
                if (!dihedral_types.emplace(name, DihedralType{name, k, phase, multiplicity}).second) {
                    fail("duplicate torsion parameter ID '" + tokens[0] + "'");
                }
            } else if (section == "improper_parameters") {
                const std::string name = "ip" + tokens[0];
                if (tokens.size() < 4 || (tokens[1] != "periodic" && tokens[1] != "harmonic")) {
                    fail("improper requires either 'param_id periodic k phase_deg multiplicity' or 'param_id harmonic k phi0_deg'");
                }
                const Real k = std::stod(tokens[2]);
                const Real phase = std::stod(tokens[3]) * M_PI / 180.0;
                const int multiplicity = tokens[1] == "harmonic" ? 0 :
                    (tokens.size() >= 5 ? std::stoi(tokens[4]) : -1);
                if (!std::isfinite(k) || !std::isfinite(phase) || k < 0.0 ||
                    (tokens[1] == "periodic" && multiplicity <= 0)) {
                    fail("improper k/phase must be finite, k non-negative, and periodic multiplicity positive");
                }
                if (!dihedral_types.emplace(name, DihedralType{name, k, phase, multiplicity}).second) {
                    fail("duplicate improper parameter ID '" + tokens[0] + "'");
                }
            } else if (section == "special_pair_parameters") {
                if (tokens.size() < 3) fail("special pair parameter requires: param_id sigma epsilon");
                const int id = std::stoi(tokens[0]);
                const Real sigma = std::stod(tokens[1]);
                const Real epsilon = std::stod(tokens[2]);
                if (id < 0 || !std::isfinite(sigma) || !std::isfinite(epsilon) ||
                    sigma < 0.0 || epsilon < 0.0 ||
                    !special_pair_params.emplace(id, std::make_pair(sigma, epsilon)).second) {
                    fail("invalid or duplicate special pair parameter");
                }
            } else {
                fail("unknown parameter section [" + section + "]");
            }
        }

        if (!mie_component_seen) throw std::runtime_error("parameter file has no supported Mie nonbonded component");
        if (type_names_by_id.empty()) throw std::runtime_error("parameter file has no [atom_types]");
        const int type_count = type_names_by_id.rbegin()->first + 1;
        if (static_cast<int>(type_names_by_id.size()) != type_count) {
            throw std::runtime_error("atom type IDs must be contiguous and start at zero");
        }
        pair_sigma.assign(type_count, std::vector<Real>(type_count,
            std::numeric_limits<Real>::quiet_NaN()));
        pair_epsilon.assign(type_count, std::vector<Real>(type_count,
            std::numeric_limits<Real>::quiet_NaN()));
        for (const PairRow& row : pair_rows) {
            if (row.i < 0 || row.j < 0 || row.i >= type_count || row.j >= type_count) {
                throw std::runtime_error("pair parameter references an invalid atom type ID");
            }
            if (!std::isfinite(row.sigma) || !std::isfinite(row.epsilon) ||
                row.sigma < 0.0 || row.epsilon < 0.0) {
                throw std::runtime_error("pair sigma and epsilon must be finite and non-negative");
            }
            if (std::isfinite(pair_sigma[row.i][row.j]) || std::isfinite(pair_epsilon[row.i][row.j])) {
                throw std::runtime_error("duplicate pair parameter for type IDs " +
                                         std::to_string(row.i) + "," + std::to_string(row.j));
            }
            pair_sigma[row.i][row.j] = pair_sigma[row.j][row.i] = row.sigma;
            pair_epsilon[row.i][row.j] = pair_epsilon[row.j][row.i] = row.epsilon;
        }
        for (int i = 0; i < type_count; ++i) {
            for (int j = 0; j < type_count; ++j) {
                if (!std::isfinite(pair_sigma[i][j]) || !std::isfinite(pair_epsilon[i][j])) {
                    throw std::runtime_error("pair parameter matrix is incomplete for type IDs " +
                                             std::to_string(i) + "," + std::to_string(j));
                }
            }
            const std::string& name = type_names_by_id.at(i);
            atom_types[name] = AtomType{name, 0.0, 0.0,
                                        pair_sigma[i][i], pair_epsilon[i][i]};
        }

        std::ifstream topology_file(topology_filename);
        if (!topology_file) {
            throw std::runtime_error("cannot open topology file: " + topology_filename);
        }
        section.clear();
        line_number = 0;
        int expected_atom_count = -1;
        while (std::getline(topology_file, line)) {
            ++line_number;
            const std::string cleaned = trim(line);
            if (cleaned.empty() || cleaned[0] == ';' || cleaned[0] == '#') continue;
            if (cleaned.front() == '[' && cleaned.back() == ']') {
                section = trim(cleaned.substr(1, cleaned.size() - 2));
                std::transform(section.begin(), section.end(), section.begin(), ::tolower);
                continue;
            }
            const auto tokens = tokenize_data_line(line);
            if (tokens.empty()) continue;
            auto fail = [&](const std::string& message) {
                throw std::runtime_error(topology_filename + ":" +
                                         std::to_string(line_number) + ": " + message);
            };

            if (section == "meta") {
                if (tokens.size() < 2) fail("metadata requires key and value");
                if (tokens[0] == "schema_version" && tokens[1] != "1") {
                    fail("unsupported schema_version '" + tokens[1] + "'");
                }
                if (tokens[0] == "atom_count") {
                    expected_atom_count = std::stoi(tokens[1]);
                    if (expected_atom_count <= 0) fail("atom_count must be positive");
                }
            } else if (section == "atoms") {
                if (tokens.size() < 5) fail("atom requires: atom_id element type_id mass charge [molecule_id residue_id]");
                const int id = std::stoi(tokens[0]);
                if (id != static_cast<int>(atom_type_names.size())) {
                    fail("atom IDs must be contiguous and start at zero");
                }
                const int type_id = std::stoi(tokens[2]);
                auto type_it = type_names_by_id.find(type_id);
                if (type_it == type_names_by_id.end()) fail("atom references an invalid type ID");
                atom_type_names.push_back(type_it->second);
                atom_elements.push_back(tokens[1]);
                atom_type_ids.push_back(type_id);
                const Real mass = std::stod(tokens[3]);
                const Real charge = std::stod(tokens[4]);
                if (!std::isfinite(mass) || mass <= 0.0 || !std::isfinite(charge)) {
                    fail("atom mass must be finite and positive, and charge must be finite");
                }
                atom_masses.push_back(mass);
                atom_charges.push_back(charge);
            } else if (section == "bonds") {
                if (tokens.size() < 3) fail("bond requires: atom_i atom_j bond_param_id");
                const std::string name = "p" + tokens[2];
                if (!has_bond_type(name)) fail("bond references an invalid parameter ID");
                bonds.push_back({std::stoi(tokens[0]), std::stoi(tokens[1]), name});
            } else if (section == "angles") {
                if (tokens.size() < 4) fail("angle requires: atom_i atom_j atom_k angle_param_id");
                const std::string name = "p" + tokens[3];
                if (!has_angle_type(name)) fail("angle references an invalid parameter ID");
                angles.push_back({std::stoi(tokens[0]), std::stoi(tokens[1]),
                                  std::stoi(tokens[2]), name});
            } else if (section == "dihedrals") {
                if (tokens.size() < 5) fail("dihedral requires: atom_i atom_j atom_k atom_l torsion_param_id");
                const std::string name = "p" + tokens[4];
                if (!has_dihedral_type(name)) fail("dihedral references an invalid parameter ID");
                dihedrals.push_back({std::stoi(tokens[0]), std::stoi(tokens[1]),
                    std::stoi(tokens[2]), std::stoi(tokens[3]), name});
            } else if (section == "exclusions") {
                if (tokens.size() < 2) fail("exclusion requires: atom_i atom_j");
                exclusion_keys.push_back(pair_key(std::stoi(tokens[0]), std::stoi(tokens[1])));
            } else if (section == "impropers") {
                if (tokens.size() < 5) fail("improper requires: atom_i atom_j atom_k atom_l torsion_param_id");
                const std::string name = "ip" + tokens[4];
                if (!has_dihedral_type(name)) fail("improper references an invalid parameter ID");
                impropers.push_back({std::stoi(tokens[0]), std::stoi(tokens[1]),
                    std::stoi(tokens[2]), std::stoi(tokens[3]), name});
            } else if (section == "special_pairs") {
                if (tokens.size() < 5) fail("special_pair requires: atom_i atom_j pair_param_id lj_scale coulomb_scale");
                const int param_id = std::stoi(tokens[2]);
                Real sigma = -1.0, epsilon = -1.0;
                auto param = special_pair_params.find(param_id);
                if (param != special_pair_params.end()) {
                    sigma = param->second.first;
                    epsilon = param->second.second;
                }
                special_pairs.push_back({std::stoi(tokens[0]), std::stoi(tokens[1]),
                    std::stod(tokens[3]), std::stod(tokens[4]), sigma, epsilon});
            } else if (section == "constraints") {
                if (tokens.size() < 3) fail("constraint requires: atom_i atom_j target_distance");
                constraints.push_back({std::stoi(tokens[0]), std::stoi(tokens[1]), std::stod(tokens[2])});
            } else if (section == "virtual_sites") {
                fail("[" + section + "] is represented by the format but not supported by the current runtime");
            } else {
                fail("unknown topology section [" + section + "]");
            }
        }

        if (atom_type_names.empty()) throw std::runtime_error("topology file has no [atoms]");
        if (expected_atom_count < 0) throw std::runtime_error("topology [meta] has no atom_count");
        if (expected_atom_count != static_cast<int>(atom_type_names.size())) {
            throw std::runtime_error("topology atom_count is " + std::to_string(expected_atom_count) +
                                     " but [atoms] contains " +
                                     std::to_string(atom_type_names.size()) + " entries");
        }
        auto validate_atom = [&](int id, const char* kind) {
            if (id < 0 || id >= static_cast<int>(atom_type_names.size())) {
                throw std::runtime_error(std::string(kind) + " references invalid atom ID " + std::to_string(id));
            }
        };
        for (const Bond& b : bonds) { validate_atom(b.atom1, "bond"); validate_atom(b.atom2, "bond"); }
        for (const Angle& a : angles) { validate_atom(a.atom1, "angle"); validate_atom(a.atom2, "angle"); validate_atom(a.atom3, "angle"); }
        for (const Dihedral& d : dihedrals) { validate_atom(d.atom1, "dihedral"); validate_atom(d.atom2, "dihedral"); validate_atom(d.atom3, "dihedral"); validate_atom(d.atom4, "dihedral"); }
        for (const Dihedral& d : impropers) { validate_atom(d.atom1, "improper"); validate_atom(d.atom2, "improper"); validate_atom(d.atom3, "improper"); validate_atom(d.atom4, "improper"); }
        for (const SpecialPair& sp : special_pairs) { validate_atom(sp.atom1, "special_pair"); validate_atom(sp.atom2, "special_pair"); }
        for (const Constraint& c : constraints) {
            validate_atom(c.atom1, "constraint"); validate_atom(c.atom2, "constraint");
            if (!std::isfinite(c.distance) || c.distance <= 0.0) {
                throw std::runtime_error("constraint target_distance must be finite and positive");
            }
        }
        for (const std::uint64_t key : exclusion_keys) {
            validate_atom(static_cast<int>(key >> 32), "exclusion");
            validate_atom(static_cast<int>(key & 0xffffffffu), "exclusion");
        }

        for (const Bond& b : bonds) {
            exclusion_keys.push_back(pair_key(b.atom1, b.atom2));
        }
        for (const Angle& a : angles) {
            exclusion_keys.push_back(pair_key(a.atom1, a.atom3));
        }
        for (const Dihedral& d : dihedrals) {
            exclusion_keys.push_back(pair_key(d.atom1, d.atom4));
        }

        std::sort(exclusion_keys.begin(), exclusion_keys.end());
        exclusion_keys.erase(std::unique(exclusion_keys.begin(), exclusion_keys.end()), exclusion_keys.end());
        for (const SpecialPair& sp : special_pairs) {
            if (!std::isfinite(sp.lj_scale) || !std::isfinite(sp.coulomb_scale) ||
                sp.lj_scale < 0.0 || sp.coulomb_scale < 0.0) {
                throw std::runtime_error("special_pair scale factors must be finite and non-negative");
            }
            const bool uses_mixed_parameters = sp.sigma < 0.0 && sp.epsilon < 0.0;
            const bool has_explicit_parameters = std::isfinite(sp.sigma) &&
                std::isfinite(sp.epsilon) && sp.sigma >= 0.0 && sp.epsilon >= 0.0;
            if (!uses_mixed_parameters && !has_explicit_parameters) {
                throw std::runtime_error(
                    "special_pair sigma/epsilon must both be absent or finite and non-negative");
            }
            if (!std::binary_search(exclusion_keys.begin(), exclusion_keys.end(),
                                    pair_key(sp.atom1, sp.atom2))) {
                throw std::runtime_error("special_pair " + std::to_string(sp.atom1) + "-" +
                    std::to_string(sp.atom2) + " must also appear in [exclusions]");
            }
        }

        if (rank == 0) {
            std::cout << "dsMD_input topology loaded: " << atom_type_names.size()
                      << " atoms, " << type_count << " types, " << bonds.size()
                      << " bonds, " << angles.size() << " angles, "
                      << dihedrals.size() << " dihedrals, " << impropers.size() << " impropers, "
                      << special_pairs.size() << " special pairs, "
                      << constraints.size() << " constraints, "
                      << exclusion_keys.size() << " exclusions" << std::endl;
        }
        return true;
    } catch (const std::exception& error) {
        if (rank == 0) std::cerr << "dsMD_input topology error: " << error.what() << std::endl;
        return false;
    }
}

const AtomType& Topology::get_atom_type(const std::string& type_name) const {
    auto it = atom_types.find(type_name);
    if (it == atom_types.end()) {
        throw std::runtime_error("Atom type not found: " + type_name);
    }
    return it->second;
}

const std::string& Topology::get_atom_type_name(int atom_id) const {
    if (atom_id < 0 || static_cast<size_t>(atom_id) >= atom_type_names.size()) {
        throw std::runtime_error("Invalid atom ID: " + std::to_string(atom_id));
    }
    return atom_type_names[atom_id];
}

const std::string& Topology::get_atom_element(int atom_id) const {
    if (atom_id < 0 || static_cast<size_t>(atom_id) >= atom_elements.size()) {
        throw std::runtime_error("Invalid atom ID: " + std::to_string(atom_id));
    }
    return atom_elements[atom_id];
}

const BondType& Topology::get_bond_type(const std::string& type_name) const {
    auto it = bond_types.find(type_name);
    if (it == bond_types.end()) {
        throw std::runtime_error("Bond type not found: " + type_name);
    }
    return it->second;
}

const AngleType& Topology::get_angle_type(const std::string& type_name) const {
    auto it = angle_types.find(type_name);
    if (it == angle_types.end()) {
        throw std::runtime_error("Angle type not found: " + type_name);
    }
    return it->second;
}

const DihedralType& Topology::get_dihedral_type(const std::string& type_name) const {
    auto it = dihedral_types.find(type_name);
    if (it == dihedral_types.end()) {
        throw std::runtime_error("Dihedral type not found: " + type_name);
    }
    return it->second;
}

Real Topology::get_atom_mass(int atom_id) const {
    if (!atom_masses.empty()) {
        if (atom_id < 0 || static_cast<size_t>(atom_id) >= atom_masses.size())
            throw std::runtime_error("Invalid atom ID: " + std::to_string(atom_id));
        return atom_masses[atom_id];
    }
    const std::string& type_name = get_atom_type_name(atom_id);
    return get_atom_type(type_name).mass;
}

Real Topology::get_atom_charge(int atom_id) const {
    if (!atom_charges.empty()) {
        if (atom_id < 0 || static_cast<size_t>(atom_id) >= atom_charges.size())
            throw std::runtime_error("Invalid atom ID: " + std::to_string(atom_id));
        return atom_charges[atom_id];
    }
    const std::string& type_name = get_atom_type_name(atom_id);
    return get_atom_type(type_name).charge;
}

Real Topology::get_atom_sigma(int atom_id) const {
    const std::string& type_name = get_atom_type_name(atom_id);
    return get_atom_type(type_name).sigma;
}

Real Topology::get_atom_epsilon(int atom_id) const {
    const std::string& type_name = get_atom_type_name(atom_id);
    return get_atom_type(type_name).epsilon;
}

int Topology::get_atom_type_id(int atom_id) const {
    if (atom_id < 0 || static_cast<size_t>(atom_id) >= atom_type_ids.size()) {
        throw std::runtime_error("Invalid atom ID: " + std::to_string(atom_id));
    }
    return atom_type_ids[atom_id];
}

bool Topology::is_nonbonded_excluded(int atom1, int atom2) const {
    return std::binary_search(exclusion_keys.begin(), exclusion_keys.end(), pair_key(atom1, atom2));
}

bool Topology::has_atom_type(const std::string& type_name) const {
    return atom_types.find(type_name) != atom_types.end();
}

bool Topology::has_bond_type(const std::string& type_name) const {
    return bond_types.find(type_name) != bond_types.end();
}

bool Topology::has_angle_type(const std::string& type_name) const {
    return angle_types.find(type_name) != angle_types.end();
}

bool Topology::has_dihedral_type(const std::string& type_name) const {
    return dihedral_types.find(type_name) != dihedral_types.end();
}
