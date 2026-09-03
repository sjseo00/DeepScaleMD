
#include "topology.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdint>
#ifdef USE_MPI
#include <mpi.h>
#endif

Topology::Topology() : rank(0) {}

Topology::~Topology() {}

namespace {

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
    if (rank != root) atom_type_names.resize(atom_count);
    for (size_t i = 0; i < atom_count; ++i) {
        broadcast_string(atom_type_names[i], comm, root, rank);
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
}

bool Topology::load_topology(const std::string& filename, int mpi_rank) {
    rank = mpi_rank;  // Store the rank for use throughout the class
    // Clean the filename - remove quotes and trim whitespace/control characters
    std::string clean_filename = filename;
    
    // First, trim whitespace and control characters from both ends
    auto start = clean_filename.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        clean_filename = "";
    } else {
        auto end = clean_filename.find_last_not_of(" \t\r\n");
        clean_filename = clean_filename.substr(start, end - start + 1);
    }
    
    // Then remove quotes if present
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
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#' || line[0] == ';') continue;
        
        // Remove leading/trailing whitespace
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);
        if (line.empty()) continue;
        
        // Check for section headers
        if (line[0] == '[' && line.back() == ']') {
            current_section = line.substr(1, line.length() - 2);
            
            // Remove leading/trailing whitespace from section name
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
        
        // Parse different sections
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
            if (tokens.size() >= 3) {
                BondType bond_type;
                bond_type.name = tokens[0] + "-" + tokens[1];  // e.g., "C-C"
                bond_type.k = std::stod(tokens[2]);
                bond_type.r0 = std::stod(tokens[3]);
                bond_types[bond_type.name] = bond_type;
            }
        }
        else if (current_section == "angletypes") {
            if (tokens.size() >= 4) {
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
                
                // Resize vector if necessary
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
    const std::string& type_name = get_atom_type_name(atom_id);
    return get_atom_type(type_name).mass;
}

Real Topology::get_atom_charge(int atom_id) const {
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
