#include "topology.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#ifdef USE_MPI
#include <mpi.h>
#endif

// External debug level variable
extern int g_debug_level;
inline bool debug_level(int level) { return g_debug_level >= level; }

Topology::Topology() : rank(0) {}

Topology::~Topology() {}

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
        std::cerr << "Error: Cannot open topology file: " << clean_filename << std::endl;
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
