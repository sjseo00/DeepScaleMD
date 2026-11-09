
#include "topology.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
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

// One broadcast of a large buffer. MPI counts are int, so a buffer of more than 1 GiB goes in pieces.
void broadcast_bytes(void* data, std::uint64_t bytes, MPI_Comm comm, int root) {
    char* cursor = static_cast<char*>(data);
    const std::uint64_t piece_limit = std::uint64_t(1) << 30;
    while (bytes != 0) {
        const int piece = static_cast<int>(std::min(bytes, piece_limit));
        MPI_Bcast(cursor, piece, MPI_BYTE, root, comm);
        cursor += piece;
        bytes -= static_cast<std::uint64_t>(piece);
    }
}

std::uint64_t broadcast_size(size_t local_size, MPI_Comm comm, int root, int rank) {
    std::uint64_t count = (rank == root) ? static_cast<std::uint64_t>(local_size) : 0;
    MPI_Bcast(&count, 1, MPI_UINT64_T, root, comm);
    return count;
}

// All strings of a list in one buffer (each ends with '\0'); the list already has its final length on every rank.
void broadcast_strings(std::vector<std::string>& values, MPI_Comm comm, int root, int rank) {
    std::string joined;
    if (rank == root) {
        for (const auto& value : values) { joined += value; joined += '\0'; }
    }
    std::uint64_t size = broadcast_size(joined.size(), comm, root, rank);
    if (rank != root) joined.resize(static_cast<size_t>(size));
    broadcast_bytes(&joined[0], size, comm, root);
    if (rank != root) {
        size_t begin = 0;
        for (auto& value : values) {
            const size_t end = joined.find('\0', begin);
            value.assign(joined, begin, end - begin);
            begin = end + 1;
        }
    }
}

// A table of names: the count, then all names in one buffer.
void broadcast_table(StringTable& table, MPI_Comm comm, int root, int rank) {
    std::vector<std::string> names = table.names;
    names.resize(static_cast<size_t>(broadcast_size(names.size(), comm, root, rank)));
    broadcast_strings(names, comm, root, rank);
    if (rank != root) table.assign(std::move(names));
}

// Bonds, angles, dihedrals and impropers: the atom indices of every record and the id of its type go in one int array
// (a record is `atom_fields` indices followed by the type id). A call per record took about 10 us each, i.e. minutes
// for a system of millions of atoms.
template <class Record>
void broadcast_records(std::vector<Record>& records, std::initializer_list<int Record::*> atom_fields,
                       MPI_Comm comm, int root, int rank) {
    const size_t count = static_cast<size_t>(broadcast_size(records.size(), comm, root, rank));
    if (rank != root) records.resize(count);
    const size_t stride = atom_fields.size() + 1;
    std::vector<int> packed(count * stride);
    if (rank == root) {
        for (size_t i = 0; i < count; ++i) {
            size_t k = i * stride;
            for (auto field : atom_fields) packed[k++] = records[i].*field;
            packed[k] = records[i].type;
        }
    }
    broadcast_bytes(packed.data(), static_cast<std::uint64_t>(packed.size()) * sizeof(int), comm, root);
    if (rank != root) {
        for (size_t i = 0; i < count; ++i) {
            size_t k = i * stride;
            for (auto field : atom_fields) records[i].*field = packed[k++];
            records[i].type = packed[k];
        }
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

    // Bond, angle and dihedral parameters: a package has one entry per interaction, so the names go in one string
    // buffer and the values in one array each (a call per entry took minutes for a system of millions of atoms).
    {
        std::vector<std::string> names;
        std::vector<Real> values;
        if (rank == root) {
            for (const auto& entry : bond_types) {
                names.push_back(entry.first);
                values.push_back(entry.second.k);
                values.push_back(entry.second.r0);
            }
        }
        names.resize(static_cast<size_t>(broadcast_size(names.size(), comm, root, rank)));
        values.resize(names.size() * 2);
        broadcast_strings(names, comm, root, rank);
        broadcast_bytes(values.data(), values.size() * sizeof(Real), comm, root);
        if (rank != root) {
            bond_types.clear();
            for (size_t i = 0; i < names.size(); ++i) {
                bond_types[names[i]] = BondType{names[i], values[2 * i], values[2 * i + 1]};
            }
        }
    }

    {
        std::vector<std::string> names;
        std::vector<Real> values;
        if (rank == root) {
            for (const auto& entry : angle_types) {
                names.push_back(entry.first);
                values.push_back(entry.second.k);
                values.push_back(entry.second.theta0);
            }
        }
        names.resize(static_cast<size_t>(broadcast_size(names.size(), comm, root, rank)));
        values.resize(names.size() * 2);
        broadcast_strings(names, comm, root, rank);
        broadcast_bytes(values.data(), values.size() * sizeof(Real), comm, root);
        if (rank != root) {
            angle_types.clear();
            for (size_t i = 0; i < names.size(); ++i) {
                angle_types[names[i]] = AngleType{names[i], values[2 * i], values[2 * i + 1]};
            }
        }
    }

    {
        std::vector<std::string> names;
        std::vector<Real> values;
        std::vector<int> multiplicities;
        if (rank == root) {
            for (const auto& entry : dihedral_types) {
                names.push_back(entry.first);
                values.push_back(entry.second.k);
                values.push_back(entry.second.phi0);
                multiplicities.push_back(entry.second.n);
            }
        }
        names.resize(static_cast<size_t>(broadcast_size(names.size(), comm, root, rank)));
        values.resize(names.size() * 2);
        multiplicities.resize(names.size());
        broadcast_strings(names, comm, root, rank);
        broadcast_bytes(values.data(), values.size() * sizeof(Real), comm, root);
        broadcast_bytes(multiplicities.data(), multiplicities.size() * sizeof(int), comm, root);
        if (rank != root) {
            dihedral_types.clear();
            for (size_t i = 0; i < names.size(); ++i) {
                dihedral_types[names[i]] = DihedralType{names[i], values[2 * i], values[2 * i + 1], multiplicities[i]};
            }
        }
    }

    const size_t atom_count = broadcast_count(atom_name_ids.size());
    if (rank != root) {
        atom_name_ids.resize(atom_count);
        atom_element_ids.resize(atom_count);
        atom_type_ids.resize(atom_count);
        atom_masses.resize(atom_count);
        atom_charges.resize(atom_count);
    }
    broadcast_table(atom_name_table, comm, root, rank);
    broadcast_table(atom_element_table, comm, root, rank);
    broadcast_bytes(atom_name_ids.data(), static_cast<std::uint64_t>(atom_count) * sizeof(int), comm, root);
    broadcast_bytes(atom_element_ids.data(), static_cast<std::uint64_t>(atom_count) * sizeof(int), comm, root);
    if (atom_count != 0) {
        MPI_Bcast(atom_type_ids.data(), static_cast<int>(atom_count), MPI_INT, root, comm);
        MPI_Bcast(atom_masses.data(), static_cast<int>(atom_count), MPI_DOUBLE, root, comm);
        MPI_Bcast(atom_charges.data(), static_cast<int>(atom_count), MPI_DOUBLE, root, comm);
    }

    broadcast_table(bond_type_table, comm, root, rank);
    broadcast_table(angle_type_table, comm, root, rank);
    broadcast_table(dihedral_type_table, comm, root, rank);
    broadcast_records(bonds, {&Bond::atom1, &Bond::atom2}, comm, root, rank);
    broadcast_records(angles, {&Angle::atom1, &Angle::atom2, &Angle::atom3}, comm, root, rank);
    broadcast_records(dihedrals, {&Dihedral::atom1, &Dihedral::atom2, &Dihedral::atom3, &Dihedral::atom4}, comm, root, rank);
    broadcast_records(impropers, {&Dihedral::atom1, &Dihedral::atom2, &Dihedral::atom3, &Dihedral::atom4}, comm, root, rank);

    {
        const size_t special_pair_count = broadcast_count(special_pairs.size());
        if (rank != root) special_pairs.resize(special_pair_count);
        std::vector<int> atoms(special_pair_count * 2);
        std::vector<Real> scales(special_pair_count * 4);
        if (rank == root) {
            for (size_t i = 0; i < special_pair_count; ++i) {
                const SpecialPair& pair = special_pairs[i];
                atoms[2 * i] = pair.atom1; atoms[2 * i + 1] = pair.atom2;
                scales[4 * i] = pair.lj_scale; scales[4 * i + 1] = pair.coulomb_scale;
                scales[4 * i + 2] = pair.sigma; scales[4 * i + 3] = pair.epsilon;
            }
        }
        broadcast_bytes(atoms.data(), atoms.size() * sizeof(int), comm, root);
        broadcast_bytes(scales.data(), scales.size() * sizeof(Real), comm, root);
        if (rank != root) {
            for (size_t i = 0; i < special_pair_count; ++i) {
                SpecialPair& pair = special_pairs[i];
                pair.atom1 = atoms[2 * i]; pair.atom2 = atoms[2 * i + 1];
                pair.lj_scale = scales[4 * i]; pair.coulomb_scale = scales[4 * i + 1];
                pair.sigma = scales[4 * i + 2]; pair.epsilon = scales[4 * i + 3];
            }
        }
    }

    {
        const size_t constraint_count = broadcast_count(constraints.size());
        if (rank != root) constraints.resize(constraint_count);
        std::vector<int> atoms(constraint_count * 2);
        std::vector<Real> distances(constraint_count);
        if (rank == root) {
            for (size_t i = 0; i < constraint_count; ++i) {
                atoms[2 * i] = constraints[i].atom1; atoms[2 * i + 1] = constraints[i].atom2;
                distances[i] = constraints[i].distance;
            }
        }
        broadcast_bytes(atoms.data(), atoms.size() * sizeof(int), comm, root);
        broadcast_bytes(distances.data(), distances.size() * sizeof(Real), comm, root);
        if (rank != root) {
            for (size_t i = 0; i < constraint_count; ++i) {
                constraints[i].atom1 = atoms[2 * i]; constraints[i].atom2 = atoms[2 * i + 1];
                constraints[i].distance = distances[i];
            }
        }
    }

    resolve_type_ids();

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
    atom_name_ids.clear(); atom_element_ids.clear(); atom_name_table.clear(); atom_element_table.clear();
    bond_type_table.clear(); angle_type_table.clear(); dihedral_type_table.clear();
    bond_by_id.clear(); angle_by_id.clear(); dihedral_by_id.clear();
    atom_type_ids.clear(); atom_masses.clear(); atom_charges.clear();
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
        if (!line.empty() && line.back() == '\r') line.pop_back();    // a topology file written on Windows has CRLF line ends
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
                
                if (static_cast<size_t>(atom_id) >= atom_name_ids.size()) {
                    atom_name_ids.resize(atom_id + 1, atom_name_table.intern(""));
                }
                atom_name_ids[atom_id] = atom_name_table.intern(type_name);
            }
        }
        else if (current_section == "bonds") {
            if (tokens.size() >= 3) {
                Bond bond;
                bond.atom1 = std::stoi(tokens[0]) - 1;  // Convert to 0-based indexing
                bond.atom2 = std::stoi(tokens[1]) - 1;
                bond.type = bond_type_table.intern(tokens[2]);
                bonds.push_back(bond);
            }
        }
        else if (current_section == "angles") {
            if (tokens.size() >= 4) {
                Angle angle;
                angle.atom1 = std::stoi(tokens[0]) - 1;  // Convert to 0-based indexing
                angle.atom2 = std::stoi(tokens[1]) - 1;
                angle.atom3 = std::stoi(tokens[2]) - 1;
                angle.type = angle_type_table.intern(tokens[3]);
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
                dihedral.type = dihedral_type_table.intern(tokens[4]);
                dihedrals.push_back(dihedral);
            }
        }
    }

    std::map<std::string, int> legacy_type_ids;
    for (size_t i = 0; i < atom_name_ids.size(); ++i) {
        const std::string& name = atom_name_table.name(atom_name_ids[i]);
        auto inserted = legacy_type_ids.emplace(name, static_cast<int>(legacy_type_ids.size()));
        atom_type_ids.push_back(inserted.first->second);
        atom_element_ids.push_back(atom_element_table.intern(""));
        const AtomType& type = get_atom_type(name);
        atom_masses.push_back(type.mass);
        atom_charges.push_back(type.charge);
    }
    
    if (rank == 0) {
        std::cout << "Topology loaded successfully:" << std::endl;
        std::cout << "  Atom types: " << atom_types.size() << std::endl;
        std::cout << "  Atoms: " << atom_name_ids.size() << std::endl;
        std::cout << "  Bond types: " << bond_types.size() << std::endl;
        std::cout << "  Bonds: " << bonds.size() << std::endl;
        std::cout << "  Angle types: " << angle_types.size() << std::endl;
        std::cout << "  Angles: " << angles.size() << std::endl;
        std::cout << "  Dihedral types: " << dihedral_types.size() << std::endl;
        std::cout << "  Dihedrals: " << dihedrals.size() << std::endl;
    }
    resolve_type_ids();
        
    return true;
}

bool Topology::load_canonical(const std::string& topology_filename,
                              const std::string& parameter_filename,
                              int mpi_rank) {
    rank = mpi_rank;
    atom_types.clear(); bond_types.clear(); angle_types.clear(); dihedral_types.clear();
    atom_name_ids.clear(); atom_element_ids.clear(); atom_name_table.clear(); atom_element_table.clear();
    bond_type_table.clear(); angle_type_table.clear(); dihedral_type_table.clear();
    bond_by_id.clear(); angle_by_id.clear(); dihedral_by_id.clear();
    atom_type_ids.clear(); atom_masses.clear(); atom_charges.clear();
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
                if (id != static_cast<int>(atom_name_ids.size())) {
                    fail("atom IDs must be contiguous and start at zero");
                }
                const int type_id = std::stoi(tokens[2]);
                auto type_it = type_names_by_id.find(type_id);
                if (type_it == type_names_by_id.end()) fail("atom references an invalid type ID");
                atom_name_ids.push_back(atom_name_table.intern(type_it->second));
                atom_element_ids.push_back(atom_element_table.intern(tokens[1]));
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
                bonds.push_back({std::stoi(tokens[0]), std::stoi(tokens[1]), bond_type_table.intern(name)});
            } else if (section == "angles") {
                if (tokens.size() < 4) fail("angle requires: atom_i atom_j atom_k angle_param_id");
                const std::string name = "p" + tokens[3];
                if (!has_angle_type(name)) fail("angle references an invalid parameter ID");
                angles.push_back({std::stoi(tokens[0]), std::stoi(tokens[1]),
                                  std::stoi(tokens[2]), angle_type_table.intern(name)});
            } else if (section == "dihedrals") {
                if (tokens.size() < 5) fail("dihedral requires: atom_i atom_j atom_k atom_l torsion_param_id");
                const std::string name = "p" + tokens[4];
                if (!has_dihedral_type(name)) fail("dihedral references an invalid parameter ID");
                dihedrals.push_back({std::stoi(tokens[0]), std::stoi(tokens[1]),
                    std::stoi(tokens[2]), std::stoi(tokens[3]), dihedral_type_table.intern(name)});
            } else if (section == "exclusions") {
                if (tokens.size() < 2) fail("exclusion requires: atom_i atom_j");
                exclusion_keys.push_back(pair_key(std::stoi(tokens[0]), std::stoi(tokens[1])));
            } else if (section == "impropers") {
                if (tokens.size() < 5) fail("improper requires: atom_i atom_j atom_k atom_l torsion_param_id");
                const std::string name = "ip" + tokens[4];
                if (!has_dihedral_type(name)) fail("improper references an invalid parameter ID");
                impropers.push_back({std::stoi(tokens[0]), std::stoi(tokens[1]),
                    std::stoi(tokens[2]), std::stoi(tokens[3]), dihedral_type_table.intern(name)});
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

        if (atom_name_ids.empty()) throw std::runtime_error("topology file has no [atoms]");
        if (expected_atom_count < 0) throw std::runtime_error("topology [meta] has no atom_count");
        if (expected_atom_count != static_cast<int>(atom_name_ids.size())) {
            throw std::runtime_error("topology atom_count is " + std::to_string(expected_atom_count) +
                                     " but [atoms] contains " +
                                     std::to_string(atom_name_ids.size()) + " entries");
        }
        auto validate_atom = [&](int id, const char* kind) {
            if (id < 0 || id >= static_cast<int>(atom_name_ids.size())) {
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
            std::cout << "dsMD_input topology loaded: " << atom_name_ids.size()
                      << " atoms, " << type_count << " types, " << bonds.size()
                      << " bonds, " << angles.size() << " angles, "
                      << dihedrals.size() << " dihedrals, " << impropers.size() << " impropers, "
                      << special_pairs.size() << " special pairs, "
                      << constraints.size() << " constraints, "
                      << exclusion_keys.size() << " exclusions" << std::endl;
        }
        resolve_type_ids();
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
    if (atom_id < 0 || static_cast<size_t>(atom_id) >= atom_name_ids.size()) {
        throw std::runtime_error("Invalid atom ID: " + std::to_string(atom_id));
    }
    return atom_name_table.name(atom_name_ids[atom_id]);
}

const std::string& Topology::get_atom_element(int atom_id) const {
    if (atom_id < 0 || static_cast<size_t>(atom_id) >= atom_element_ids.size()) {
        throw std::runtime_error("Invalid atom ID: " + std::to_string(atom_id));
    }
    return atom_element_table.name(atom_element_ids[atom_id]);
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

void Topology::resolve_type_ids() {
    atom_name_table.release_index();
    atom_element_table.release_index();
    bond_type_table.release_index();
    angle_type_table.release_index();
    dihedral_type_table.release_index();
    auto resolve = [](const StringTable& table, const auto& parameters, auto& by_id) {
        by_id.assign(table.size(), nullptr);
        for (size_t id = 0; id < table.size(); ++id) {
            const auto found = parameters.find(table.names[id]);
            if (found != parameters.end()) by_id[id] = &found->second;
        }
    };
    resolve(bond_type_table, bond_types, bond_by_id);
    resolve(angle_type_table, angle_types, angle_by_id);
    resolve(dihedral_type_table, dihedral_types, dihedral_by_id);
}

const BondType& Topology::get_bond_type(int type_id) const {
    if (type_id >= 0 && static_cast<size_t>(type_id) < bond_by_id.size() && bond_by_id[type_id]) return *bond_by_id[type_id];
    return get_bond_type(bond_type_table.name(type_id));   // throws the error of the unknown name
}

const AngleType& Topology::get_angle_type(int type_id) const {
    if (type_id >= 0 && static_cast<size_t>(type_id) < angle_by_id.size() && angle_by_id[type_id]) return *angle_by_id[type_id];
    return get_angle_type(angle_type_table.name(type_id));
}

const DihedralType& Topology::get_dihedral_type(int type_id) const {
    if (type_id >= 0 && static_cast<size_t>(type_id) < dihedral_by_id.size() && dihedral_by_id[type_id]) return *dihedral_by_id[type_id];
    return get_dihedral_type(dihedral_type_table.name(type_id));
}
