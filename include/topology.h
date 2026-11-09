#ifndef TOPOLOGY_H
#define TOPOLOGY_H

#include "md_types.h"
#include <vector>
#include <string>
#include <map>
#include <mpi.h>
#include <unordered_map>
#include <cstdint>

struct AtomType {
    std::string name;
    Real mass;
    Real charge;
    Real sigma;    // LJ sigma parameter
    Real epsilon;  // LJ epsilon parameter
};

struct BondType {
    std::string name;
    Real k;        // Force constant
    Real r0;       // Equilibrium bond length
};

struct AngleType {
    std::string name;
    Real k;        // Force constant
    Real theta0;   // Equilibrium angle (radians)
};

struct DihedralType {
    std::string name;
    Real k;        // Force constant
    Real phi0;     // Phase shift (radians)
    int n;         // Multiplicity
};

// The type of a bond, angle or dihedral is the id of its parameter name in the matching StringTable of the Topology
// (an int instead of a std::string: 12 instead of 40 bytes per bond, and no string copies to broadcast).
struct Bond {
    int atom1, atom2;
    int type;
};

struct Angle {
    int atom1, atom2, atom3;  // atom2 is the center
    int type;
};

struct Dihedral {
    int atom1, atom2, atom3, atom4;
    int type;
};

struct SpecialPair {
    int atom1, atom2;
    Real lj_scale, coulomb_scale;
    Real sigma = -1.0, epsilon = -1.0;
};

struct Constraint {
    int atom1, atom2;
    Real distance;
};

// Distinct strings with a stable id each: the names the atoms and interactions refer to are stored once.
struct StringTable {
    std::vector<std::string> names;
    std::unordered_map<std::string, int> index;
    int intern(const std::string& value) {
        const auto found = index.find(value);
        if (found != index.end()) return found->second;
        const int id = static_cast<int>(names.size());
        names.push_back(value);
        index.emplace(value, id);
        return id;
    }
    const std::string& name(int id) const { return names[static_cast<size_t>(id)]; }
    size_t size() const { return names.size(); }
    void clear() { names.clear(); index.clear(); }
    // The names of a table that arrived over MPI (no index: nothing is added to it any more).
    void assign(std::vector<std::string> values) {
        names = std::move(values);
        index.clear();
    }
    // The index is only needed while a file is read; afterwards the names alone are kept.
    void release_index() {
        std::unordered_map<std::string, int>().swap(index);
    }
};

class Topology {
private:
    std::map<std::string, AtomType> atom_types;
    std::map<std::string, BondType> bond_types;
    std::map<std::string, AngleType> angle_types;
    std::map<std::string, DihedralType> dihedral_types;
    
    // Per atom: the id of its type name and of its element symbol in the two tables (not one std::string per atom).
    std::vector<int> atom_name_ids;
    std::vector<int> atom_element_ids;
    StringTable atom_name_table;
    StringTable atom_element_table;
    // Parameter names that bonds, angles and dihedrals (impropers too) refer to by id.
    StringTable bond_type_table;
    StringTable angle_type_table;
    StringTable dihedral_type_table;
    // The parameters of every id, found once after loading (nullptr: the name has no parameters, the lookup by name throws).
    std::vector<const BondType*> bond_by_id;
    std::vector<const AngleType*> angle_by_id;
    std::vector<const DihedralType*> dihedral_by_id;
    std::vector<int> atom_type_ids;
    std::vector<Real> atom_masses;
    std::vector<Real> atom_charges;
    std::vector<Bond> bonds;
    std::vector<Angle> angles;
    std::vector<Dihedral> dihedrals;
    std::vector<Dihedral> impropers;        // Same functional form/types as dihedrals
    std::vector<SpecialPair> special_pairs; // Scaled 1-4 nonbonded pairs
    std::vector<Constraint> constraints;    // Rigid-bond distance constraints (SHAKE/RATTLE)

    std::vector<std::vector<Real>> pair_sigma;
    std::vector<std::vector<Real>> pair_epsilon;
    std::vector<std::uint64_t> exclusion_keys;
    int mie_repulsive_exponent = 12;
    int mie_attractive_exponent = 6;
    bool canonical_format = false;
    
    int rank;

    void resolve_type_ids();
    
public:
    Topology();
    ~Topology();
    
    bool load_topology(const std::string& filename, int mpi_rank = 0);
    bool load_canonical(const std::string& topology_filename,
                        const std::string& parameter_filename,
                        int mpi_rank = 0);
    void broadcast(MPI_Comm comm = MPI_COMM_WORLD, int root = 0);
    
    const AtomType& get_atom_type(const std::string& type_name) const;
    const std::string& get_atom_type_name(int atom_id) const;
    const std::string& get_atom_element(int atom_id) const;
    
    const BondType& get_bond_type(const std::string& type_name) const;
    
    const AngleType& get_angle_type(const std::string& type_name) const;
    
    const DihedralType& get_dihedral_type(const std::string& type_name) const;
    // The same by the id a Bond, Angle or Dihedral carries as its type.
    const BondType& get_bond_type(int type_id) const;
    const AngleType& get_angle_type(int type_id) const;
    const DihedralType& get_dihedral_type(int type_id) const;
    const std::string& bond_type_name(int type_id) const { return bond_type_table.name(type_id); }
    const std::string& angle_type_name(int type_id) const { return angle_type_table.name(type_id); }
    const std::string& dihedral_type_name(int type_id) const { return dihedral_type_table.name(type_id); }
    
    const std::vector<Bond>& get_bonds() const { return bonds; }
    const std::vector<Angle>& get_angles() const { return angles; }
    const std::vector<Dihedral>& get_dihedrals() const { return dihedrals; }
    const std::vector<Dihedral>& get_impropers() const { return impropers; }
    const std::vector<SpecialPair>& get_special_pairs() const { return special_pairs; }
    const std::vector<Constraint>& get_constraints() const { return constraints; }
    
    Real get_atom_mass(int atom_id) const;
    Real get_atom_charge(int atom_id) const;
    Real get_atom_sigma(int atom_id) const;
    Real get_atom_epsilon(int atom_id) const;
    int get_atom_type_id(int atom_id) const;

    bool is_canonical() const { return canonical_format; }
    bool has_explicit_pair_parameters() const { return !pair_sigma.empty(); }
    const std::vector<std::vector<Real>>& get_pair_sigma() const { return pair_sigma; }
    const std::vector<std::vector<Real>>& get_pair_epsilon() const { return pair_epsilon; }
    const std::vector<std::uint64_t>& get_exclusion_keys() const { return exclusion_keys; }
    int get_mie_repulsive_exponent() const { return mie_repulsive_exponent; }
    int get_mie_attractive_exponent() const { return mie_attractive_exponent; }
    bool is_nonbonded_excluded(int atom1, int atom2) const;
    
    int get_num_atoms() const { return static_cast<int>(atom_name_ids.size()); }
    bool has_atom_type(const std::string& type_name) const;
    bool has_bond_type(const std::string& type_name) const;
    bool has_angle_type(const std::string& type_name) const;
    bool has_dihedral_type(const std::string& type_name) const;
};

#endif // TOPOLOGY_H
