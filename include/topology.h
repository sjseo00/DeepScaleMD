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

struct Bond {
    int atom1, atom2;
    std::string type;
};

struct Angle {
    int atom1, atom2, atom3;  // atom2 is the center
    std::string type;
};

struct Dihedral {
    int atom1, atom2, atom3, atom4;
    std::string type;
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

class Topology {
private:
    std::map<std::string, AtomType> atom_types;
    std::map<std::string, BondType> bond_types;
    std::map<std::string, AngleType> angle_types;
    std::map<std::string, DihedralType> dihedral_types;
    
    std::vector<std::string> atom_type_names;  // atom_type_names[i] = type of atom i
    std::vector<std::string> atom_elements;
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
    
    int get_num_atoms() const { return atom_type_names.size(); }
    bool has_atom_type(const std::string& type_name) const;
    bool has_bond_type(const std::string& type_name) const;
    bool has_angle_type(const std::string& type_name) const;
    bool has_dihedral_type(const std::string& type_name) const;
};

#endif // TOPOLOGY_H
