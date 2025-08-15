#ifndef TOPOLOGY_H
#define TOPOLOGY_H

#include "md_types.h"
#include <vector>
#include <string>
#include <map>
#include <unordered_map>

// Atom type information
struct AtomType {
    std::string name;
    Real mass;
    Real charge;
    Real sigma;    // LJ sigma parameter
    Real epsilon;  // LJ epsilon parameter
};

// Bond type information
struct BondType {
    std::string name;
    Real k;        // Force constant
    Real r0;       // Equilibrium bond length
};

// Angle type information
struct AngleType {
    std::string name;
    Real k;        // Force constant
    Real theta0;   // Equilibrium angle (radians)
};

// Dihedral type information
struct DihedralType {
    std::string name;
    Real k;        // Force constant
    Real phi0;     // Phase shift (radians)
    int n;         // Multiplicity
};

// Bond connection
struct Bond {
    int atom1, atom2;
    std::string type;
};

// Angle connection
struct Angle {
    int atom1, atom2, atom3;  // atom2 is the center
    std::string type;
};

// Dihedral connection
struct Dihedral {
    int atom1, atom2, atom3, atom4;
    std::string type;
};

// Topology class to manage all force field parameters
class Topology {
private:
    std::map<std::string, AtomType> atom_types;
    std::map<std::string, BondType> bond_types;
    std::map<std::string, AngleType> angle_types;
    std::map<std::string, DihedralType> dihedral_types;
    
    std::vector<std::string> atom_type_names;  // atom_type_names[i] = type of atom i
    std::vector<Bond> bonds;
    std::vector<Angle> angles;
    std::vector<Dihedral> dihedrals;
    
    // MPI rank for output control
    int rank;
    
public:
    Topology();
    ~Topology();
    
    // Load topology from file
    bool load_topology(const std::string& filename, int mpi_rank = 0);
    
    // Getters for atom types
    const AtomType& get_atom_type(const std::string& type_name) const;
    const std::string& get_atom_type_name(int atom_id) const;
    
    // Getters for bond types
    const BondType& get_bond_type(const std::string& type_name) const;
    
    // Getters for angle types
    const AngleType& get_angle_type(const std::string& type_name) const;
    
    // Getters for dihedral types
    const DihedralType& get_dihedral_type(const std::string& type_name) const;
    
    // Getters for connections
    const std::vector<Bond>& get_bonds() const { return bonds; }
    const std::vector<Angle>& get_angles() const { return angles; }
    const std::vector<Dihedral>& get_dihedrals() const { return dihedrals; }
    
    // Get atom properties by atom ID
    Real get_atom_mass(int atom_id) const;
    Real get_atom_charge(int atom_id) const;
    Real get_atom_sigma(int atom_id) const;
    Real get_atom_epsilon(int atom_id) const;
    
    // Utility functions
    int get_num_atoms() const { return atom_type_names.size(); }
    bool has_atom_type(const std::string& type_name) const;
    bool has_bond_type(const std::string& type_name) const;
    bool has_angle_type(const std::string& type_name) const;
    bool has_dihedral_type(const std::string& type_name) const;
};

#endif // TOPOLOGY_H
