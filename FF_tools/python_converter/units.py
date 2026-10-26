"""
units.py - Unit conversions from source-engine units to the DeepScaleMD
dsMD_input units (angstrom, kcal/mol, g/mol, elementary charge).

GROMACS uses nm / kJ-per-mol / ps internally. Angles in .itp force-field
files are already written in degrees (not radians), so no angle-value
conversion is needed -- only the force constants carry rad^-2 dimensions.
"""

NM_TO_ANGSTROM = 10.0
ANGSTROM_TO_NM = 1.0 / NM_TO_ANGSTROM

KJ_PER_MOL_TO_KCAL_PER_MOL = 1.0 / 4.184
KCAL_PER_MOL_TO_KJ_PER_MOL = 4.184


def gmx_length_to_angstrom(nm: float) -> float:
    return nm * NM_TO_ANGSTROM


def gmx_energy_to_kcal(kj_per_mol: float) -> float:
    return kj_per_mol * KJ_PER_MOL_TO_KCAL_PER_MOL


def gmx_bond_k_to_kcal_per_ang2(kj_per_mol_per_nm2: float) -> float:
    """kJ/mol/nm^2 -> kcal/mol/Angstrom^2."""
    return kj_per_mol_per_nm2 * KJ_PER_MOL_TO_KCAL_PER_MOL / (NM_TO_ANGSTROM ** 2)


def gmx_angle_k_to_kcal_per_rad2(kj_per_mol_per_rad2: float) -> float:
    """kJ/mol/rad^2 -> kcal/mol/rad^2 (radians are dimensionless, so only
    the energy unit changes)."""
    return kj_per_mol_per_rad2 * KJ_PER_MOL_TO_KCAL_PER_MOL


def gmx_dihedral_k_to_kcal(kj_per_mol: float) -> float:
    return kj_per_mol * KJ_PER_MOL_TO_KCAL_PER_MOL
