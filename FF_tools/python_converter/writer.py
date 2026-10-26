"""
writer.py - Serializes a CanonicalSystem into dsMD_input:

    <out_dir>/
      manifest.yaml
      coordinates.xyz
      topology.dsMDtop
      parameters.dsMDprm

The section layout, column order, and defaults here are the single source
of truth for the writer side of the format; they must stay in lockstep with
the C++ reader (Topology::load_canonical in src/topology.cpp) and with
docs/06_dsMD_input_Format.md. If you change one, change all three.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from typing import TextIO

from canonical_model import CanonicalSystem
from progress import stage

SCHEMA_VERSION = 1
CONVERTER_VERSION = "0.1.0"


def _sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_coordinates(system: CanonicalSystem, path: str) -> None:
    with open(path, "w", newline="\n") as f:
        f.write(f"{len(system.atoms)}\n")
        f.write("dsMD_atom_order=global_id; length_unit=angstrom\n")
        for atom in system.atoms:
            f.write(f"{atom.element:<3s} {atom.x:14.8f} {atom.y:14.8f} {atom.z:14.8f}\n")


def _write_topology(system: CanonicalSystem, path: str) -> None:
    with open(path, "w", newline="\n") as f:
        f.write("[meta]\n")
        f.write(f"schema_version {SCHEMA_VERSION}\n")
        f.write(f"atom_count {len(system.atoms)}\n")
        f.write("resolution all_atom\n\n")

        f.write("[atoms]\n")
        f.write("; atom_id element type_id mass charge molecule_id residue_id\n")
        for a in system.atoms:
            f.write(f"{a.atom_id} {a.element} {a.type_id} {a.mass:.6f} {a.charge:.6f} "
                     f"{a.molecule_id} {a.residue_id}\n")
        f.write("\n")

        if system.bonds:
            f.write("[bonds]\n")
            f.write("; atom_i atom_j bond_param_id\n")
            for idx, b in enumerate(system.bonds):
                f.write(f"{b.atom_i} {b.atom_j} {idx}\n")
            f.write("\n")

        if system.angles:
            f.write("[angles]\n")
            f.write("; atom_i atom_j atom_k angle_param_id\n")
            for idx, a in enumerate(system.angles):
                f.write(f"{a.atom_i} {a.atom_j} {a.atom_k} {idx}\n")
            f.write("\n")

        if system.torsions:
            f.write("[dihedrals]\n")
            f.write("; atom_i atom_j atom_k atom_l torsion_param_id\n")
            for idx, t in enumerate(system.torsions):
                f.write(f"{t.atom_i} {t.atom_j} {t.atom_k} {t.atom_l} {idx}\n")
            f.write("\n")

        if system.impropers:
            f.write("[impropers]\n")
            f.write("; atom_i atom_j atom_k atom_l improper_param_id\n")
            for idx, t in enumerate(system.impropers):
                f.write(f"{t.atom_i} {t.atom_j} {t.atom_k} {t.atom_l} {idx}\n")
            f.write("\n")

        if system.exclusions:
            f.write("[exclusions]\n")
            f.write("; atom_i atom_j\n")
            for i, j in sorted(system.exclusions):
                f.write(f"{i} {j}\n")
            f.write("\n")

        if system.special_pairs:
            f.write("[special_pairs]\n")
            f.write("; atom_i atom_j pair_param_id lj_scale coulomb_scale\n")
            for idx, sp in enumerate(system.special_pairs):
                f.write(f"{sp.atom_i} {sp.atom_j} {idx} {sp.lj_scale:.10g} {sp.coulomb_scale:.10g}\n")
            f.write("\n")

        if system.constraints:
            f.write("[constraints]\n")
            f.write("; atom_i atom_j target_distance\n")
            for c in system.constraints:
                f.write(f"{c.atom_i} {c.atom_j} {c.distance:.8f}\n")
            f.write("\n")


def _write_parameters(system: CanonicalSystem, path: str) -> None:
    with open(path, "w", newline="\n") as f:
        f.write("[meta]\n")
        f.write(f"schema_version {SCHEMA_VERSION}\n")
        f.write(f"forcefield_family {system.forcefield_family}\n")
        f.write(f"forcefield_version {system.forcefield_variant}\n")
        f.write("energy_unit kcal_per_mol\n")
        f.write("length_unit angstrom\n")
        f.write("charge_unit elementary_charge\n\n")

        f.write("[nonbonded]\n")
        f.write(f"component 0 mie n={system.mie_n} m={system.mie_m}\n")
        f.write("component 1 coulomb method=pme\n\n")

        f.write("[atom_types]\n")
        f.write("; type_id name\n")
        for t in system.atom_types:
            f.write(f"{t.type_id} {t.name}\n")
        f.write("\n")

        f.write("[pair_parameters]\n")
        f.write("; component_id type_i type_j sigma epsilon\n")
        n_types = len(system.atom_types)
        for i in range(n_types):
            for j in range(i, n_types):
                sigma, epsilon = system.get_pair(i, j)
                f.write(f"0 {i} {j} {sigma:.8f} {epsilon:.8f}\n")
        f.write("\n")

        if system.bonds:
            f.write("[bond_parameters]\n")
            f.write("; param_id style k r0\n")
            for idx, b in enumerate(system.bonds):
                f.write(f"{idx} harmonic {b.k:.6f} {b.r0:.6f}\n")
            f.write("\n")

        if system.angles:
            f.write("[angle_parameters]\n")
            f.write("; param_id style k theta0_deg\n")
            for idx, a in enumerate(system.angles):
                f.write(f"{idx} harmonic {a.k:.6f} {a.theta0_deg:.4f}\n")
            f.write("\n")

        if system.torsions:
            f.write("[torsion_parameters]\n")
            f.write("; param_id style k phase_deg multiplicity\n")
            for idx, t in enumerate(system.torsions):
                f.write(f"{idx} periodic {t.k:.8f} {t.phase_deg:.4f} {t.multiplicity}\n")
            f.write("\n")

        if system.impropers:
            f.write("[improper_parameters]\n")
            f.write("; periodic: param_id periodic k phase_deg multiplicity\n")
            f.write("; harmonic: param_id harmonic k phi0_deg\n")
            for idx, t in enumerate(system.impropers):
                if t.style == "harmonic":
                    f.write(f"{idx} harmonic {t.k:.8f} {t.phase_deg:.4f}\n")
                else:
                    f.write(f"{idx} periodic {t.k:.8f} {t.phase_deg:.4f} {t.multiplicity}\n")
            f.write("\n")

        overridden_pairs = [sp for sp in system.special_pairs
                            if sp.sigma is not None and sp.epsilon is not None]
        if overridden_pairs:
            f.write("[special_pair_parameters]\n")
            f.write("; param_id sigma epsilon\n")
            for idx, sp in enumerate(system.special_pairs):
                if sp.sigma is not None and sp.epsilon is not None:
                    f.write(f"{idx} {sp.sigma:.8f} {sp.epsilon:.8f}\n")
            f.write("\n")


def _write_manifest(system: CanonicalSystem, out_dir: str, source_engine: str,
                     source_files: list) -> None:
    coord_path = os.path.join(out_dir, "coordinates.xyz")
    topo_path = os.path.join(out_dir, "topology.dsMDtop")
    param_path = os.path.join(out_dir, "parameters.dsMDprm")

    lines = [
        f"schema_version: {SCHEMA_VERSION}",
        "format: dsMD_input",
        "coordinate_file: coordinates.xyz",
        "topology_file: topology.dsMDtop",
        "parameter_file: parameters.dsMDprm",
        f"forcefield_family: {system.forcefield_family}",
        f"forcefield_variant: {system.forcefield_variant}",
        "polarizable: false",
        "",
        f"# converter: deepscalemd-python-converter {CONVERTER_VERSION}",
        f"# source_engine: {source_engine}",
        f"# generated: {datetime.now(timezone.utc).isoformat()}",
    ]
    for src in source_files:
        lines.append(f"# source_file: {src}")
    lines.append(f"# coordinates_sha256: {_sha256_of_file(coord_path)}")
    lines.append(f"# topology_sha256: {_sha256_of_file(topo_path)}")
    lines.append(f"# parameters_sha256: {_sha256_of_file(param_path)}")
    lines.append("")

    with open(os.path.join(out_dir, "manifest.yaml"), "w", newline="\n") as f:
        f.write("\n".join(lines))


def write_package(system: CanonicalSystem, out_dir: str, source_engine: str,
                   source_files: list) -> None:
    """Write a complete dsMD_input directory. Calls system.finalize()
    first (idempotent) so a package is never written from an unvalidated
    CanonicalSystem."""
    if not system.finalized:
        stage("checking the system")
        system.finalize()

    os.makedirs(out_dir, exist_ok=True)
    stage(f"writing the package to {out_dir}")
    _write_coordinates(system, os.path.join(out_dir, "coordinates.xyz"))
    stage("  topology.dsMDtop")
    _write_topology(system, os.path.join(out_dir, "topology.dsMDtop"))
    stage("  parameters.dsMDprm")
    _write_parameters(system, os.path.join(out_dir, "parameters.dsMDprm"))
    # Manifest last: it records hashes of the three files above.
    stage("  manifest.yaml (hashes of the files above)")
    _write_manifest(system, out_dir, source_engine, source_files)
