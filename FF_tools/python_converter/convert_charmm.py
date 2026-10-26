#!/usr/bin/env python3
"""
convert_charmm.py - CLI: CHARMM files (PSF + PRM/PAR + PDB [+ stream .inp]) -> DeepScaleMD input.

Usage:
    python convert_charmm.py --psf SYSTEM.psf --param PARAM.prm [--param PARAM2.prm]... \\
        --pdb COORDS.pdb --out OUT_DIR [--box LX LY LZ] [--inp RUN.inp] [--ff-name NAME] \\
        [--dihedral-terms {namd,all}] [--e14fac X] [--allow-lossy] \\
        [--min-force-tolerance F] [--input-system PATH]

CHARMM/NAMD systems commonly split parameters across multiple files (a base
force field .prm/.par plus per-molecule .str/.prm patches) -- give every one
with its own --param; they are merged in the order given, a later file
replacing what an earlier one defines.

No box size is stored in a PSF or a plain PDB, so it must be supplied with --box
(Angstrom) unless the --inp stream defines it (CRYStal DEFIne). With --inp, the run
settings (DYNAmics, MINImize, NBONds, CRYStal) become OUT_DIR/input_dsmd.yaml too; a setting
DeepScaleMD cannot reproduce is an error unless --allow-lossy is given. To convert from a
NAMD config file instead, use convert_namd.py. dsMD reads `input_system` relative to the
directory it is started in, so the default `manifest.yaml` is meant for
    cd OUT_DIR && dsMD input_dsmd.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import dsmd_input
from progress import stage
from canonical_model import ConversionError
from charmm_input import charmm_run_settings
from charmm_reader import convert
from writer import write_package


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--psf", required=True, help="CHARMM/NAMD .psf topology file")
    parser.add_argument("--param", dest="params", action="append", required=True, metavar="FILE",
                         help="CHARMM/XPLOR parameter file (.prm/.par/.xplor/.str); repeat for "
                              "every file, merged in the order given")
    parser.add_argument("--pdb", required=True, help="PDB coordinate file (ATOM/HETATM records)")
    parser.add_argument("--out", dest="out_dir", required=True,
                         help="Output directory for dsMD_input (and input_dsmd.yaml)")
    parser.add_argument("--box", nargs=3, type=float, metavar=("LX", "LY", "LZ"), default=None,
                         help="Orthorhombic box dimensions in Angstrom (not stored in PSF/PDB; "
                              "optional when --inp defines it)")
    parser.add_argument("--inp", default=None, help="CHARMM stream (.inp) file -> input_dsmd.yaml")
    parser.add_argument("--ff-name", default="unknown",
                         help="Recorded as forcefield_variant in manifest.yaml (e.g. 'charmm36')")
    parser.add_argument("--allow-lossy", action="store_true",
                         help="Turn --inp settings DeepScaleMD cannot reproduce into warnings "
                              "instead of an error (the force field itself is always converted "
                              "exactly)")
    parser.add_argument("--dihedral-terms", choices=("namd", "all"), default="namd",
                         help="namd (default): a dihedral listed N times in the PSF gets its "
                              "first N parameter terms, as NAMD does; all: every parameter "
                              "term of every distinct dihedral")
    parser.add_argument("--e14fac", type=float, default=None,
                         help="1-4 electrostatic scale (NAMD 1-4scaling); default: the "
                              "parameter files' E14FAC, else 1.0")
    dsmd_input.add_yaml_arguments(parser)
    args = parser.parse_args(argv)

    warnings = []
    yaml_path = None
    try:
        yaml_args = dsmd_input.yaml_options(args)
        box = tuple(args.box) if args.box else None
        settings = None
        if args.inp:
            stage(f"reading stream file {os.path.basename(args.inp)}")
            settings = charmm_run_settings(args.inp, box)
            box = box or settings.box
            warnings = dsmd_input.check(settings, args.allow_lossy)
        if box is None:
            raise ConversionError("no periodic box: pass --box LX LY LZ (or a CRYStal DEFIne in --inp)")
        system = convert(args.psf, args.params, args.pdb,
                          box_angstrom=box,
                          forcefield_variant=args.ff_name,
                          allow_lossy=args.allow_lossy,
                          dihedral_terms=args.dihedral_terms,
                          e14fac=args.e14fac)
        write_package(system, args.out_dir, source_engine="charmm",
                      source_files=[args.psf] + args.params + [args.pdb]
                                   + ([args.inp] if args.inp else []))
        if settings is not None:
            stage("writing input_dsmd.yaml")
            yaml_path = os.path.join(args.out_dir, "input_dsmd.yaml")
            dsmd_input.write_dsmd_input(settings, yaml_path, **yaml_args)
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for message in warnings:
        print(f"warning: dropped CHARMM setting: {message}", file=sys.stderr)
    print(f"Wrote {args.out_dir}/manifest.yaml" + (f" and {yaml_path}" if yaml_path else "") + " "
          f"({len(system.atoms)} atoms, {len(system.bonds)} bonds, "
          f"{len(system.angles)} angles, {len(system.torsions)} dihedrals, "
          f"{len(system.special_pairs)} special pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
