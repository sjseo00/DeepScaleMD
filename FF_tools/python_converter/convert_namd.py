#!/usr/bin/env python3
"""
convert_namd.py - CLI: NAMD config file (+ the files it names) -> DeepScaleMD input.

Usage:
    python convert_namd.py --namd input.namd --out OUT_DIR [--box LX LY LZ] \\
        [--ff-name NAME] [--dihedral-terms {namd,all}] [--e14fac X] \\
        [--allow-lossy] [--min-force-tolerance F] [--input-system PATH]

Everything NAMD needs to run is read from the config: the structure (PSF), coordinates
(PDB) and parameter files it names (relative to the config's directory) become the
dsMD_input package (manifest.yaml, coordinates.xyz, topology.dsMDtop,
parameters.dsMDprm), and the run settings (temperature, timestep, steps, minimization,
cutoffs, PME grid, thermostat, barostat, output) become OUT_DIR/input_dsmd.yaml, which
refers to that manifest. dsMD reads `input_system` relative to the directory it is started
in, so the default (`manifest.yaml`) is meant for starting it inside OUT_DIR:
    cd OUT_DIR && dsMD input_dsmd.yaml
Use --input-system to write a different path (e.g. OUT_DIR/manifest.yaml).

A NAMD setting DeepScaleMD cannot reproduce (rigid bonds, multiple time stepping, a
thermostat it does not have, ...) is an error unless --allow-lossy is given, which turns
it into a warning. --box and --e14fac override the config. To give the files one by one
instead, use convert_charmm.py.
"""

from __future__ import annotations

import argparse
import os
import sys

import dsmd_input
from progress import stage
from canonical_model import ConversionError
from charmm_reader import convert
from namd_config import namd_run_settings, parse_namd_config
from writer import write_package


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--namd", required=True, help="NAMD configuration file")
    parser.add_argument("--out", dest="out_dir", required=True,
                         help="Output directory for dsMD_input and input_dsmd.yaml")
    parser.add_argument("--box", nargs=3, type=float, metavar=("LX", "LY", "LZ"), default=None,
                         help="Orthorhombic box in Angstrom (default: cellBasisVector1/2/3)")
    parser.add_argument("--ff-name", default="namd",
                         help="Recorded as forcefield_variant in manifest.yaml")
    parser.add_argument("--dihedral-terms", choices=("namd", "all"), default="namd",
                         help="namd (default): a dihedral listed N times in the PSF gets its "
                              "first N parameter terms, as NAMD does; all: every parameter "
                              "term of every distinct dihedral")
    parser.add_argument("--e14fac", type=float, default=None,
                         help="1-4 electrostatic scale (default: the config's 1-4scaling)")
    parser.add_argument("--allow-lossy", action="store_true",
                         help="Turn run settings DeepScaleMD cannot reproduce into warnings "
                              "instead of an error")
    dsmd_input.add_yaml_arguments(parser)
    args = parser.parse_args(argv)

    try:
        yaml_args = dsmd_input.yaml_options(args)
        stage(f"reading NAMD config {os.path.basename(args.namd)}")
        config = parse_namd_config(args.namd)
        box = tuple(args.box) if args.box else config.box
        if box is None:
            raise ConversionError(
                f"{args.namd} has no cellBasisVector1/2/3 lines; pass the box with --box LX LY LZ")
        stage("run settings")
        settings = namd_run_settings(config, box)
        warnings = dsmd_input.check(settings, args.allow_lossy)
        e14fac = args.e14fac if args.e14fac is not None else config.scale14
        system = convert(config.structure, config.parameters, config.coordinates,
                          box_angstrom=box, forcefield_variant=args.ff_name,
                          dihedral_terms=args.dihedral_terms, e14fac=e14fac)
        write_package(system, args.out_dir, source_engine="namd",
                      source_files=[args.namd, config.structure] + config.parameters
                                   + [config.coordinates])
        stage("writing input_dsmd.yaml")
        yaml_path = os.path.join(args.out_dir, "input_dsmd.yaml")
        dsmd_input.write_dsmd_input(settings, yaml_path, **yaml_args)
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for message in warnings:
        print(f"warning: dropped NAMD setting: {message}", file=sys.stderr)
    print(f"Wrote {args.out_dir}/manifest.yaml and {yaml_path} "
          f"({len(system.atoms)} atoms, {len(system.bonds)} bonds, "
          f"{len(system.angles)} angles, {len(system.torsions)} dihedrals, "
          f"{len(system.special_pairs)} special pairs; "
          f"box {box[0]:g} x {box[1]:g} x {box[2]:g} A, 1-4scaling {e14fac:g})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
