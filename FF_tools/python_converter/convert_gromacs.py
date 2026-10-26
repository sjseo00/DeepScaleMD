#!/usr/bin/env python3
"""
convert_gromacs.py - CLI: GROMACS (.top + .gro [+ .mdp]) -> DeepScaleMD input.

Usage:
    python convert_gromacs.py --top TOP --gro GRO --out OUT_DIR [--mdp RUN.mdp] \\
        [--ff-name NAME] [-I DIR]... [--allow-lossy] [--min-force-tolerance F] \\
        [--input-system PATH]

Example:
    python convert_gromacs.py --top water.top --gro water.gro --mdp md.mdp \\
        --out examples/dsMD_input_tip3p --ff-name tip3p -I /path/to/gromacs/share/gromacs/top

The topology and coordinates become the dsMD_input package (OUT_DIR/manifest.yaml, ...).
With --mdp the run settings (temperature, dt, nsteps, thermostat, barostat, cutoffs, PME
grid, output) become OUT_DIR/input_dsmd.yaml as well. A setting DeepScaleMD cannot
reproduce is an error unless --allow-lossy is given. dsMD reads `input_system` relative to
the directory it is started in, so the default `manifest.yaml` is meant for
    cd OUT_DIR && dsMD input_dsmd.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import dsmd_input
from progress import stage
from canonical_model import ConversionError
from gromacs_mdp import mdp_run_settings, parse_mdp
from gromacs_reader import convert
from writer import write_package


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top", required=True, help="GROMACS .top file")
    parser.add_argument("--gro", required=True, help="GROMACS .gro coordinate file")
    parser.add_argument("--out", dest="out_dir", required=True,
                         help="Output directory for dsMD_input (and input_dsmd.yaml)")
    parser.add_argument("--mdp", default=None, help="GROMACS .mdp run file -> input_dsmd.yaml")
    parser.add_argument("--ff-name", default="unknown",
                         help="Recorded as forcefield_variant in manifest.yaml (e.g. 'amber99sb-ildn', 'tip3p')")
    parser.add_argument("-I", dest="include_dirs", action="append", default=[],
                         help="Additional #include search directory (repeatable, "
                              "e.g. a GROMACS share/gromacs/top installation)")
    parser.add_argument("--allow-lossy", action="store_true",
                         help="Turn .mdp settings DeepScaleMD cannot reproduce into warnings "
                              "instead of an error")
    dsmd_input.add_yaml_arguments(parser)
    args = parser.parse_args(argv)

    warnings = []
    yaml_path = None
    try:
        yaml_args = dsmd_input.yaml_options(args)
        system = convert(args.top, args.gro, args.out_dir,
                          forcefield_variant=args.ff_name,
                          include_dirs=args.include_dirs)
        settings = None
        if args.mdp:
            stage(f"reading run file {os.path.basename(args.mdp)}")
            settings = mdp_run_settings(parse_mdp(args.mdp), system.box)
            warnings = dsmd_input.check(settings, args.allow_lossy)
        write_package(system, args.out_dir, source_engine="gromacs",
                      source_files=[args.top, args.gro] + ([args.mdp] if args.mdp else []))
        if settings is not None:
            stage("writing input_dsmd.yaml")
            yaml_path = os.path.join(args.out_dir, "input_dsmd.yaml")
            dsmd_input.write_dsmd_input(settings, yaml_path, **yaml_args)
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for message in warnings:
        print(f"warning: dropped .mdp setting: {message}", file=sys.stderr)
    print(f"Wrote {args.out_dir}/manifest.yaml" + (f" and {yaml_path}" if yaml_path else "") + " "
          f"({len(system.atoms)} atoms, {len(system.bonds)} bonds, "
          f"{len(system.angles)} angles, {len(system.torsions)} dihedrals, "
          f"{len(system.impropers)} impropers, {len(system.constraints)} constraints, "
          f"{len(system.special_pairs)} special pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
