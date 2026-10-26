#!/usr/bin/env python3
"""
convert_lammps.py - CLI: LAMMPS (input script + data file) -> DeepScaleMD input.

Usage:
    python convert_lammps.py --input input.in --out out/my_system \\
        [--data path/to/data.file] [--ff-name NAME] [--allow-lossy] \\
        [--min-force-tolerance F] [--input-system PATH]

The data file is normally named in the input script's own `read_data`
command; pass --data only to override or supply it explicitly (e.g. if the
script's read_data path is relative to a directory this CLI isn't run
from).

The script and data file become the dsMD_input package (OUT_DIR/manifest.yaml, ...), and
the script's run settings (timestep, run, minimize, velocity, fixes, thermo, dump, cutoff,
kspace) become OUT_DIR/input_dsmd.yaml. A setting DeepScaleMD cannot reproduce is an error
unless --allow-lossy is given (which also lets improper_style harmonic impropers be
dropped). dsMD reads `input_system` relative to the directory it is started in, so the
default `manifest.yaml` is meant for
    cd OUT_DIR && dsMD input_dsmd.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import dsmd_input
from progress import stage
from canonical_model import ConversionError
from lammps_reader import convert
from lammps_run import lammps_run_settings
from writer import write_package


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", dest="input_script", required=True,
                         help="LAMMPS input script (units/style/read_data commands)")
    parser.add_argument("--out", dest="out_dir", required=True,
                         help="Output directory for dsMD_input and input_dsmd.yaml")
    parser.add_argument("--data", dest="data_file", default=None,
                         help="LAMMPS data file (default: the input script's own read_data path)")
    parser.add_argument("--ff-name", default="unknown",
                         help="Recorded as forcefield_variant in manifest.yaml")
    parser.add_argument("--allow-lossy", action="store_true",
                         help="Drop improper_style harmonic impropers and turn run settings "
                              "DeepScaleMD cannot reproduce into warnings instead of an error")
    dsmd_input.add_yaml_arguments(parser)
    args = parser.parse_args(argv)

    try:
        yaml_args = dsmd_input.yaml_options(args)
        system = convert(args.input_script, args.data_file,
                          forcefield_variant=args.ff_name, allow_lossy=args.allow_lossy)
        stage("run settings")
        settings = lammps_run_settings(args.input_script, system.box, "lammps")
        warnings = dsmd_input.check(settings, args.allow_lossy)
        sources = [args.input_script] + ([args.data_file] if args.data_file else [])
        write_package(system, args.out_dir, source_engine="lammps", source_files=sources)
        stage("writing input_dsmd.yaml")
        yaml_path = os.path.join(args.out_dir, "input_dsmd.yaml")
        dsmd_input.write_dsmd_input(settings, yaml_path, **yaml_args)
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for message in warnings:
        print(f"warning: dropped LAMMPS setting: {message}", file=sys.stderr)
    print(f"Wrote {args.out_dir}/manifest.yaml and {yaml_path} "
          f"({len(system.atoms)} atoms, {len(system.bonds)} bonds, "
          f"{len(system.angles)} angles, {len(system.torsions)} dihedrals, "
          f"{len(system.impropers)} impropers, {len(system.special_pairs)} special pairs, "
          f"mie({system.mie_n},{system.mie_m}))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
