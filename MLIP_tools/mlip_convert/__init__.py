"""MLIP model converter for DeepScaleMD (docs/mlip_integration_design.md).

wrapper.py     MDWrapper: the standard model interface the C++ MD code calls (3.1-3.3)
adapters/      one adapter per MLIP family: standard input -> model-specific input, atom energies out
geometry.py    brute-force edges and periodic ghosts for reference structures and tests
"""
