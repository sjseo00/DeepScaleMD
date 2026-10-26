#!/bin/bash
# Run the MLIP model menu: activate the python environment that has the model packages installed, then  bash MLIP_tools/mlip_cli_ui/run.sh
cd "$(dirname "$0")/../.."
PYTHONPATH=MLIP_tools python -m mlip_cli_ui "$@"
