#!/bin/bash
# Run the force-field conversion menu:  bash FF_tools/ff_cli_ui/run.sh
cd "$(dirname "$0")/../.."
PYTHONPATH=FF_tools python -m ff_cli_ui "$@"
