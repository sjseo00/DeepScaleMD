#!/bin/bash
# 힘장 변환 메뉴 실행:  bash FF_tools/ff_cli_ui/run.sh
cd "$(dirname "$0")/../.."
PYTHONPATH=FF_tools python -m ff_cli_ui "$@"
