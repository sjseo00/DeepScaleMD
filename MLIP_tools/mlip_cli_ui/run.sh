#!/bin/bash
# MLIP 모델 메뉴 실행: 모델 패키지가 설치된 python 환경을 activate 한 뒤  bash MLIP_tools/mlip_cli_ui/run.sh
cd "$(dirname "$0")/../.."
PYTHONPATH=MLIP_tools python -m mlip_cli_ui "$@"
