#!/bin/bash
# MLIP 모델 메뉴 실행: 모델 패키지가 설치된 python 환경을 activate 한 뒤  bash mlip_cli_ui/run.sh
cd "$(dirname "$0")/.."
python -m mlip_cli_ui "$@"
