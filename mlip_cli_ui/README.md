# mlip_cli_ui — MLIP 모델 고르기·변환 메뉴

`python -m mlip_convert ...` 명령을 직접 치는 대신 번호 메뉴로 모델을 고르고, 받고, 변환한다. 변환은 `mlip_convert`를 그대로 부른다
(이 폴더는 질문하고 명령을 만들어 실행할 뿐이다). 표준 라이브러리만 쓴다: 설치할 것이 없다.

```bash
# 모델 패키지(sevenn, mace, nequip ...)가 설치된 python 환경을 activate 한 뒤, 저장소 맨 위에서
python -m mlip_cli_ui            # 또는  bash mlip_cli_ui/run.sh
python -m mlip_cli_ui --list     # 지원하는 MLIP와 모델을 찾는 곳만 출력
python -m mlip_cli_ui --dry-run  # 메뉴는 다 돌지만 받지도 변환하지도 않고 명령만 보여 준다
```

## 흐름

1. **MLIP 선택**: MACE, NequIP, Allegro, SevenNet, PaiNN, CHGNet, MatterSim, DeePMD, TensorNet(matgl), BAM, UMA, ORB, GAP.
2. **할 일 선택**
   - *LibTorch 모델 폴더로 변환*: `model.pt`(또는 `model.pt2`) + `metadata.json` + `reference.json` → `mlip_backend: libtorch`
   - *Python MLIP용 모델 파일 받기*: 파일을 출력 폴더에 받고 `mlip_config.yaml`을 써 준다 → `mlip_backend: python`
   - UMA, ORB, GAP는 변환기가 없어서 Python MLIP만, PaiNN은 변환만 된다.
3. **모델 파일 선택**: 아래 곳에서 지금 있는 파일을 찾아 목록으로 보여 준다. 없으면 `L`(내 디스크의 경로), `A`(URL, Hugging Face repo, `nequip.net:...`),
   `S`(Hugging Face 검색), `G`(GitHub repo의 release 파일)로 직접 고른다.
4. **옵션**: 변환이면 backend(torchscript/aoti), 정밀도, 가속 커널(e3nn/cueq/openeq/flash), head, PaiNN의 원소 번호, 그 밖의 `mlip_convert` 인자.
5. **출력 폴더**를 정한다. 마지막에 실제로 칠 명령을 보여 주고 확인을 받는다.

## 모델 파일을 찾는 곳 (2026-10-14에 확인한 것)

| MLIP | 찾는 곳 |
|---|---|
| MACE | GitHub release `ACEsuit/mace-mp` (MP-0, MPA, OMAT, MATPES, MH, POLAR, OMOL), HF `InstaDeepAI/mace-organics` |
| NequIP, Allegro | HF `JinukMoon/oh-my-mlip-mirror-nequip` (nequip.net 모델의 미러), HF `InstaDeepAI/nequip-organics`. nequip.net 주소는 `A`로 입력 |
| SevenNet | 설치된 sevenn 패키지 안의 체크포인트, GitHub `MDIL-SNU/SevenNet`의 `pretrained_potentials`와 release |
| CHGNet | 패키지 안의 0.3.0 가중치(파일 필요 없음), HF `LostDecimal/chgnet-0.3.0-mirror` |
| MatterSim | GitHub `microsoft/mattersim`의 `pretrained_models` (1M, 5M) |
| TensorNet | HF `materialyze/TensorNet-PES-*` (모델 폴더 통째로 받는다) |
| DeePMD, PaiNN, GAP | 공개 체크포인트가 없어서 내 파일(`L`) 또는 HF 검색(`S`) |
| BAM | HF `myung-group/BAM-MP-core` (`BAM-MP-core-model.pkl`, 약 410 MB). 변환에는 `bam_torch` 패키지가 필요하다 |
| UMA | HF `facebook/UMA` (gated: 라이선스 동의 + `HF_TOKEN`) |
| ORB | 패키지가 받는다, HF 미러 있음 |

`JinukMoon/`, `LostDecimal/`, `Kameshr/`, `materialyze/`는 공식 배포처가 아니거나 제3자 미러일 수 있다. 목록에 repo 이름이 같이 나오니 라이선스와 출처를 확인하고 쓴다.
목록은 실행할 때마다 Hugging Face와 GitHub에서 새로 읽는다. 인터넷이 안 되면 그 곳만 건너뛰고(`--offline`이면 전부) 내 디스크의 파일만 고를 수 있다.

## 환경 변수와 인자

| | |
|---|---|
| `HF_TOKEN` | gated 또는 비공개 Hugging Face repo (UMA 등) |
| `GITHUB_TOKEN` | GitHub API 횟수 제한에 걸릴 때 |
| `--python /경로/python` | 변환을 돌릴 python (그 모델의 패키지가 설치된 환경). 기본은 이 프로그램을 띄운 python |
| `--download-dir 폴더` | 변환할 때 받는 파일을 두는 곳 (기본 `models/downloads`) |

## 참고

- 변환은 모델마다 다른 환경에서 한다(`docs/mlip_converter_models.md` 6장). 패키지가 없으면 변환 전에 멈추고 설치 명령을 알려 준다.
- NequIP/Allegro의 Python MLIP 모드는 `nequip-compile`로 만든 파일을 읽는다. 그 모드를 고르면 compile 명령을 출력한다.
- 받은 파일은 `.part`로 받은 뒤 이름을 바꾼다: 중간에 끊겨도 반쪽 파일이 남지 않는다.
- 이 폴더는 `mlip_convert`를 바꾸지 않는다.
