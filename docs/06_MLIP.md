# 06. Machine-Learned Potentials (MLIP)

## Contents

1. [Overview and choosing a path](#1-overview-and-choosing-a-path)
2. [Python MLIP](#2-python-mlip)
3. [LibTorch MLIP](#3-libtorch-mlip)
4. [Supported models](#4-supported-models)
5. [Multi-GPU MLIP](#5-multi-gpu-mlip)
6. [Tutorial: water box](#6-tutorial-water-box)

This page explains how to run DeepScaleMD with a machine-learned interatomic potential (MLIP) such as MACE, NequIP, SevenNet or MatterSim,
from the model file to a running simulation. It assumes dsMD is already built ([01_Installation](01_Installation.md)) and that you
can run a plain example ([02_Quick_Start](02_Quick_Start.md)).

---

## 1. Overview and choosing a path

### What an MLIP run needs

| You provide | Where it is described |
|---|---|
| Atom positions and the box | `coordinate_file`, `box_x/y/z`, `periodic_*` ([03_Input_Reference](03_Input_Reference.md), [04_Preparing_a_System](04_Preparing_a_System.md)) |
| A topology that lists the atom types and their masses | the same `.top` file as for any run. The charges and Lennard-Jones numbers in it are not used by an MLIP |
| A model | this page |
| The normal MD settings | integrator, thermostat, timestep, output: unchanged ([03_Input_Reference](03_Input_Reference.md)) |

Units are the same as everywhere in dsMD: Å, kcal/mol, fs. A model that works in eV is converted for you.

### Two ways to run a model

dsMD has two backends for an MLIP. You choose one with `mlip_backend:` in the input file.

| | **Python MLIP** (`mlip_backend: python`) | **LibTorch MLIP** (`mlip_backend: libtorch`) |
|---|---|---|
| How the model is used | dsMD starts an embedded Python interpreter and calls the model package (an ASE calculator) each step | you convert the model once into a model folder; dsMD loads that folder in C++ and runs it on the GPU |
| Builds | CPU and CUDA builds | **CUDA builds only**, configured with `-DENABLE_LIBTORCH=ON` |
| Model files | the original files of the package (a `.model`, a checkpoint, or a name the package knows) | a folder with `model.pt`, `metadata.json`, `reference.json` made by `mlip_convert` |
| Python needed while running | yes (the model's packages must be importable) | not for e3nn kernels. cuEquivariance and OpenEquivariance models import their package once at start-up through the embedded Python |
| Speed | slow: positions and forces cross the host every step | the production path: everything stays on the GPU |
| Several GPUs / MPI ranks | correct, but no speed-up (every rank computes the whole system) | the box is divided between ranks; see [section 5](#5-multi-gpu-mlip) |
| Models | 12 models have an adapter, see [section 2](#2-python-mlip) | 10 models have a converter, see [section 3](#3-libtorch-mlip) |

Pick like this:

- *You want to try a model, compare it with its own package, or run on CPU* → **Python MLIP**.
- *You want speed, or several GPUs* → **LibTorch MLIP** (needs a CUDA build and a model with a converter).
- *The model has no converter* (UMA, ORB, GAP, MatGL M3GNet) → **Python MLIP** is the only way.

The tensor-product kernels (e3nn, cuEquivariance, OpenEquivariance, FlashTP) are chosen separately in both backends:
[section 3.6](#36-kernels-e3nn-cueq-openeq-flash).

### The input file keys for an MLIP

These keys are specific to MLIP runs. All other keys are explained in [03_Input_Reference](03_Input_Reference.md).

```yaml
force_calculation:
  mode: python            # "MLIP on, built-in force field off". See the note below.

mlip_backend: libtorch    # python (default) | libtorch
```

> **`mode: python` does not mean the Python backend.** It is the name of the switch that turns the MLIP on and the built-in force
> field off, for both backends. The backend is `mlip_backend`. `mode: libtorch` is rejected with an error.

| Key | Used by | Meaning |
|---|---|---|
| `force_calculation.mode` | both | `python` = run the MLIP (no built-in force field). `builtin` = the classical force field |
| `mlip_backend` | both | `python` (default when absent) or `libtorch` |
| `mlip_model` | libtorch | the model folder written by `mlip_convert` (required) |
| `python_mlip_path` | python | folder that holds `ml_forces.py`; default `mlip_python` |
| `mlip_config:` | python | section with `model_type`, `model_path`, `model_size`, `device`, `dtype`, `accel`, and model-specific keys ([section 2](#22-the-mlip_config-section)) |
| `mlip_elements:` | both | which element each atom type is, only when dsMD cannot tell ([below](#which-element-is-each-atom-type)) |
| `advanced.tf32` | libtorch | run float32 matrix products as TF32 on the GPU (default `false`); ignored with a message by the Python backend |
| `advanced.timing_trace` | both | per-step timing lines: `off` (default), `step`, `every` |

### Which element is each atom type?

An MLIP needs the atomic number of every atom, but the MD input only has atom types and force-field names (`OW`, `HW`, `CA`).
dsMD finds the element of each type itself, from these sources, and they must agree:

1. the element column of the topology, if it has one;
2. the element symbol in the first column of the coordinate file;
3. the mass of the type (15.999 is O, 1.008 is H).

The result is printed at start-up as `MLIP elements : O = O (coordinate file, mass), H = H (coordinate file, mass)`.
When dsMD cannot decide, or two sources disagree, it stops and prints the block to add to your input file. You can also write it
yourself (atom **type name**, then element):

```yaml
mlip_elements:
  OW: O
  HW: H
```

The old key `mlip_type_to_z` is gone and stops the run with this message.

### Limitations

- The model must know every element in your system. The converter writes the supported elements into `metadata.json`; the Python
  backend leaves this check to the model package.
- Only the settings listed above are MLIP-specific. A comma list in `force_calculation.mode` (`builtin, python`) is accepted by the
  parser for classical + MLIP forces together; it is not covered here.

---

## 2. Python MLIP

dsMD starts a Python interpreter inside the program, imports `MLIP_tools/mlip_python/ml_forces.py`, and every step hands it the positions, atom
types, box and periodic flags. The adapter for your model (a small class in `MLIP_tools/mlip_python/adapters/`, one file per model) wraps the
model's own ASE calculator, and the forces come back to dsMD. Any failure in Python (a missing package, a bad model file, an error in
a force call) stops the run with the Python traceback. It never turns into zero forces.

### 2.1 Requirements

- dsMD built with `-DENABLE_PYTHON=ON` (the default of CMake). Of the build scripts in the repository root, only `build_cuda.sh` turns it on;
  `build.sh` and `build_cuda_gh200.sh` pass `-DENABLE_PYTHON=OFF`. A build without it refuses `mlip_backend: python` at start-up.
- A Python environment with NumPy and the packages of your model, **activated before you run dsMD**. dsMD adds the activated
  environment's packages (`VIRTUAL_ENV`, `CONDA_PREFIX`) to the embedded interpreter. That environment must use the **same Python
  version** dsMD was built with.
- For a GPU run, a CUDA build and `device: cuda` in `mlip_config`.

### 2.2 The `mlip_config` section

```yaml
force_calculation:
  mode: python

mlip_backend:     python
python_mlip_path: MLIP_tools/mlip_python      # default; the folder with ml_forces.py

mlip_config:
  model_type: mace                 # which adapter (table below)
  model_path: models/mace-mp-0-small.model
  device:     cuda                 # cuda | cpu
  dtype:      float32              # float32 | float64
  accel:      e3nn                 # e3nn | cueq | openeq | flash
```

| Key | Default | Meaning |
|---|---|---|
| `model_type` | none, required | the adapter name, from the table below |
| `model_path` | empty | model file or name. What it means depends on the model |
| `model_size` | `small` | for foundation models the package downloads by name: MACE (`small`, `medium`, `large`), MatterSim (`1M` or `5M`: **write it, the default `small` is not a MatterSim size**) |
| `device` | `cpu` | `cuda` or `cpu`. **The default is `cpu`, also in a CUDA build**, so write `device: cuda` for a GPU run |
| `dtype` | `float32` | `float32` or `float64` |
| `accel` | `e3nn` | tensor-product kernels: `e3nn`, `cueq`, `openeq`, and `flash` (MACE and SevenNet only; `flash`: SevenNet only). A NequIP or Allegro model carries its kernels from `nequip-compile` |
| anything else | | passed to the adapter unchanged (for example `modal`, `head`, `task_name`) |

Adapters, by `model_type`:

| `model_type` (aliases) | Package to install | `model_path` | Extra keys |
|---|---|---|---|
| `mace` (`mace_mp`, `mace_off`) | `mace-torch` | a saved MACE model file. Empty: the package downloads MACE-MP-0 (or MACE-OFF for `mace_off`) of `model_size` | |
| `nequip`, `allegro` | `nequip` (`nequip-allegro` for Allegro) | a file made by `nequip-compile` (required) | `chemical_symbols`: optional; by default taken from the elements dsMD found |
| `sevennet` (`7net`) | `sevenn` | a checkpoint `.pth`, or a name such as `7net-0`. Empty: `7net-0` | `modal`: for multi-modal models |
| `chgnet` | `chgnet` | empty: the weights inside the package | |
| `mattersim` | `mattersim` | a checkpoint file. Empty: the package's model of `model_size` (`1M` or `5M`) | |
| `matgl` (`m3gnet`, `tensornet`) | `matgl` | not used | `model_name` (write it): a matgl model name or a model folder |
| `deepmd` (`deepmd_kit`) | `deepmd-kit` | a frozen model (required) | `head` |
| `bam` (`bam_torch`) | `bam_torch` (from the myung-group/BAM-torch repository) | a BAM checkpoint `.pkl` (required) | |
| `uma` (`uma_s`, `uma_m`, `fairchem`) | `fairchem-core` | a UMA checkpoint (required; the weights are gated) | `task_name`: `omat` (default), `omol`, `oc20`, `odac`, `omc` |
| `orb` | `orb-models` | not used (the package loads ORB v2) | |
| `gap` (`quip`) | `quippy-ase` | the GAP `.xml` file (required) | `args_str` |

`model_type: auto` guesses the adapter from words in `model_path`; naming the type explicitly is safer.

### 2.3 Getting a model file

You can get the file by hand, from the model's own website or package. Two helpers exist in the repository:

- `PYTHONPATH=MLIP_tools python -m mlip_cli_ui` is a numbered menu: choose the MLIP, choose "get the model file for the Python MLIP", pick a file from
  Hugging Face, GitHub, the installed package or your disk. It saves the file and writes a `mlip_config.yaml` snippet next to it.
  **That snippet is not read by dsMD**: copy its lines (`mlip_backend:` and the `mlip_config:` block) into your input file.
  The menu is described in [section 3.3](#33-the-menu-mlip_cli_ui).
- NequIP and Allegro need one extra step: compile the file for the Python MLIP:
  `nequip-compile <model> <out>.nequip.pt2 --mode aotinductor --device cuda --target ase` (with torch older than 2.10:
  `--mode torchscript` and the name `<out>.nequip.pth`).

A MACE run that downloads its foundation model (`model_path` empty) needs internet access on the machine that runs dsMD.

### 2.4 Example

`examples/mlip_water100/` has the whole thing ([tutorial in section 6](#6-tutorial-water-box)). The MLIP part of
`input_python_e3nn.yaml`:

```yaml
force_calculation:
  mode: python

mlip_backend:     python
python_mlip_path: MLIP_tools/mlip_python

mlip_config:
  model_type: sevennet
  model_path: 7net-0              # SevenNet-0 ships inside the sevenn package
  device: cuda
  dtype:  float32
  accel:  e3nn
```

### 2.5 Several MPI ranks

The Python backend works the same way with any number of ranks: every rank collects the atoms of all ranks, runs the model on the
whole system, and keeps the forces of its own atoms. The result is correct, but more ranks do not make it faster. For several GPUs
use the LibTorch MLIP.

### Limitations

- **Speed.** Every step copies positions to the host and forces back. Use it for correctness checks and model comparisons, not
  for long production runs.
- `advanced.tf32` has no effect (dsMD prints a message and ignores it).
- Only MACE and SevenNet were run through dsMD in this project with real model files (see [section 4](#4-supported-models)). The
  other adapters are written but were not exercised on real checkpoints here; treat them as untested until you check step 0
  against the package itself.
- UMA, ORB and GAP can be used only through the Python MLIP (no converter).

---

## 3. LibTorch MLIP

The converter `mlip_convert` turns a model into a **model folder**. dsMD loads the folder in C++ with LibTorch and runs the model on
the GPU together with the MD, so no Python is in the force loop. This is the fast path and the one for several GPUs.

### 3.1 Requirements and build

- A **CUDA build** of dsMD configured with `-DENABLE_LIBTORCH=ON` and `-DLIBTORCH_PYTHON=<python that has torch>`. CMake takes
  LibTorch from that Python's `torch` package, which must be built with the cxx11 ABI (a CUDA build, not a CPU-only torch).
  The compiler must be GCC 9 or newer, and the CUDA toolkit should be the same family as torch's CUDA (12.x).
  If you want cuEquivariance or OpenEquivariance models, also pass `-DENABLE_PYTHON=ON` (dsMD imports their packages at start-up).

  ```bash
  cmake -S . -B build_mlip -DCMAKE_BUILD_TYPE=Release -DBACKEND=CUDA \
        -DDSDECOMPX_DIR=<dsDecompX install> \
        -DENABLE_PYTHON=ON -DENABLE_LIBTORCH=ON -DLIBTORCH_PYTHON=$(which python)
  cmake --build build_mlip -j
  ```

  `build_cuda.sh` in the repository root is a complete example for the project's cluster (A100, CUDA 12.8, GCC 9.5).
  The executable is `build_mlip/bin/dsMD`. dsDecompX must be built and installed first ([01_Installation](01_Installation.md)).
- **The torch versions must match.** The torch you convert with and the LibTorch dsMD was built with must have the same
  `major.minor` (for example 2.7.x). The converter records its torch and CUDA versions in `metadata.json`, and dsMD compares them at
  start-up: a different `major.minor` of torch stops the run with both versions printed.
- Because each model package pins its own torch, convert each model in the Python environment of that model.

### 3.2 Converting a model

```bash
PYTHONPATH=MLIP_tools python -m mlip_convert <model> --model <model file> -o <output folder> --dtype float32
```

`<model>` is one of: `mace`, `nequip`, `allegro`, `sevennet`, `painn`, `chgnet`, `mattersim`, `deepmd`, `tensornet`, `bam`.
The output folder gets three files:

| File | Content |
|---|---|
| `model.pt` | the model as TorchScript (`model.pt2` with `--backend aoti`) |
| `metadata.json` | what dsMD must know: model family and name, cutoff, number of layers, elements, precision, kernels, torch/CUDA versions, edge-shift and layer-exchange flags |
| `reference.json` | a small test structure with the model's own energy and forces. The converter checks the saved model against it |

Commands that were used for real models (each in the Python environment of that model):

```bash
# SevenNet: the checkpoint is inside the sevenn package, nothing to download
PYTHONPATH=MLIP_tools python -m mlip_convert sevennet --model $(python -c "import os, sevenn; print(os.path.dirname(sevenn.__file__))")/pretrained_potentials/SevenNet_0__11Jul2024/checkpoint_sevennet_0.pth -o models/libtorch_sevennet --dtype float32

# MACE: a saved model file (see examples/mlip_water100/libtorch_e3nn.sh for how to save MACE-MP-0)
PYTHONPATH=MLIP_tools python -m mlip_convert mace --model models/mace-mp-0-small.model -o models/libtorch_mace --dtype float32

# NequIP / Allegro: a .nequip.zip or .ckpt file, or an address on nequip.net
PYTHONPATH=MLIP_tools python -m mlip_convert nequip --model nequip.net:mir-group/NequIP-OAM-L:0.1 -o models/libtorch_nequip

# CHGNet: the weights inside the package
PYTHONPATH=MLIP_tools python -m mlip_convert chgnet -o models/libtorch_chgnet --dtype float32

# MatterSim: a checkpoint .pth;  TensorNet: a matgl model folder;  BAM: a checkpoint .pkl
PYTHONPATH=MLIP_tools python -m mlip_convert mattersim --model mattersim-v1.0.0-1M.pth -o models/libtorch_mattersim --dtype float32
PYTHONPATH=MLIP_tools python -m mlip_convert tensornet --model <matgl model folder> -o models/libtorch_tensornet --dtype float32
PYTHONPATH=MLIP_tools python -m mlip_convert bam --model BAM-MP-core-model.pkl -o models/libtorch_bam --dtype float32

# PaiNN (your own schnetpack best_model) and DeePMD (your own training checkpoint .pt)
PYTHONPATH=MLIP_tools python -m mlip_convert painn --model best_model --elements 1,8 -o models/libtorch_painn --dtype float32
PYTHONPATH=MLIP_tools python -m mlip_convert deepmd --model model.ckpt-10000.pt -o models/libtorch_deepmd
```

Options:

| Option | Meaning |
|---|---|
| `-o`, `--output` | output folder (required) |
| `--model` | the model file (or folder/address, depending on the model) |
| `--dtype float32\|float64` | model precision. **The default is `float64`; use `float32` for MD**, as the examples do. NequIP and DeePMD models carry their own precision and refuse a different one |
| `--accel e3nn\|cueq\|openeq\|flash` | tensor-product kernels ([section 3.6](#36-kernels-e3nn-cueq-openeq-flash)). With `cueq`, `openeq`, `flash` also pass `--device cuda`; convert on the GPU machine |
| `--head <name>` | the head (MACE) or modal (SevenNet) of a model that has several. `--list-heads --model <file>` prints them. A multi-modal SevenNet model (Omni, MF-0) cannot be converted without one |
| `--layer-exchange` | for several GPUs ([section 5](#5-multi-gpu-mlip)) |
| `--no-edge-shifts` | use the older, slower way of handling periodic images (extra atoms instead of shifts); only if you need models that run on an older dsMD. Cannot be combined with `--layer-exchange` |
| `--elements 1,8` | the atomic numbers the model knows. Needed for PaiNN, and for a NequIP model whose type names are not element symbols |
| `--cutoff` | cutoff in Å. Real models use their own cutoff (for example MACE's `r_max`); this is for the toy test models |
| `--backend torchscript\|aoti` | default `torchscript`; see [section 3.7](#37-aoti-backend) |
| `--reference-atoms N` | owned atoms of `reference.json` (default 12). Lower it (for example 4) for very deep models if conversion runs out of memory |

A converted folder is tied to the machine it was made on when it uses `cueq`, `openeq` or `flash`: `metadata.json` then holds absolute
paths of the kernel libraries. Convert such models on the machine and in the environment where you run them.

### 3.3 The menu: `mlip_cli_ui`

If you do not want to type the command, `mlip_cli_ui` asks the questions and runs `mlip_convert` for you. It uses only the Python
standard library; activate the environment that has the model's packages and run it from the repository root:

```bash
PYTHONPATH=MLIP_tools python -m mlip_cli_ui              # or: bash MLIP_tools/mlip_cli_ui/run.sh
PYTHONPATH=MLIP_tools python -m mlip_cli_ui --list       # only list the MLIPs and where model files are looked for
PYTHONPATH=MLIP_tools python -m mlip_cli_ui --dry-run    # go through the menus, print the commands, download and convert nothing
```

The steps: choose the MLIP → choose what to do (**convert to a LibTorch model folder**, or **get the model file for the Python MLIP**) →
choose the model file (found in Hugging Face repos, GitHub releases, the installed package, or typed by you) → options (precision,
kernel, head, layer exchange) → output folder. Before it runs anything it shows the exact command and asks for confirmation.
It prints the lines to put into your input file at the end. Other arguments: `--python /path/to/python` (the Python that does the
conversion), `--download-dir <folder>` (default `models/downloads`), `--offline`. Tokens: `HF_TOKEN` for gated Hugging Face
repositories (UMA), `GITHUB_TOKEN` if GitHub rate-limits you. Some Hugging Face repositories listed are third-party mirrors; the menu shows the
repository name, so check the source and licence before use.

### 3.4 The input file

```yaml
force_calculation:
  mode: python                     # turns the MLIP on (both backends)

mlip_backend: libtorch
mlip_model:   models/libtorch_sevennet       # the folder made by mlip_convert

# optional
# mlip_elements:
#   OW: O
#   HW: H

# advanced:
#   tf32: false                    # float32 matrix products in TF32 (Ampere GPUs and newer)
```

`mlip_config:` and `python_mlip_path:` are not used by this backend. The rest of the file is an ordinary MD input; the complete
water example is `examples/mlip_water100/input_libtorch.yaml`.

Start-up lines to look for in the log:

```text
MLIP backend          : libtorch
MLIP model            : models/libtorch_sevennet
MLIP elements         : O = O (coordinate file, mass), H = H (coordinate file, mass)
[MLIP] ghost depth ... A, halo width ... A, edge cutoff ... A
```

What dsMD checks when it loads the folder: `metadata.json` is complete and consistent; the torch major.minor matches (error) and the
CUDA version is close (warning for TorchScript, error for AOTI); the model knows the elements of your system; every kernel library
named in `metadata.json` exists. If one of these fails the run stops before the first step with the reason.

### 3.5 What the model sees: ghost atoms

A model with several message-passing layers needs the atoms around each owned atom as deep as *layers × cutoff* (the
`ghost_cutoff` in `metadata.json`). With a normal conversion every MPI rank holds that many extra "ghost" atoms; for deep models
this makes small boxes expensive, which is why [section 5](#5-multi-gpu-mlip) exists. On one rank the periodic images are handled as
shifts of the edges, so a small box costs little.

### 3.6 Kernels: e3nn, cueq, openeq, flash

Equivariant models can use faster tensor-product kernels. You choose them when you convert (`--accel`), and in the Python MLIP with `accel:`.

| Kernel | `--accel` | Models | What it needs |
|---|---|---|---|
| e3nn | `e3nn` (default) | all | nothing extra |
| cuEquivariance | `cueq` | MACE, SevenNet, BAM | Python packages `cuequivariance`, `cuequivariance-torch` and the matching `cuequivariance-ops-torch-cu12` (or cu11/cu13); a GPU; dsMD built with `-DENABLE_PYTHON=ON`, because dsMD imports `cuequivariance_ops_torch` once at start-up so its operators are registered |
| OpenEquivariance | `openeq` | MACE, SevenNet, BAM (BAM: `interaction_block: fast` checkpoints only) | Python package `openequivariance`; on first import it **compiles a CUDA extension**, so a C++ compiler, `ninja` and `nvcc` must be on `PATH` (also inside the job); dsMD built with `-DENABLE_PYTHON=ON` |
| FlashTP | `flash` | SevenNet only | the `flashTP_e3nn` package built for your GPU |

Things to know:

- NequIP and Allegro get their kernels when you compile the model with the nequip tools, not through `--accel`.
- A BAM checkpoint can be converted for `cueq` only if it was trained with cuEquivariance. The converter stops with the reason.
- If OpenEquivariance cannot build its extension, the forces would be wrong without any error in a naive setup (energies right,
  forces wrong, temperature climbing). dsMD therefore checks after the import that the extension is built and that the backward
  pass is registered, and stops with the build error text if not. Typical cause: no compiler on `PATH` in the batch job.
- The torch of the Python environment that dsMD imports from (`VIRTUAL_ENV`) must be the same build as the LibTorch dsMD was linked with
  for `cueq`/`openeq`; otherwise dsMD stops and prints both versions.

### 3.7 AOTI backend

`--backend aoti` compiles the model ahead of time into `model.pt2` for one GPU (the architecture is stored as `target_gpu` in
`metadata.json`). It supports only the e3nn kernels, has no edge shifts and no layer exchange, and a CUDA major-version mismatch is an error.
All example runs in this repository use the default TorchScript backend; AOTI is available but was not exercised on a GPU here.

### Limitations

- **CUDA builds only.** CPU builds refuse `mlip_backend: libtorch`. Use the Python MLIP there.
- Models without a converter cannot be used: UMA, ORB, GAP, MatGL M3GNet, eSEN/eSCN/EquiformerV2, FeNNix-Bio1, GRACE
  (reasons are written in `docs_old/mlip_converter_models.md`, chapter 4).
- **Memory.** Deep models on a small GPU run out of memory. In the logs kept in `examples/mlip_water_multigpu/`, BAM-MP-core (5
  layers) ran out of memory on 40 GB A100s for 3000 atoms in several runs, and NequIP-OAM-L did for 24000 atoms.
- **The first two MD steps are slow** (seconds): the model optimizes itself for the real sizes on its first calls. Judge speed
  over hundreds of steps, or use `advanced: timing_trace: every` and ignore the first two lines.
- `advanced.tf32` trades accuracy for speed. Measured by the project on an A100 (MACE, 3000 atoms): step 0 energy changed by 9e-7 relative,
  largest force by 0.35 %, step time 1.4× faster. Check energy drift on a short NVE run before using it for production.
- A converted folder follows the torch version it was made with; a model converted with torch 2.4 is refused by a dsMD built with
  LibTorch 2.7.

---

## 4. Supported models

"Converter" means `PYTHONPATH=MLIP_tools python -m mlip_convert <model>` (LibTorch MLIP). "Python MLIP" means an adapter exists for `mlip_backend: python`.
The last column says what was actually run, so you know what to check yourself.

**Legend for verification**

- **Real weights, A100**: ran in dsMD on the project's cluster with a real model; the logs are kept in `examples/`. Only what the logs show is claimed: the step 0 energy and largest force compared with other runs of the same system.
- **Real weights, dev PC**: ran in dsMD or against the model's own package on the small development machine.
- **Random model**: the adapter was checked against the package's own calculator with a randomly initialized model, not a real checkpoint.
- **Not run**: written, but not exercised.

| Model | Converter | Kernels (converter) | Python MLIP (`model_type`) | Layer exchange (several GPUs) | What was verified |
|---|---|---|---|---|---|
| MACE (MP-0, OMAT, ...) | `mace` | e3nn, cueq, openeq | `mace`, `mace_mp`, `mace_off` | yes (not for models with joint embedding) | **Real weights, A100**: MACE-OMAT-0 small, water 3000 atoms, libtorch e3nn and openeq, Python MLIP with e3nn, cueq and openeq: the step 0 energy and largest force agree (energy within 1e-5 relative, largest force 23.6657); layered model on 1, 2, 4 GPUs. **Real weights, dev PC**: MACE-MP-0 small, libtorch against Python. libtorch + cueq with real kernels: **not run** |
| NequIP | `nequip` | e3nn (others by `nequip-compile`) | `nequip` (compiled file) | yes | **Real weights, A100**: NequIP-OAM-L, layered, 1, 2 and 4 GPUs at 3000 atoms (out of memory at 24000 atoms) |
| Allegro | `allegro` | e3nn | `allegro` (compiled file) | no (strictly local, nothing to gain) | Same path as NequIP; no run with real Allegro weights in this repository |
| SevenNet | `sevennet` | e3nn, cueq, openeq, flash | `sevennet`, `7net` | yes (also multi-modal) | **Real weights, A100**: SevenNet Omni (head `omol25_low`), openeq, layered, 1, 2, 4 GPUs. **Real weights, dev PC**: SevenNet-0 with the Python MLIP. `flash`: **not run** (FlashTP could not be installed) |
| MatterSim | `mattersim` | e3nn | `mattersim` | yes | **Real weights, A100**: MatterSim 1M, layered, 1, 2, 4 GPUs. Python adapter: **not run** |
| BAM (RACE) | `bam` | e3nn, cueq (cuEquivariance-trained only), openeq (`fast` blocks only) | `bam` | yes | **Real weights, A100**: BAM-MP-core, e3nn, layered; several runs ran out of memory on 40 GB. cueq, openeq: **not run** |
| TensorNet (matgl) | `tensornet` | e3nn | `matgl`, `tensornet` | yes | **Random model** only. Python adapter: **not run** |
| PaiNN (schnetpack) | `painn` | e3nn | none | yes | **Random model** only (no public pretrained PaiNN potential) |
| DeePMD-kit (PyTorch) | `deepmd` | e3nn | `deepmd` | local model: ranks compute only their own atoms | **Random model** only; strictly local descriptors only |
| CHGNet | `chgnet` | e3nn | `chgnet` | **no** | Converter checked with the weights shipped in the CHGNet package; Python adapter: **not run**. Needs a ghost depth of 24 Å |
| UMA (fairchem) | none | | `uma` | no | Python adapter: **not run** (gated weights) |
| ORB | none | | `orb` | no | Python adapter: **not run** |
| GAP | none | | `gap` | no | Python adapter: **not run** |

Notes:

- *Same step 0 on 1, 2 and 4 GPUs.* In the A100 logs of MACE, NequIP, SevenNet and MatterSim, the step 0 potential energy is the same
  on 1, 2 and 4 GPUs to about 1e-8 relative with TF32 off. (A BAM log shows only two completed starts.)
- *SevenNet energies look huge.* The SevenNet model that was run is **Omni with head `omol25_low`**, whose atomic reference energies
  are total DFT energies (about -4.8e7 kcal/mol for 3000 water atoms; the PBE materials models give about -3.4e5). This is expected
  and does not change the forces. Compare energies only between runs of the same model and head.
- CHGNet and MatterSim are trained on crystal data; whether they suit water was not checked. Choose a model trained for your system.
- Models that are not listed here (FeNNix-Bio1, GRACE, eSEN, eSCN, EquiformerV2) have no converter and no adapter.

### Limitations

- Everything marked **Random model** or **not run** is exactly that: expect to debug the first real run, and compare step 0 of the
  LibTorch MLIP against the Python MLIP (or the package itself) before trusting a long run
  (`examples/mlip_water100/reference_energy.py` does this comparison for water).
- No model weights are bundled with dsMD. You download them from the model's own source.

---

## 5. Multi-GPU MLIP

Several GPUs mean several MPI ranks. `mpirun -n 4 build_mlip/bin/dsMD input.yaml` with one GPU per rank

### 5.1 What you need

- **The Python MLIP gives no speed-up** with more ranks ([section 2.5](#25-several-mpi-ranks)). Use the LibTorch MLIP.
- **A CUDA-aware MPI**, or a dsDecompX built with NCCL. dsMD hands GPU memory to MPI directly. If you `export CUDA_AWARE_MPI=1` on an MPI
  that is not CUDA-aware, the first exchange crashes (Open MPI: `Read -1, expected N, errno = 14` and a segmentation fault in `mca_btl_vader`).
  Without that variable, a non-CUDA-aware MPI stops with `selected backend cannot migrate GPU-resident particles`. dsMD warns at
  start-up when the two disagree. dsMD never stages halo buffers through host memory.
- A dsMD and dsDecompX built from the same sources: layer exchange needs the `forward_exchange` call in dsDecompX, so rebuild and
  reinstall dsDecompX before dsMD. An old dsMD fails on a layered model with
  `field 'ghost_cutoff' (6.0) must be at least num_layers * cutoff (12.0)`.

### 5.2 Without layer exchange

dsMD divides the box into one subdomain per rank, as in any run. Each rank's model computes its own atoms plus ghost atoms
*layers × cutoff* deep, and the forces on the ghost atoms are sent back to their owners; energies are summed.

- The subdomains must be at least **ghost depth + skin + 1 Å** wide along each split axis. When a subdomain is narrower, dsMD
  prints `[MLIP] A subdomain is narrower than the ghost layer of the model ...` and every rank takes the atoms of all ranks and computes the whole
  system. That is correct but gives no speed-up.
- The cost of the ghost atoms means a small system or a deep model can be slower on more ranks than on one. At the first step dsMD prints
  `[MLIP]   per rank: owned ..., model atoms ..., edges ...` so you can see the overhead. Edges that cannot change the energy of the owned atoms are left out.

### 5.3 With layer exchange

Layer exchange removes the ghost depth. After each layer the ranks send the node features of their atoms to the ranks that hold copies
of them. Ghost atoms then need to be only **one cutoff** deep, and **every edge is computed on exactly one rank**.

```bash
PYTHONPATH=MLIP_tools python -m mlip_convert mace --model mace-mp-0-small.model -o models/libtorch_mace_layer --dtype float32 --layer-exchange
```

Use the folder as `mlip_model:` like any other. Nothing else changes in the input file. The first-step line then reads
`[MLIP] first step (layer exchange: ghost atoms one cutoff deep, node features sent between the layers) ...` and
`[MLIP]   per rank: ...; every edge is computed by one rank only (N edges in total)`: the edge counts of the ranks add up to the count of one rank.
A subdomain only has to be `cutoff + skin + 1 Å` wide, so more ranks fit into the same box.

| Model | `--layer-exchange` | Ghost depth without → with |
|---|---|---|
| MACE | yes (not a model with joint embedding) | layers × cutoff (12 Å) → 6 Å |
| SevenNet | yes, including multi-modal models with `--head` | 25 Å → 5 Å (SevenNet-0) |
| BAM | yes (`race` models without DeNS heads) | 30 Å → 6 Å |
| NequIP | yes (models built from `ConvNetLayer`) | layers × r_max → r_max |
| MatterSim | yes | 20 Å → 5 Å |
| TensorNet | yes (extensive models without bias) | (blocks + 1) × cutoff → cutoff. Re-convert older TensorNet folders: they were one cutoff short |
| PaiNN | yes | n_interactions × cutoff → cutoff |
| DeePMD | yes, but it is a local model: nothing is sent, each rank only computes its own atoms | cutoff |
| CHGNet | **refused** (bond and angle states live on edges shared by two atoms) | 24 Å |
| Allegro | **refused** (already local) | cutoff |

It needs TorchScript and edge shifts (the defaults). The folder runs on any number of ranks, including one.

### 5.4 Getting good speed

- Judge speed after the first two steps (the first two steps are slow, see the Limitations of [section 3](#3-libtorch-mlip)) and on a system that is large for the number of GPUs. A
  3000-atom water box is small for several GPUs; on 24000 atoms the benefit of more GPUs is larger.
- `advanced: tf32: true` helps float32 models on Ampere GPUs; read the TF32 caveats above. Results with TF32 depend slightly on the rank count.
- `advanced: timing_trace: every` prints one timing line per step; [07_Performance_Tuning](07_Performance_Tuning.md) explains the lines.
- Ready-made inputs for several models, 1000 and 8000 water molecules, with Slurm scripts, are in `examples/mlip_water_multigpu/`
  (one folder per model, with `convert.sh` and `slurm_water*.sh`). Their paths and cluster modules are those of the project's cluster; edit them for yours.

### Limitations

- Measured by the project (A100, MACE-MP-0 small, 3000 atoms, one node, CUDA-aware MPI): 67 ms per step on one GPU and 36 ms on four, about 1.7× to 1.85×. The
  exchange itself costs about 0.4 ms per step. The numbers of the other models are in the logs under `examples/mlip_water_multigpu/`.
- Verified: the same energy, pressure and largest force as one rank, on the development PC (small and toy models, up to 4 ranks) and in
  the cluster logs ([section 4](#4-supported-models)).
- **Not run**: the NCCL and NVSHMEM paths of dsDecompX for the layer exchange (the NCCL path uses the same send/receive code as the
  halo exchange; the NVSHMEM branch was compiled but never executed), several nodes, any case with more than 4 GPUs.
- CUDA-aware MPI is required; there is no fallback.

---

## 6. Tutorial: water box

This tutorial runs 100 water molecules (300 atoms) with an MLIP, with both backends. Everything you need is in
`examples/mlip_water100/`. Run commands from the repository root. The only model used here is **SevenNet-0**, because its
weights come inside the `sevenn` package: nothing is downloaded. (Other models: replace the conversion command with the one from [section 3.2](#32-converting-a-model).)

### Step 0. Environment

```bash
python -m venv ~/venvs/dsmd_sevenn && source ~/venvs/dsmd_sevenn/bin/activate
pip install torch sevenn numpy ase          # a torch with CUDA for the LibTorch part
```

The torch you install must match the LibTorch dsMD was built with ([section 3.1](#31-requirements-and-build)).
For the **Python MLIP** the environment's Python must be the version dsMD was built with. Keep this environment activated for all steps.

### Step 1. Build dsMD

CPU build for the Python MLIP only, or CUDA build for both ([01_Installation](01_Installation.md)):

```bash
# LibTorch + Python MLIP (CUDA)
cmake -S . -B build_mlip -DCMAKE_BUILD_TYPE=Release -DBACKEND=CUDA -DDSDECOMPX_DIR=<dsDecompX install> \
      -DENABLE_PYTHON=ON -DENABLE_LIBTORCH=ON -DLIBTORCH_PYTHON=$(which python)
cmake --build build_mlip -j
```

### Step 2. The system

`examples/mlip_water100/` already has `water100.xyz` (coordinates) and `water100.top` (types, masses, charges). The box is 14.4233 Å. To make
another size: `python examples/mlip_water100/make_water_box.py --molecules 100 --out water100` writes both files and prints the box edge for
`box_x/y/z`. The topology only needs the atom types and masses for an MLIP.

### Step 3. Convert the model (LibTorch MLIP)

```bash
PYTHONPATH=MLIP_tools python -m mlip_convert sevennet \
  --model $(python -c "import os, sevenn; print(os.path.dirname(sevenn.__file__))")/pretrained_potentials/SevenNet_0__11Jul2024/checkpoint_sevennet_0.pth \
  -o examples/mlip_water100/models/libtorch_sevennet --dtype float32
```

You should now have `model.pt`, `metadata.json`, `reference.json` in that folder. In `metadata.json`, `cutoff` is 5 Å and `ghost_cutoff` is
25 Å (5 layers): a 14.4 Å box is much smaller than that, which is why this example runs on **one rank**.

### Step 4. The input file

`examples/mlip_water100/input_libtorch.yaml` (shortened):

```yaml
coordinate_file: examples/mlip_water100/water100.xyz
topology_file:   examples/mlip_water100/water100.top

target_temperature: 300.0
timestep:           0.5          # fs; small because the water is flexible
total_steps:        100

box_x: 14.4233
box_y: 14.4233
box_z: 14.4233
periodic_x: true
periodic_y: true
periodic_z: true

output_prefix:           examples/mlip_water100/dsmd
output_freq:             10
output_info:             STEP, T, KE, PE, ETOTAL, FMAX
trajectory_output_freq:  10
trajectory_format:       xyz

cutoff_distance:           6.0
neighbor_list_type:        verlet_cell_list
verlet_skin:               1.0
neighbor_list_update_freq: 10

integrator: velocity_verlet
thermostat: nose_hoover
tau_t:      100.0                # fs
barostat:   none

force_calculation:
  mode: python                   # MLIP on, built-in force field off

mlip_backend: libtorch
mlip_model:   examples/mlip_water100/models/libtorch_sevennet
```

### Step 5. Run

```bash
mpirun -n 1 build_mlip/bin/dsMD examples/mlip_water100/input_libtorch.yaml
```

### Step 6. Check the result

In the log, look for the lines from [section 3.4](#34-the-input-file) (`MLIP backend`, `MLIP model`, `MLIP elements`), then the energy lines:

```text
Step: 0, T: 299.9..., KE: ..., PE: ..., Etot: ..., Fmax: ...
```

- `T` should stay near 300 K (Nosé–Hoover). A temperature that explodes within the first hundred steps means wrong forces: check
  the elements line, the torch versions, and the kernel (section 3.6).
- Compare step 0 with the model's own calculation. With the environment activated:

  ```bash
  python examples/mlip_water100/reference_energy.py --model sevennet \
    --path <path of checkpoint_sevennet_0.pth> --device cuda --log <your dsMD log>
  ```

  It prints the potential energy and the largest force of the starting structure through the Python MLIP and the difference to your log.
  For float32 models the two should agree to about 1e-4 relative.
- The trajectory is `examples/mlip_water100/dsmd_*.xyz`.

### Step 7. The same run through the Python MLIP

`examples/mlip_water100/input_python_e3nn.yaml` differs only in the MLIP lines:

```yaml
force_calculation:
  mode: python

mlip_backend:     python
python_mlip_path: MLIP_tools/mlip_python

mlip_config:
  model_type: sevennet
  model_path: 7net-0
  device: cuda
  dtype:  float32
  accel:  e3nn
```

```bash
mpirun -n 1 build_mlip/bin/dsMD examples/mlip_water100/input_python_e3nn.yaml
```

The start-up shows `[MLIPInterface] ✅ Active MLIP: SevenNetAdapter`. Step 0 energy should match the LibTorch run to the printed precision of float32.
It will run noticeably slower; that is expected (Limitations of [section 2](#2-python-mlip)).

### Step 8. Scale up to several GPUs

A 300-atom box cannot be divided. Use a bigger box and a layer-exchange model:

1. Take a larger water box (`examples/mlip_water_multigpu/*/water1000.xyz` has 3000 atoms in a 31.07 Å box; `tile_water.py` in
   `examples/mlip_water1000/` repeats a box to make 24000 atoms).
2. Convert with `--layer-exchange` (for example `examples/mlip_water_multigpu/sevennet/convert.sh`).
3. Point `mlip_model:` to that folder and run `mpirun -n 4 build_mlip/bin/dsMD input.yaml` on a node with four GPUs and a CUDA-aware MPI.
   The first-step line should say `layer exchange`.

### Limitations

- The steps above are taken from the scripts and input files in `examples/mlip_water100/`. The same flow was run on the project's
  cluster with MACE-OMAT-0 small (both backends, three kernels) and with SevenNet-Omni (layered, several GPUs); the SevenNet-0 variant on this page,
  as one sequence, was not run end to end for this document. If a command differs on your machine (paths, `mpirun` versus `srun`, modules),
  adapt it, but keep the order: convert, check `metadata.json`, run, compare step 0.
- The 100-water box is for correctness checks, not timing: it has one rank and a very deep ghost layer for SevenNet.
