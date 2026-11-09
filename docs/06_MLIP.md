# 06. Machine-Learned Potentials (MLIP)

## Contents

1. [Overview and choosing a path](#1-overview-and-choosing-a-path)
2. [ASE mode](#2-ase-mode)
3. [LibTorch and PyTorch modes](#3-libtorch-and-pytorch-modes)
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

### Three ways to run a model

dsMD has three modes for an MLIP. You choose one with `mlip_backend:` in the input file. The name says what runs the model.

| | **ASE mode** (`mlip_backend: ase`) | **LibTorch mode** (`mlip_backend: libtorch`) | **PyTorch mode** (`mlip_backend: pytorch`) |
|---|---|---|---|
| What computes the forces | the model package's own ASE calculator | the converter's wrapper: the energy of the owned atoms, differentiated | the same wrapper |
| What runs it | the Python interpreter inside dsMD | LibTorch in C++, from a saved TorchScript or AOTI file | the Python interpreter inside dsMD, from the model's own code |
| Conversion | none | `mlip_convert --backend torchscript` (or `aoti`) | `mlip_convert --backend pytorch` (writes no model file) |
| Builds | CPU and CUDA builds | **CUDA builds only**, configured with `-DENABLE_LIBTORCH=ON` | **CUDA builds only**, configured with `-DENABLE_LIBTORCH=ON` and `-DENABLE_PYTHON=ON` |
| Model files | the original files of the package (a `.model`, a checkpoint, or a name the package knows) | a folder with `model.pt`, `metadata.json`, `reference.json` made by `mlip_convert` | a folder with `metadata.json`, `reference.json`; the original model file is read again when dsMD starts |
| Python needed while running | yes (the model's packages must be importable) | not for e3nn kernels. cuEquivariance and OpenEquivariance models import their package once at start-up through the embedded Python | yes (the model's packages, the same environment the folder was converted in) |
| Speed | slow: positions and forces cross the host every step | the production path: everything stays on the GPU | the same data path as LibTorch mode (everything stays on the GPU); about the same speed with the same kernels in the project's MACE tests |
| Several GPUs / MPI ranks | correct, but no speed-up (every rank computes the whole system) | the box is divided between ranks; see [section 5](#5-multi-gpu-mlip) | the same as LibTorch mode |
| Models | 13 models have an adapter, see [section 2](#2-ase-mode) | 11 models have a converter, see [section 3](#3-libtorch-and-pytorch-modes) | the same 11, and UMA, eSEN and ORB, whose code is not TorchScript ([section 4](#4-supported-models)) |

Pick like this:

- *You want to try a model, compare it with its own package, or run on CPU* → **ASE mode**.
- *You want speed, or several GPUs* → **LibTorch mode** (needs a CUDA build and a model with a converter).
- *The model has a converter but its code does not become a TorchScript file, or the TorchScript file fails* → **PyTorch mode**.
- *UMA, eSEN or ORB* → **PyTorch mode** (their code does not become a TorchScript file), or the ASE mode.
- *SO3LR Tauri* (a JAX model) → **LibTorch mode**: the converter builds it again in PyTorch from the released weights. It has no ASE mode.
- *The model has no converter* (GAP, MatGL M3GNet) → **ASE mode** is the only way.

The tensor-product kernels (e3nn, cuEquivariance, OpenEquivariance, FlashTP) are chosen separately in all modes:
[section 3.6](#36-kernels-e3nn-cueq-openeq-flash).

### The input file keys for an MLIP

These keys are specific to MLIP runs. All other keys are explained in [03_Input_Reference](03_Input_Reference.md).

```yaml
force_calculation:
  mode: mlip              # "MLIP on, built-in force field off". See the note below.

mlip_backend: libtorch    # ase (default) | libtorch | pytorch
```

> **`mode: mlip` only switches the MLIP on** and the built-in force field off, for all three modes. How the model runs is
> `mlip_backend`. `mode: ase`, `mode: libtorch` and `mode: pytorch` are rejected with a message that says so.

| Key | Used by | Meaning |
|---|---|---|
| `force_calculation.mode` | all | `mlip` = run the MLIP (no built-in force field). `builtin` = the classical force field |
| `mlip_backend` | all | `ase` (default when absent), `libtorch` or `pytorch` |
| `mlip_model` | libtorch, pytorch | the model folder written by `mlip_convert` (required). A folder with `model.pt` or `model.pt2` goes with `libtorch`, a folder converted with `--backend pytorch` goes with `pytorch`; if they do not match, dsMD stops at start-up and says which value to use |
| `python_mlip_path` | ase | folder that holds `ml_forces.py`; default `mlip_python` |
| `mlip_config:` | ase | section with `model_type`, `model_path`, `model_size`, `device`, `dtype`, `accel`, and model-specific keys ([section 2](#22-the-mlip_config-section)) |
| `mlip_elements:` | all | which element each atom type is, only when dsMD cannot tell ([below](#which-element-is-each-atom-type)) |
| `advanced.tf32` | libtorch, pytorch | run float32 matrix products as TF32 on the GPU (default `false`); ignored with a message by the ASE mode |
| `advanced.timing_trace` | all | per-step timing lines: `off` (default), `step`, `every` |

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

- The model must know every element in your system. The converter writes the supported elements into `metadata.json`; the ASE
  mode leaves this check to the model package.
- Only the settings listed above are MLIP-specific. A comma list in `force_calculation.mode` (`builtin, mlip`) is accepted by the
  parser for classical + MLIP forces together; it is not covered here.

---

## 2. ASE mode

The ASE mode (`mlip_backend: ase`) needs no conversion. dsMD starts a Python interpreter inside the program, imports `MLIP_tools/mlip_python/ml_forces.py`, and every step hands it the positions, atom
types, box and periodic flags. The adapter for your model (a small class in `MLIP_tools/mlip_python/adapters/`, one file per model) wraps the
model's own ASE calculator, and the forces come back to dsMD. Any failure in Python (a missing package, a bad model file, an error in
a force call) stops the run with the Python traceback. It never turns into zero forces.

### 2.1 Requirements

- dsMD built with `-DENABLE_PYTHON=ON` (the default of CMake). Of the build scripts in the repository root, only `build_cuda.sh` turns it on;
  `build.sh` and `build_cuda_gh200.sh` pass `-DENABLE_PYTHON=OFF`. A build without it refuses `mlip_backend: ase` at start-up.
- A Python environment with NumPy and the packages of your model, **activated before you run dsMD**. dsMD adds the activated
  environment's packages (`VIRTUAL_ENV`, `CONDA_PREFIX`) to the embedded interpreter. That environment must use the **same Python
  version** dsMD was built with.
- For a GPU run, a CUDA build and `device: cuda` in `mlip_config`.

### 2.2 The `mlip_config` section

```yaml
force_calculation:
  mode: mlip

mlip_backend:     ase
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
| `esen` | `fairchem-core` | an eSEN checkpoint of fairchem 2.x (required; the weights are gated) | `task_name`: only for a checkpoint with several tasks |
| `orb` | `orb-models` | not used (the package loads ORB v2) | |
| `gap` (`quip`) | `quippy-ase` | the GAP `.xml` file (required) | `args_str` |
| `painn` (`schnetpack`) | `schnetpack` | a schnetpack `best_model` file (required; the same file the converter takes) | `energy_unit`: the unit the model was trained in (default `eV`) |

`model_type: auto` guesses the adapter from words in `model_path`; naming the type explicitly is safer.

### 2.3 Getting a model file

You can get the file by hand, from the model's own website or package. Two helpers exist in the repository:

- `PYTHONPATH=MLIP_tools python -m mlip_cli_ui` is a numbered menu: choose the MLIP, choose "get the model file for the ASE mode", pick a file from
  Hugging Face, GitHub, the installed package or your disk. It saves the file and writes a `mlip_config.yaml` snippet next to it.
  **That snippet is not read by dsMD**: copy its lines (`mlip_backend:` and the `mlip_config:` block) into your input file.
  The menu is described in [section 3.3](#33-the-menu-mlip_cli_ui).
- NequIP and Allegro need one extra step: compile the file for the ASE mode:
  `nequip-compile <model> <out>.nequip.pt2 --mode aotinductor --device cuda --target ase` (with torch older than 2.10:
  `--mode torchscript` and the name `<out>.nequip.pth`).

A MACE run that downloads its foundation model (`model_path` empty) needs internet access on the machine that runs dsMD.

### 2.4 Example

`examples/mlip_water100/` has the whole thing ([tutorial in section 6](#6-tutorial-water-box)). The MLIP part of
`input_python_e3nn.yaml` (the file names of the ASE mode examples still say `python`):

```yaml
force_calculation:
  mode: mlip

mlip_backend:     ase
python_mlip_path: MLIP_tools/mlip_python

mlip_config:
  model_type: sevennet
  model_path: 7net-0              # SevenNet-0 ships inside the sevenn package
  device: cuda
  dtype:  float32
  accel:  e3nn
```

### 2.5 Several MPI ranks

The ASE mode works the same way with any number of ranks: every rank collects the atoms of all ranks, runs the model on the
whole system, and keeps the forces of its own atoms. The result is correct, but more ranks do not make it faster. For several GPUs
use the LibTorch or PyTorch mode.

### Limitations

- **Speed.** Every step copies positions to the host and forces back. Use it for correctness checks and model comparisons, not
  for long production runs.
- `advanced.tf32` has no effect (dsMD prints a message and ignores it).
- Only MACE and SevenNet were run through dsMD in this project with real model files (see [section 4](#4-supported-models)). The
  other adapters are written but were not exercised on real checkpoints here; treat them as untested until you check step 0
  against the package itself.
- GAP can be used only through the ASE mode (no converter).

---

## 3. LibTorch and PyTorch modes

The converter `mlip_convert` turns a model into a **model folder**. In the **LibTorch mode** (`mlip_backend: libtorch`) dsMD loads the
folder in C++ with LibTorch and runs the model on the GPU together with the MD, so no Python is in the force loop. This is the fast
path and the one for several GPUs. In the **PyTorch mode** (`mlip_backend: pytorch`, [section 3.8](#38-pytorch-mode)) the folder has no model
file, and the Python inside dsMD runs the model; the GPU data path and the several-GPU support are the same.

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

`<model>` is one of: `mace`, `nequip`, `allegro`, `sevennet`, `painn`, `chgnet`, `mattersim`, `deepmd`, `tensornet`, `bam`, `so3lr`, and, for the
PyTorch mode only, `uma`, `esen` and `orb`.
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

# UMA (gated weights from Hugging Face facebook/UMA), eSEN (gated, facebook/OMol25) and ORB: PyTorch folders only (--backend pytorch)
PYTHONPATH=MLIP_tools python -m mlip_convert uma --model uma-s-1p1.pt --head omat --composition H2000O1000 --backend pytorch -o models/pytorch_uma --dtype float32
PYTHONPATH=MLIP_tools python -m mlip_convert esen --model esen_sm_conserving_all.pt --backend pytorch -o models/pytorch_esen --dtype float32
PYTHONPATH=MLIP_tools python -m mlip_convert orb --model orb-v3-conservative-inf-omat-20250404.ckpt --backend pytorch -o models/pytorch_orb --dtype float32

# SO3LR Tauri: a model folder of the so3lr repository (git clone https://github.com/general-molecular-simulations/so3lr), numpy only
PYTHONPATH=MLIP_tools python -m mlip_convert so3lr --model so3lr/so3lr/models/so3lr-tauri-m-2606 -o models/libtorch_so3lr --dtype float32
```

Options:

| Option | Meaning |
|---|---|
| `-o`, `--output` | output folder (required) |
| `--model` | the model file (or folder/address, depending on the model). SO3LR: the model folder (`params.pkl` + `hyperparameters.json`) or its `params.pkl`, or a model name of the installed so3lr package |
| `--dtype float32\|float64` | model precision. **The default is `float64`; use `float32` for MD**, as the examples do. NequIP and DeePMD models carry their own precision and refuse a different one |
| `--accel e3nn\|cueq\|openeq\|flash` | tensor-product kernels ([section 3.6](#36-kernels-e3nn-cueq-openeq-flash)). With `cueq`, `openeq`, `flash` also pass `--device cuda`; convert on the GPU machine |
| `--head <name>` | the head (MACE) or modal (SevenNet) of a model that has several. `--list-heads --model <file>` prints them. A multi-modal SevenNet model (Omni, MF-0) cannot be converted without one. UMA: the task (`omat`, `omol`, `oc20`, `odac`, `omc`). eSEN: the task, only if the checkpoint has several (the released ones have one). ORB: the orb-models name (`orb-v3-conservative-inf-omat`, ...) when the file name does not contain it |
| `--composition <formula>` | UMA only, required: the formula of the **whole** MD system with its atom counts, e.g. `H2000O1000`. UMA mixes its experts by the composition of the system; the converter fixes the mixture for this composition (as fairchem's own `merge_mole` setting does), because a rank sees only part of the system. Convert again for another composition. eSEN: not needed (no experts); the converter asks for it only if the task's energy has a mean, which it splits over the atoms |
| `--charge`, `--spin` | UMA, eSEN: total charge and spin multiplicity (defaults: 0, and fairchem's default spin of the task). ORB omol models: both are required. SO3LR: total charge and spin multiplicity (defaults 0 and 1). Fixed in the converted folder |
| `--layer-exchange` | for several GPUs ([section 5](#5-multi-gpu-mlip)) |
| `--no-edge-shifts` | use the older, slower way of handling periodic images (extra atoms instead of shifts); only if you need models that run on an older dsMD. Cannot be combined with `--layer-exchange` |
| `--elements 1,8` | the atomic numbers the model knows. Needed for PaiNN, and for a NequIP model whose type names are not element symbols |
| `--cutoff` | cutoff in Å. Real models use their own cutoff (for example MACE's `r_max`); this is for the toy test models. SO3LR: the long-range cutoff of its electrostatics and dispersion (default 12 Å, so3lr's own default for periodic boxes; 100 Å for a molecule in vacuum) |
| `--backend torchscript\|aoti\|pytorch` | default `torchscript`; see [section 3.7](#37-aoti-backend) and [section 3.8](#38-pytorch-mode). `torchscript` and `aoti` folders are run with `mlip_backend: libtorch`, a `pytorch` folder with `mlip_backend: pytorch` |
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

The steps: choose the MLIP → choose what to do (**convert to a model folder**, or **get the model file for the ASE mode**) →
choose the model file (found in Hugging Face repos, GitHub releases, the installed package, or typed by you) → options (precision,
kernel, head, layer exchange) → output folder. Before it runs anything it shows the exact command and asks for confirmation.
It prints the lines to put into your input file at the end. Other arguments: `--python /path/to/python` (the Python that does the
conversion), `--download-dir <folder>` (default `models/downloads`), `--offline`. Tokens: `HF_TOKEN` for gated Hugging Face
repositories (UMA), `GITHUB_TOKEN` if GitHub rate-limits you. Some Hugging Face repositories listed are third-party mirrors; the menu shows the
repository name, so check the source and licence before use.

### 3.4 The input file

```yaml
force_calculation:
  mode: mlip                       # turns the MLIP on (all three modes)

mlip_backend: libtorch             # pytorch for a folder converted with --backend pytorch
mlip_model:   models/libtorch_sevennet       # the folder made by mlip_convert

# optional
# mlip_elements:
#   OW: O
#   HW: H

# advanced:
#   tf32: false                    # float32 matrix products in TF32 (Ampere GPUs and newer)
```

`mlip_config:` and `python_mlip_path:` are not used by these modes. The rest of the file is an ordinary MD input; the complete
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

### 3.6 Kernels: e3nn, cueq, openeq, flash, so3lr_fused

Equivariant models can use faster tensor-product kernels. You choose them when you convert (`--accel`), and in the ASE mode with `accel:`.

| Kernel | `--accel` | Models | What it needs |
|---|---|---|---|
| e3nn | `e3nn` (default) | all | nothing extra |
| cuEquivariance | `cueq` | MACE, SevenNet, BAM | Python packages `cuequivariance`, `cuequivariance-torch` and the matching `cuequivariance-ops-torch-cu12` (or cu11/cu13); a GPU; dsMD built with `-DENABLE_PYTHON=ON`, because dsMD imports `cuequivariance_ops_torch` and `cuequivariance_torch` once at start-up so their operators are registered. Before that import dsMD sets `CUEQUIVARIANCE_OPS_U1D_ALGOS=0` unless it is already set: it fixes the kernel choice, so cuEquivariance does not tune again each time the ghost count changes on several GPUs |
| OpenEquivariance | `openeq` | MACE, SevenNet, BAM (BAM: `interaction_block: fast` checkpoints only) | Python package `openequivariance`; on first import it **compiles a CUDA extension**, so a C++ compiler, `ninja` and `nvcc` must be on `PATH` (also inside the job); dsMD built with `-DENABLE_PYTHON=ON` |
| FlashTP | `flash` | SevenNet only | the `flashTP_e3nn` package built for your GPU |
| SO3LR attention operator | `so3lr_fused` | SO3LR only | a GPU, and `nvcc`, a C++ compiler and `ninja` on `PATH` when converting: the converter builds the sources in `MLIP_tools/mlip_convert/so3lr_ops/` as a torch extension (its own name per source version, under `TORCH_EXTENSIONS_DIR`) and writes the library into `required_op_libs`. dsMD only loads it. Plain SO3LR (`e3nn`) needs nothing extra |

Things to know:

- NequIP and Allegro get their kernels when you compile the model with the nequip tools, not through `--accel`.
- `so3lr_fused` replaces these parts of SO3LR by CUDA operators with their own backward kernels:
  - the attention with its filters, in one kernel per layer: each edge's filters are made inside the kernel (no per-edge filter
    arrays in the forward or the backward), then the q·w·k sums and the sums onto the atoms
  - the radial filters: each layer's two radial filter networks are tabulated at conversion in the basis distance (2048 nodes,
    cubic Hermite), so no radial basis and no radial filter layers are computed per edge. The paper's inference code does the same.
    The spherical filter networks are evaluated per edge as they are.
  - the long-range pair terms (electrostatics and dispersion on the 12 Å edges, with the derivatives in positions, edge shifts,
    charges and dispersion ratios)
  - the contraction of the Euclidean variables that feeds the spherical filters
  - the per-edge geometry: spherical harmonics, basis distance (NLH distance floor), cutoff function and NLH repulsion in one
    kernel, with the hand-written derivative in the edge vector

  The kernels that sum edge terms onto atoms first add up, inside a warp, the edges that go to the same atom (dsMD gives the edges
  grouped by atom). It needs 32 channels per attention head and as many degree heads as heads, which every so3lr-tauri model has. Against the plain
  model in float64 the energy agrees within 1e-10 relative and the forces within 1e-7 of the largest force (the table: values 1e-9,
  derivatives 3e-7 of the filter scale); in float32 the difference is below the float32 noise. A folder converted with it needs a
  dsMD that knows the name `so3lr_fused`.
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

### 3.8 PyTorch mode

`--backend pytorch` writes no model file, only `metadata.json` and `reference.json`:

```bash
PYTHONPATH=MLIP_tools python -m mlip_convert mace --model models/mace-mp-0-small.model -o models/pytorch_mace --backend pytorch --dtype float32
```

```yaml
force_calculation:
  mode: mlip

mlip_backend: pytorch
mlip_model:   models/pytorch_mace
```

The model is run by the Python inside dsMD: dsMD builds the same wrapper the converter would save, from the original model file and the
settings in `metadata.json`, and calls it every step without TorchScript. Everything else is the same as for a TorchScript folder:
ghost atoms, edge shifts, layer exchange on several GPUs, the e3nn, cueq and openeq kernels. The positions and forces are not copied
between dsMD and Python (they stay on the GPU).

Use it for a model whose code does not convert to TorchScript, or to compare with the TorchScript run. Needs a dsMD built with
`-DENABLE_PYTHON=ON` and the model's environment activated when dsMD runs (the same one the folder was converted in). The folder
keeps the absolute paths of the model file and of `MLIP_tools`: do not move them.

`mlip_backend` has to match the folder. `mlip_backend: pytorch` with a TorchScript folder, or `libtorch` with a PyTorch folder, stops at
start-up with a message that names the other value.

### 3.9 Pressure tensor and NPT

The pressure needs the virial. For a model without edge shifts dsMD sums r·F over the atoms the model saw itself. A model with edge
shifts returns it: the wrapper scales the x, y and z components of the positions and shifts by their own small strain and returns
W_xx, W_yy, W_zz (metadata `virial_diagonal`). With them every barostat coupling works with an MLIP: `isotropic`, `anisotropic`
and `semiisotropic` (membranes: `barostat_axes: x y`), and the output fields `PXX`, `PYY`, `PZZ`. A folder converted before
`virial_diagonal` existed returns only the trace: it still runs isotropic NPT and asks to be converted again for the others. The
ASE mode gives the trace only.

### Limitations

- **CUDA builds only.** CPU builds refuse `mlip_backend: libtorch` and `pytorch`. Use the ASE mode there.
- Models without a converter cannot be used: GAP, MatGL M3GNet, eSCN/EquiformerV2, fairchem 1.x eSEN, FeNNix-Bio1, GRACE
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

"Converter" means `PYTHONPATH=MLIP_tools python -m mlip_convert <model>` (LibTorch and PyTorch modes). "ASE mode" means an adapter exists for `mlip_backend: ase`.
The last column says what was actually run, so you know what to check yourself.

**Legend for verification**

- **Real weights, A100**: ran in dsMD on the project's cluster with a real model; the logs are kept in `examples/`. Only what the logs show is claimed: the step 0 energy and largest force compared with other runs of the same system.
- **Real weights, dev PC**: ran in dsMD or against the model's own package on the small development machine.
- **Random model**: the adapter was checked against the package's own calculator with a randomly initialized model, not a real checkpoint.
- **Not run**: written, but not exercised.

| Model | Converter | Kernels (converter) | ASE mode (`model_type`) | Layer exchange (several GPUs) | What was verified |
|---|---|---|---|---|---|
| MACE (MP-0, OMAT, ...) | `mace` | e3nn, cueq, openeq | `mace`, `mace_mp`, `mace_off` | yes (not for models with joint embedding) | **Real weights, A100**: MACE-OMAT-0 small, water 3000 atoms, libtorch e3nn and openeq, ASE mode with e3nn, cueq and openeq: the step 0 energy and largest force agree (energy within 1e-5 relative, largest force 23.6657); layered model on 1, 2, 4 GPUs. **Real weights, dev PC**: MACE-MP-0 small, libtorch against the ASE mode. libtorch + cueq with real kernels: **not run** |
| NequIP | `nequip` | e3nn (others by `nequip-compile`) | `nequip` (compiled file) | yes | **Real weights, A100**: NequIP-OAM-L, layered, 1, 2 and 4 GPUs at 3000 atoms (out of memory at 24000 atoms) |
| Allegro | `allegro` | e3nn | `allegro` (compiled file) | no (strictly local, nothing to gain) | Same path as NequIP; no run with real Allegro weights in this repository |
| SevenNet | `sevennet` | e3nn, cueq, openeq, flash | `sevennet`, `7net` | yes (also multi-modal) | **Real weights, A100**: SevenNet Omni (head `omol25_low`), openeq, layered, 1, 2, 4 GPUs. **Real weights, dev PC**: SevenNet-0 with the ASE mode. `flash`: **not run** (FlashTP could not be installed) |
| MatterSim | `mattersim` | e3nn | `mattersim` | yes | **Real weights, A100**: MatterSim 1M, layered, 1, 2, 4 GPUs. ASE adapter: **not run** |
| BAM (RACE) | `bam` | e3nn, cueq (cuEquivariance-trained only), openeq (`fast` blocks only) | `bam` | yes | **Real weights, A100**: BAM-MP-core, e3nn, layered; several runs ran out of memory on 40 GB. cueq: **not run**. openeq (BAM-MP-DeNS, `oeq_conv_fusion`): layered conversion on Hangang, step-0 energy and forces equal the e3nn layered model (force difference 1e-6); peak GPU memory per edge 0.60 MB with e3nn kernels and 0.15 MB with the fused kernel (measured on one GH200); the e3nn model runs out of memory on one GH200 at 3000 atoms |
| TensorNet (matgl) | `tensornet` | e3nn | `matgl`, `tensornet` | yes | **Random model** only. ASE adapter: **not run** |
| PaiNN (schnetpack) | `painn` | e3nn | `painn` | yes | **Random model** only (no public pretrained PaiNN potential); the ASE adapter gives the converter's energy and forces on the same random model |
| DeePMD-kit (PyTorch) | `deepmd` | e3nn | `deepmd` | local model: ranks compute only their own atoms | **Random model** only; strictly local descriptors only |
| CHGNet | `chgnet` | e3nn | `chgnet` | **no** | Converter checked with the weights shipped in the CHGNet package; ASE adapter: **not run**. Needs a ghost depth of 24 Å |
| UMA (fairchem) | `uma`, PyTorch mode only | e3nn | `uma` | yes | **Random model** only (gated weights). Dev PC: the converted model gives the energy and forces of fairchem's own calculator, with its experts not merged, to float32 accuracy (largest force difference 4e-5 eV/Å, largest force 11 eV/Å). **GH200 (Hangang)**, a random model of UMA-S size (146 M parameters, 32 experts), water 3000 atoms: dsMD step 0 forces equal fairchem's calculator to 7e-6 of the force rms (float32 and float64); the energy differs by 2e-4 kcal/mol per atom, which is fairchem's float32 sum of the element references; the same on 1, 2 and 4 GPUs; the ASE mode equals fairchem. float64 at 3000 atoms and float32 at 24000 atoms run out of memory on 95 GB and finish in the low-memory mode (edges in chunks). Speed (float32, one GPU): 0.46 s/step at 3000 atoms, 4.9 s/step at 24000 atoms (low-memory mode), the same as fairchem's own calculator on the same GPU (0.45 s with dsMD's settings, 5.3 s with fairchem's default; fairchem's turbo setting with TF32 takes 0.26 s); before a fix of 2026-11-07 (frozen weights made torch.matmul run fairchem's Linears as batched products, see the GPU memory note) it was 1.06 and 12.3 s/step; four GPUs on 24000 atoms are no faster without layer exchange (every rank computes all atoms). **Layer exchange** (PyTorch mode, up to 8 nodes, measured before that fix): 24000 atoms on 8 GPUs gives the step 0 energy of one GPU to all printed digits; 24000 atoms 12.0 s/step on 1 GPU, 0.31 s/step on 32 GPUs; 192000 atoms 24.9 s/step on 4 GPUs, 3.17 s/step on 32 (7.9 times for 8 times the GPUs). Without layer exchange 192000 atoms run out of memory on 1 to 8 GPUs (benchmarks/mlip/fairchem). NVE 1820 steps: Etot changes by 10 kcal/mol out of 37154 while the random model heats to 2300 K |
| eSEN (fairchem 2.x, conservative) | `esen`, PyTorch mode only | e3nn | `esen` | yes | **Random model** only (gated weights). Dev PC: the converted model equals fairchem's calculator, also with a charge and spin. **GH200 (Hangang)**, a random model of eSEN-sm size (6.0 M parameters), water 3000 atoms: as UMA (forces 7e-6 of the force rms, energy 2e-4 kcal/mol per atom, 1, 2, 4 GPUs the same, ASE mode equal, low-memory mode at float64 3000 and 24000 atoms). Speed: 0.45 s/step at 3000 atoms, 4.9 s/step at 24000 atoms (1.05 and 12.3 before the fix). Layer exchange (before the fix): as UMA (24000 atoms: step 0 energy of 8 GPUs within 9e-8 of one GPU; 0.31 s/step on 32 GPUs; 192000 atoms 24.8 s/step on 4 GPUs, 3.11 on 32). NVE 1000 steps: Etot changes by 24 kcal/mol out of 34918 while the random model heats to 6900 K |
| ORB (orb-v3 conservative, `inf` models) | `orb`, PyTorch mode only | e3nn | `orb` | no; **whole system**: with several ranks every rank computes all atoms | **Random model** only: the converted model gives the energy and forces of orb-models' own calculator. **Not run in dsMD.** ASE adapter: **not run** |
| SO3LR Tauri (`so3lr-tauri-s/m/l`, releases 2602 and 2606) | `so3lr` | e3nn (plain PyTorch), so3lr_fused | none | yes (also the charges and the total-charge sum go between the ranks); without it **whole system**: every rank computes all atoms | **Real weights, dev PC (CPU)**: the converted model gives the energy, forces and virial of so3lr's own JAX calculator (float64) for s, m and l 2606 and s 2602: molecules in vacuum (with a close contact), a periodic cell with images as atoms and as edge shifts, charged and open-shell molecules; energy within 1e-8 relative, forces within 1e-7 of the largest force. **Real weights, GH200 (Hangang)**: so3lr-tauri-m, water 3000 atoms, libtorch: the step 0 energy and forces of dsMD equal the converter's own model (float64: energy 2e-12 relative, forces 2e-8 eV/Å; float32: 4e-8 and 1e-4 eV/Å); the same on 1, 2 and 4 GPUs; NVE 2 ps stable (from an unequilibrated start). Speed (float32, one GPU, 3000 atoms): m 30 ms/step over 4000 steps; in 200-step runs, which include the start-up, s 51, m 57, l 87 ms/step. 24000 atoms (m): 245 ms/step on one GPU, 225 on four (whole system). PyTorch mode (`--backend pytorch`): the same step 0 energy on 1 and 4 GPUs, forces within 5e-5 kcal/mol/Å rms of the LibTorch mode; 1000 NVT steps: 36.3 ms/step (LibTorch 36.9). With `--accel so3lr_fused` (m, float32, one GH200, TF32 off): 12.8 ms/step at 3000 atoms (NVE, 4000 steps; plain 30.2), about 30 ms/step at 24000 atoms in the steady state (nsys; plain about 245 including the start-up), i.e. 1.3 µs/atom/step; step 0 energy as the plain model, forces within 5e-5 of their rms (float32 noise), NVE 2 ps drift -9.2 meV/atom/ns (plain -7.8). Short runs overstate the cost per step: the first steps include TorchScript's optimization (seconds). TF32 on gave 3% more speed and 1% force error, so it stays off |
| GAP | none | | `gap` | no | ASE adapter: **not run** |

Notes:

- *Same step 0 on 1, 2 and 4 GPUs.* In the A100 logs of MACE, NequIP, SevenNet and MatterSim, the step 0 potential energy is the same
  on 1, 2 and 4 GPUs to about 1e-8 relative with TF32 off. (A BAM log shows only two completed starts.)
- *SevenNet energies look huge.* The SevenNet model that was run is **Omni with head `omol25_low`**, whose atomic reference energies
  are total DFT energies (about -4.8e7 kcal/mol for 3000 water atoms; the PBE materials models give about -3.4e5). This is expected
  and does not change the forces. Compare energies only between runs of the same model and head.
- CHGNet and MatterSim are trained on crystal data; whether they suit water was not checked. Choose a model trained for your system.
- Models that are not listed here (FeNNix-Bio1, GRACE, eSCN, EquiformerV2, the fairchem 1.x eSEN checkpoints such as eSEN-30M-OAM) have no converter and no adapter.
- *UMA* mixes its experts by the composition, charge, spin and task of the whole system. The converter fixes that mixture once
  (`--composition`, `--charge`, `--spin`, `--head`), so a converted UMA folder is for one composition. Its edge-degree embedding is
  one more hop than its layers: ghost atoms are (layers + 1) × 6 Å deep.
- *eSEN* (fairchem 2.x checkpoints of OMol25, OC25, ODAC25) is the network UMA is made of, without the mixture of experts: the
  converter reuses the UMA code, and one converted folder works for any system (no `--composition`). Only the `conserving` models
  are accepted: a `direct` model predicts its forces separately from its energy, so an MD run with it does not conserve energy.
- *UMA and eSEN GPU memory.* For the forces, the model keeps the intermediate values of every edge of every layer, about 220 kB
  per edge in float32 for a UMA-S-sized model (3000 water atoms, 257,000 edges: about 56 GB). dsMD (PyTorch mode) then switches on
  fairchem's own low-memory mode by itself: the edges are computed in chunks of 131,072 and computed again for the forces. Before
  the first step each rank estimates the memory of its edges and switches the mode on if they do not fit (`[MLIP] ... edges need
  about ... GB`). Without layer exchange, a step that still runs out of memory is repeated with the mode on (`[MLIP] out of GPU
  memory ...`); with layer exchange it stops with an error. The mode stays on. The results are the same; the steps are slower.
  The converter freezes the weights, and fairchem gives some of its Linear layers a strided input. With frozen weights,
  torch.matmul then runs such a layer as one small matrix product per edge, which made a step 2.3 times slower. The adapter
  therefore makes every Linear input contiguous first. The numbers stay the same, and so does the memory.
- *ORB* does not sum per-atom energies: its energy head averages the node features of **all** atoms of the system before its MLP,
  and it adds its ZBL repulsion as a mean over the atoms. The energy therefore reads the whole system, and the converter marks the
  folder `whole_system`: dsMD never splits it over ranks (every rank computes all atoms, as in the replicated case of
  [section 5.2](#52-without-layer-exchange)). Only the conservative `inf` models are converted: the direct-force models (orb-v2,
  orb-v3 direct) do not give forces that are the gradient of an energy, and the `20` models keep only 20 neighbors per atom, which
  the edge list of dsMD (all pairs within the cutoff) does not reproduce.
- *SO3LR Tauri* is written in JAX. The converter does not run JAX: it builds the same network again in PyTorch and loads the released
  weights of a model folder (`params.pkl`, `hyperparameters.json`) and so3lr's reference tables, so converting needs numpy only and the
  result is an ordinary TorchScript folder. The energy is the network's (head 5, as so3lr reads it) plus NLH repulsion, electrostatics
  of its partial charges (damped, switched off at the long-range cutoff) and C6/C8/C10 dispersion (switched off over the last 2 Å).
  The edge list is built with the long-range cutoff (`--cutoff`, default 12 Å, so3lr's own default for periodic boxes); the short-range
  graph (4.5 Å, 5 Å for `-l`) is the part of it within the model cutoff. The partial charges are made to sum to the total charge over
  the **whole** system, so by default the folder is marked `whole_system` (as ORB: with several ranks every rank computes all atoms).
  Converted with `--layer-exchange`, the model is split over the ranks instead ([section 5.3](#53-with-layer-exchange)): each rank
  computes its own atoms, and the sums over the whole system (the charge sum, the charge and spin embeddings) are sums over the ranks
  (`dsmd::allreduce_sum`).
  **PME (`--kspace pme`)**: the electrostatics and the C6 dispersion are split Ewald-style, as so3lr does with its own k-space option
  (and as the SO3LR Tauri paper ran its liquids and biomolecules). The model keeps the real-space parts on its edges; dsMD computes the
  reciprocal sums of the model's charges and of the dispersion charges sqrt(C6) with smooth PME (operator `dsmd::kspace`,
  `src/MLIPKspaceOp.cpp`, new files only; other models and the classical PME are not touched). The smearings (4.0 Å and 2.0 Å by
  default, `--kspace-smearing`, `--kspace-dispersion-smearing`) are fixed at conversion; the grid spacings (2.0 Å and 3.0 Å by
  default) and the B-spline order (4) are written to `metadata.json` "kspace" and can be changed in the dsMD input. The input must
  say `mlip_kspace: pme`; a `--kspace pme` folder without it, or `mlip_kspace: pme` with a folder converted without it, stops at
  start-up with a message, because the long-range energy would be wrong. The box must be periodic along x, y and z. Works with the
  whole-system and the `--layer-exchange` folders (the grid is then summed over the ranks), and with the pressure tensor (the
  reciprocal energy's dependence on the box is added to the model's virial). Checked on the dev PC: so3lr's own k-space MD
  potential with Ewald sums (192-atom water, float64): energy within 2e-6, forces within 2e-5 of the largest; the virial against a
  finite difference of the energy; the C++ operator against the Python one (1e-9). GH200 (Hangang, SO3LR-T m, water 24000 atoms):
  step 0 (float64) of the whole-system folder and of the `--layer-exchange` folder on 1, 2, 4 and 8 GPUs identical to all printed
  digits; Pxx and Pzz against a finite difference of the energy (box and positions scaled by 1 +- 1e-4): 1.4e-6 and 1.5e-6
  relative; NVE 2 ps drift -5.8 (1 GPU) and -5.7 (4 GPUs) meV/atom/ns; semi-isotropic NPT with layer exchange on 4 GPUs keeps
  Lx = Ly; both mismatches stop at start-up. Cost of the k-space sums (default grids): +3 % (1 GPU) to +13 % (8 GPUs) per step
  at 192000 atoms, +13 to +26 % at 24000 atoms. The grid is summed over the ranks through host memory (MPI_Allreduce).
  so3lr's Ewald and P3M k-space methods and the legacy `so3lr` model are not supported. The Tauri weights are licensed CC BY-NC 4.0.
- *GAP* has no converter: it is not a PyTorch model (SOAP descriptors and a Gaussian-process kernel in Fortran, through quippy),
  so its energy has no autograd graph to differentiate and cannot be split per owned atom for several ranks.

### Limitations

- Everything marked **Random model** or **not run** is exactly that: expect to debug the first real run, and compare step 0 of the
  LibTorch mode against the ASE mode (or the package itself) before trusting a long run
  (`examples/mlip_water100/reference_energy.py` does this comparison for water).
- No model weights are bundled with dsMD. You download them from the model's own source.

---

## 5. Multi-GPU MLIP

Several GPUs mean several MPI ranks. `mpirun -n 4 build_mlip/bin/dsMD input.yaml` with one GPU per rank

### 5.1 What you need

- **The ASE mode gives no speed-up** with more ranks ([section 2.5](#25-several-mpi-ranks)). Use the LibTorch or PyTorch mode.
- **A whole-system model (ORB, SO3LR without `--layer-exchange`) gives no speed-up either**: its energy reads all atoms at once, so
  with several ranks every rank computes all atoms. dsMD says so at start-up. SO3LR converted with `--layer-exchange` is split over the
  ranks ([section 5.3](#53-with-layer-exchange)).
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
| SO3LR | yes: the node features between the layers, then the charges and a0 / C6 ratios before the long-range terms; the charge sum (and the charge and spin embeddings) are summed over the ranks (`dsmd::allreduce_sum`, needs a dsMD that has it). Without it: every rank computes all atoms | whole system → long-range cutoff (12 Å) |
| UMA, eSEN | yes (`--backend pytorch`). GPU memory: before the first step each rank estimates what its edges need and, if it is more than is free, runs its edges in chunks (a step that still runs out of memory stops with an error: one rank cannot run a step again while the others are in their exchanges) | (blocks + 1) × 6 Å (30 Å) → 6 Å |
| CHGNet | **refused** (bond and angle states live on edges shared by two atoms) | 24 Å |
| Allegro | **refused** (already local) | cutoff |

It needs a TorchScript or PyTorch folder and edge shifts (the defaults). The folder runs on any number of ranks, including one.

### 5.4 Getting good speed

- Judge speed after the first two steps (the first two steps are slow, see the Limitations of [section 3](#3-libtorch-and-pytorch-modes)) and on a system that is large for the number of GPUs. A
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

This tutorial runs 100 water molecules (300 atoms) with an MLIP, in the LibTorch and the ASE mode. Everything you need is in
`examples/mlip_water100/`. Run commands from the repository root. The only model used here is **SevenNet-0**, because its
weights come inside the `sevenn` package: nothing is downloaded. (Other models: replace the conversion command with the one from [section 3.2](#32-converting-a-model).)

### Step 0. Environment

```bash
python -m venv ~/venvs/dsmd_sevenn && source ~/venvs/dsmd_sevenn/bin/activate
pip install torch sevenn numpy ase          # a torch with CUDA for the LibTorch part
```

The torch you install must match the LibTorch dsMD was built with ([section 3.1](#31-requirements-and-build)).
For the **ASE mode** the environment's Python must be the version dsMD was built with. Keep this environment activated for all steps.

### Step 1. Build dsMD

CPU build for the ASE mode only, or CUDA build for both ([01_Installation](01_Installation.md)):

```bash
# LibTorch + ASE mode (CUDA)
cmake -S . -B build_mlip -DCMAKE_BUILD_TYPE=Release -DBACKEND=CUDA -DDSDECOMPX_DIR=<dsDecompX install> \
      -DENABLE_PYTHON=ON -DENABLE_LIBTORCH=ON -DLIBTORCH_PYTHON=$(which python)
cmake --build build_mlip -j
```

### Step 2. The system

`examples/mlip_water100/` already has `water100.xyz` (coordinates) and `water100.top` (types, masses, charges). The box is 14.4233 Å. To make
another size: `python examples/mlip_water100/make_water_box.py --molecules 100 --out water100` writes both files and prints the box edge for
`box_x/y/z`. The topology only needs the atom types and masses for an MLIP.

### Step 3. Convert the model (LibTorch mode)

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
  mode: mlip                     # MLIP on, built-in force field off

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

  It prints the potential energy and the largest force of the starting structure through the ASE mode and the difference to your log.
  For float32 models the two should agree to about 1e-4 relative.
- The trajectory is `examples/mlip_water100/dsmd_*.xyz`.

### Step 7. The same run through the ASE mode

`examples/mlip_water100/input_python_e3nn.yaml` differs only in the MLIP lines:

```yaml
force_calculation:
  mode: mlip

mlip_backend:     ase
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
It will run noticeably slower; that is expected (Limitations of [section 2](#2-ase-mode)).

The PyTorch mode is the Step 3 and Step 4 of the LibTorch mode with two changes: convert with `--backend pytorch`, and write
`mlip_backend: pytorch` ([section 3.8](#38-pytorch-mode)).

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
