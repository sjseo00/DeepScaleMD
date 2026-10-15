"""Acceleration kernels (design 6): what a model converted for cuEquivariance or OpenEquivariance needs at run time.

resolve_accel(kind) finds the installed packages and returns what goes into metadata.json:
  required_op_libs         absolute paths of shared libraries the C++ side dlopens before it loads the model
  required_python_modules  modules that must be imported because they register their operators from Python
                           (cuEquivariance, 11.3 / A.11-9: its operators are torch.library.custom_op definitions, so a
                           dlopen of its libraries does not register them)
  versions                 installed versions of the packages involved
The libraries are looked for without importing the ops packages (importing them needs the CUDA libraries); OpenEquivariance is
the exception: its library is built when the package is first imported, and its path is only known afterwards.
"""
import importlib
import importlib.metadata
import importlib.util
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

ACCELS = ("e3nn", "cueq", "openeq", "flash")


class AccelError(RuntimeError):
    pass


class AccelInfo(NamedTuple):
    required_op_libs: List[str]
    required_python_modules: List[str]
    versions: Dict[str, Optional[str]]


def package_version(*names: str) -> Optional[str]:
    """Version of the first installed distribution among `names`, or None."""
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _package_dir(module: str) -> Optional[Path]:
    """Directory of an installed top-level package, without importing it."""
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0]).resolve()


def _shared_libraries(directory: Path, pattern: str) -> List[str]:
    return sorted(str(path.resolve()) for path in directory.glob(pattern) if path.is_file())


def _resolve_cueq() -> AccelInfo:
    missing = [name for name in ("cuequivariance", "cuequivariance_torch") if _package_dir(name) is None]
    if missing:
        raise AccelError(f"--accel cueq needs the Python packages {', '.join(missing)} "
                         f"(pip install cuequivariance cuequivariance-torch)")
    ops_dir = _package_dir("cuequivariance_ops_torch")
    native_dir = _package_dir("cuequivariance_ops")
    if ops_dir is None or native_dir is None:
        raise AccelError("--accel cueq needs the cuEquivariance kernel package matching the CUDA version of torch "
                         "(pip install cuequivariance-ops-torch-cu12, or -cu11 / -cu13); it is not installed")
    libs = _shared_libraries(native_dir, "lib/*.so*") + _shared_libraries(ops_dir, "_ext/*.so")
    if not libs:
        raise AccelError(f"no shared library found in {native_dir} and {ops_dir}: the cuEquivariance kernel package is "
                         f"incomplete")
    versions = {
        "cuequivariance": package_version("cuequivariance"),
        "cuequivariance-torch": package_version("cuequivariance-torch", "cuequivariance_torch"),
        "cuequivariance-ops-torch": package_version("cuequivariance-ops-torch-cu12", "cuequivariance-ops-torch-cu13",
                                                    "cuequivariance-ops-torch-cu11"),
        "cuequivariance-ops": package_version("cuequivariance-ops-cu12", "cuequivariance-ops-cu13", "cuequivariance-ops-cu11"),
    }
    return AccelInfo(libs, ["cuequivariance_ops_torch"], versions)


def _resolve_openeq() -> AccelInfo:
    if _package_dir("openequivariance") is None:
        raise AccelError("--accel openeq needs the Python package openequivariance (pip install openequivariance; it "
                         "compiles a torch extension with g++ and the CUDA toolkit the first time it is imported)")
    try:
        extlib = importlib.import_module("openequivariance._torch.extlib")
        library = extlib.torch_ext_so_path()
    except Exception as error:      # noqa: BLE001  (the package's own build errors come in many classes)
        raise AccelError(f"openequivariance could not build or load its torch extension: {error}") from error
    path = Path(library)
    if not path.is_file():
        raise AccelError(f"openequivariance reports its extension at {library}, which does not exist")
    # The library holds the kernels, but the backward pass of its operators is registered by Python code when the package is
    # imported (torch.library.register_autograd). Without that import a C++ program gets correct energies and wrong forces
    # ("an autograd kernel was not registered ... silently incorrect behavior"), so the package is imported first like cueq's.
    return AccelInfo([str(path.resolve())], ["openequivariance"], {"openequivariance": package_version("openequivariance")})


def _resolve_flash() -> AccelInfo:
    """FlashTP (SNU-ARC/flashTP, Python package flashTP_e3nn; used by SevenNet). A model built for LAMMPS (sevenn deploy:
    `_flash_lammps`) calls operators that the package's flashtp_*.so files register when they are loaded; the LAMMPS build of
    SevenNet copies exactly these files and links them. No Python module is imported at run time."""
    package = _package_dir("flashTP_e3nn")
    if package is None:
        raise AccelError("--accel flash needs the Python package flashTP_e3nn (FlashTP, https://github.com/SNU-ARC/flashTP: "
                         "pip install . --no-build-isolation, with CUDA_ARCH_LIST set to your GPU); it is not installed")
    libs = _shared_libraries(package, "flashtp_*.so*")
    if not libs:
        raise AccelError(f"no flashtp_*.so found in {package}: the FlashTP package is incomplete or was built without its "
                         f"LAMMPS kernels")
    return AccelInfo(libs, [], {"flashTP_e3nn": package_version("flashTP_e3nn", "flashtp-e3nn", "flashTP-e3nn")})


def resolve_accel(kind: str) -> AccelInfo:
    if kind not in ACCELS:
        raise AccelError(f"unknown acceleration '{kind}'; supported: {', '.join(ACCELS)}")
    if kind == "e3nn":
        return AccelInfo([], [], {})
    if kind == "flash":
        return _resolve_flash()
    return _resolve_cueq() if kind == "cueq" else _resolve_openeq()
