"""What each source engine's converter needs: the input files, the extra options, the run input."""
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class FileSpec:
    flag: str                 # command-line option that takes the file
    label: str
    exts: Tuple[str, ...]     # extensions looked for when listing a folder (a path can always be typed)
    required: bool = True
    multiple: bool = False    # the option is repeated, one per file
    detail: str = ""


@dataclass(frozen=True)
class Engine:
    key: str
    title: str
    script: str
    files: Tuple[FileSpec, ...]
    run_input: str            # flag of the file that also gives input_dsmd.yaml
    ff_default: str = "unknown"
    box: bool = False         # --box LX LY LZ
    box_optional_with: str = ""      # flag of a file that can give the box; "" = the box is always optional
    dihedral_terms: bool = False
    e14fac: bool = False
    include_dirs: bool = False       # -I
    note: str = ""


ENGINES = {e.key: e for e in (
    Engine("namd", "NAMD", "convert_namd.py",
           (FileSpec("--namd", "NAMD configuration file", (".namd", ".conf", ".config", ".inp"),
                     detail="structure, coordinates, parameters, box and run settings are read from it"),),
           run_input="--namd", ff_default="namd", box=True, dihedral_terms=True, e14fac=True,
           note="the box and 1-4scaling come from the config; the box and 1-4 scale asked below override them"),
    Engine("gromacs", "GROMACS", "convert_gromacs.py",
           (FileSpec("--top", "topology (.top)", (".top",)),
            FileSpec("--gro", "coordinates (.gro)", (".gro",)),
            FileSpec("--mdp", "run file (.mdp)", (".mdp",), required=False,
                     detail="without it only the package is written, no input_dsmd.yaml")),
           run_input="--mdp", include_dirs=True,
           note="#ifdef/#define must be resolved by hand first; orthorhombic box only"),
    Engine("charmm", "CHARMM files (PSF + PRM + PDB)", "convert_charmm.py",
           (FileSpec("--psf", "topology (.psf)", (".psf",)),
            FileSpec("--param", "parameter file", (".prm", ".par", ".str", ".xplor"), multiple=True,
                     detail="merged in the order picked"),
            FileSpec("--pdb", "coordinates (.pdb)", (".pdb",)),
            FileSpec("--inp", "CHARMM stream file (.inp)", (".inp",), required=False,
                     detail="gives the run settings (input_dsmd.yaml) and can give the box")),
           run_input="--inp", ff_default="charmm36", box=True, box_optional_with="--inp",
           dihedral_terms=True, e14fac=True),
    Engine("lammps", "LAMMPS (units real)", "convert_lammps.py",
           (FileSpec("--input", "input script", (".in", ".lmp", ".inp", ".lammps"),
                     detail="units / styles / read_data; it is also the run input"),
            FileSpec("--data", "data file", (".data", ".dat", ".lmp"), required=False,
                     detail="only if the script's read_data path cannot be found from here")),
           run_input="--input"),
    Engine("spica", "SPICA coarse-grained (LAMMPS lj/spica or mie/cut)", "convert_spica.py",
           (FileSpec("--input", "input script", (".in", ".lmp", ".inp", ".lammps"),
                     detail="pair_style lj/spica or mie/cut; it is also the run input"),
            FileSpec("--data", "data file", (".data", ".dat", ".lmp"), required=False,
                     detail="only if the script's read_data path cannot be found from here")),
           run_input="--input", ff_default="spica-2019",
           note="only Mie (9,6) and (12,6), one pair for the whole system; check sigma/epsilon in parameters.dsMDprm"),
)}
