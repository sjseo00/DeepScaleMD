"""Finding and downloading model files: Hugging Face, GitHub, the installed package, the local disk. Standard library only."""
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .catalog import MLIP, Source

TIMEOUT = 25
HF = "https://huggingface.co"
GH_API = "https://api.github.com"
_SKIP_NAMES = {".gitattributes", "README.md", "LICENSE", "USE_POLICY.md", "MODEL-LICENSE.md", "config.json"}


class SourceError(RuntimeError):
    pass


@dataclass
class Candidate:
    """One thing the user can pick: a file (or a model folder) and where it comes from."""
    name: str                          # shown in the list
    origin: str                        # "Hugging Face InstaDeepAI/..." etc.
    size: Optional[int] = None
    files: List[Tuple[str, str]] = field(default_factory=list)   # (url, relative path under the destination folder)
    local_path: Optional[str] = None   # already on disk (package file, local file, ...): nothing to download
    is_folder: bool = False            # the model is the folder holding `files` (matgl)
    gated: bool = False
    bundled: bool = False              # no file at all: the converter uses what is inside the package (CHGNet)

    @property
    def needs_download(self) -> bool:
        return self.local_path is None and not self.bundled


def human_size(n: Optional[int]) -> str:
    if n is None:
        return ""
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return ""


# ---------------------------------------------------------------- HTTP

class _KeepAuthOnlyOnSameHost(urllib.request.HTTPRedirectHandler):
    """Hugging Face redirects downloads to a CDN with a signed address: the token must not follow it there."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(newurl).netloc != urllib.parse.urlparse(req.full_url).netloc:
            new.headers.pop("Authorization", None)
            new.unredirected_hdrs.pop("Authorization", None)
        return new


_opener = urllib.request.build_opener(_KeepAuthOnlyOnSameHost)


def _request(url: str, token: Optional[str] = None) -> urllib.request.Request:
    headers = {"User-Agent": "mlip_cli_ui"}
    host = urllib.parse.urlparse(url).netloc
    if token and host in ("huggingface.co", "api.github.com"):
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def _explain(error: Exception, url: str) -> SourceError:
    if isinstance(error, urllib.error.HTTPError):
        if error.code in (401, 403):
            if "huggingface.co" in url:
                return SourceError("access denied: this repo is gated or private. Accept its license on huggingface.co, then "
                                   "set the token (HF_TOKEN=hf_...) and run again")
            if error.code == 403 and "api.github.com" in url:
                return SourceError("GitHub refused the request (rate limit of the free API). Set GITHUB_TOKEN or wait a while")
            return SourceError(f"access denied (HTTP {error.code})")
        if error.code == 404:
            return SourceError("not found (HTTP 404)")
        return SourceError(f"HTTP {error.code}")
    return SourceError(f"cannot reach the server ({getattr(error, 'reason', error)}). No internet? Use a local file instead")


def get_json(url: str, token: Optional[str] = None):
    try:
        with _opener.open(_request(url, token), timeout=TIMEOUT) as response:
            return json.load(response)
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise _explain(error, url)


# ---------------------------------------------------------------- listing

def _wanted(name: str, mlip: MLIP) -> bool:
    base = name.rsplit("/", 1)[-1]
    if base in _SKIP_NAMES:
        return False
    if mlip.extensions and not any(base.endswith(ext) for ext in mlip.extensions):
        return False
    if mlip.include and not re.search(mlip.include, base):
        return False
    if mlip.exclude and re.search(mlip.exclude, base):
        return False
    return True


def list_hf(repo: str, mlip: MLIP, gated: bool = False, folder: bool = False, token: Optional[str] = None) -> List[Candidate]:
    tree = get_json(f"{HF}/api/models/{repo}/tree/main?recursive=true", token)
    files = [t for t in tree if t.get("type") == "file"]
    origin = f"Hugging Face {repo}"
    quoted = urllib.parse.quote(repo, safe="/")

    def url_of(path: str) -> str:
        return f"{HF}/{quoted}/resolve/main/{urllib.parse.quote(path)}"

    if folder:
        keep = [f for f in files if f["path"] not in _SKIP_NAMES and not f["path"].startswith(".")]
        if not keep:
            return []
        return [Candidate(repo.split("/")[-1], origin, sum(f.get("size") or 0 for f in keep),
                          [(url_of(f["path"]), f["path"]) for f in keep], is_folder=True, gated=gated)]
    out = []
    for f in files:
        if _wanted(f["path"], mlip):
            out.append(Candidate(f["path"], origin, f.get("size"), [(url_of(f["path"]), f["path"].rsplit("/", 1)[-1])], gated=gated))
    return out


def list_github_release(repo: str, mlip: MLIP, token: Optional[str] = None) -> List[Candidate]:
    releases = get_json(f"{GH_API}/repos/{repo}/releases?per_page=40", token)
    out = []
    for release in releases:
        for asset in release.get("assets", []):
            if _wanted(asset["name"], mlip):
                out.append(Candidate(asset["name"], f"GitHub {repo} release {release['tag_name']}", asset.get("size"),
                                     [(asset["browser_download_url"], asset["name"])]))
    return out


def list_github_dir(repo: str, path: str, mlip: MLIP, token: Optional[str] = None, depth: int = 1) -> List[Candidate]:
    entries = get_json(f"{GH_API}/repos/{repo}/contents/{path}", token)
    out = []
    for entry in entries:
        if entry["type"] == "file" and _wanted(entry["name"], mlip) and entry.get("download_url"):
            out.append(Candidate(entry["name"], f"GitHub {repo}", entry.get("size"), [(entry["download_url"], entry["name"])]))
        elif entry["type"] == "dir" and depth > 0 and entry["name"] != "archive":
            try:
                for cand in list_github_dir(repo, f"{path}/{entry['name']}", mlip, token, depth - 1):
                    out.append(Candidate(f"{entry['name']}/{cand.name.split('/')[-1]}", cand.origin, cand.size, cand.files))
            except SourceError:
                continue
    return out


def search_hf(query: str, limit: int = 20, token: Optional[str] = None) -> List[Dict]:
    url = f"{HF}/api/models?" + urllib.parse.urlencode({"search": query, "limit": limit, "sort": "downloads", "direction": -1})
    return get_json(url, token)


def list_package_files(python: str, package: str, relpath: str, mlip: MLIP) -> List[Candidate]:
    """Model files shipped inside an installed package (SevenNet keeps its checkpoints there). The package is not imported."""
    code = ("import importlib.util, sys\n"
            f"spec = importlib.util.find_spec({package!r})\n"
            "print(list(spec.submodule_search_locations)[0] if spec and spec.submodule_search_locations else '')\n")
    try:
        result = subprocess.run([python, "-I", "-c", code], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return []
    root = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if not root:
        return []
    base = Path(root) / relpath if relpath else Path(root)
    out = []
    if base.is_dir():
        for path in sorted(base.rglob("*")):
            if path.is_file() and _wanted(path.name, mlip) and "archive" not in path.parts:
                out.append(Candidate(str(path.relative_to(base)).replace("\\", "/"), f"installed {package} package",
                                     path.stat().st_size, local_path=str(path)))
    return out


def scan_local(folder: Path, mlip: MLIP, limit: int = 30) -> List[Candidate]:
    out = []
    if folder.is_dir():
        for path in sorted(folder.rglob("*")):
            if path.is_file() and _wanted(path.name, mlip):
                out.append(Candidate(str(path), "on this disk", path.stat().st_size, local_path=str(path)))
                if len(out) >= limit:
                    break
    return out


def collect(mlip: MLIP, python: str, token: Optional[str], gh_token: Optional[str], offline: bool,
            report: Callable[[str], None]) -> List[Candidate]:
    """Every model file the catalog can find for this MLIP. A source that cannot be reached is reported and skipped."""
    found: List[Candidate] = []
    for source in mlip.sources:
        if source.kind == "package":
            if not source.path:
                found.append(Candidate("use the weights inside the package (no file)", source.label, bundled=True))
            elif not offline:
                files = list_package_files(python, mlip.package, source.path, mlip)
                if files:
                    found.extend(files)
                else:
                    report(f"  ({mlip.package} is not installed in {python}: its bundled checkpoints are not listed)")
            continue
        if offline:
            continue
        try:
            if source.kind == "hf":
                found.extend(list_hf(source.ref, mlip, source.gated, source.folder, token))
            elif source.kind == "github_release":
                found.extend(list_github_release(source.ref, mlip, gh_token))
            elif source.kind == "github_dir":
                found.extend(list_github_dir(source.ref, source.path, mlip, gh_token))
        except SourceError as error:
            report(f"  ({source.label}: {error})")
    return found


# ---------------------------------------------------------------- download

def download(url: str, dest: Path, token: Optional[str] = None, expected: Optional[int] = None) -> Path:
    """Download to dest (through dest.part, so an interrupted download never leaves a half file under the real name)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        with _opener.open(_request(url, token), timeout=TIMEOUT) as response, open(part, "wb") as out:
            total = int(response.headers.get("Content-Length") or 0) or expected or 0
            done = 0
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                _progress(dest.name, done, total)
    except (urllib.error.URLError, OSError) as error:
        sys.stderr.write("\n")
        if part.exists():
            part.unlink()
        raise _explain(error, url)
    sys.stderr.write("\n")
    os.replace(part, dest)
    return dest


def _progress(name: str, done: int, total: int) -> None:
    if total:
        width = 28
        filled = int(width * done / total)
        sys.stderr.write(f"\r  {name[:40]:<40} [{'#' * filled}{'.' * (width - filled)}] {human_size(done)} / {human_size(total)}   ")
    else:
        sys.stderr.write(f"\r  {name[:40]:<40} {human_size(done)}   ")
    sys.stderr.flush()


def fetch(candidate: Candidate, target_dir: Path, token: Optional[str]) -> str:
    """Make the candidate a path on disk and return it (the folder for a model folder)."""
    if candidate.local_path:
        return candidate.local_path
    if candidate.is_folder:
        folder = target_dir / candidate.name
        for url, rel in candidate.files:
            dest = folder / rel
            if not dest.exists():
                download(url, dest, token)
        return str(folder)
    (url, rel), = candidate.files
    dest = target_dir / rel
    if dest.exists() and (candidate.size is None or dest.stat().st_size == candidate.size):
        print(f"  already downloaded: {dest}")
    else:
        download(url, dest, token, candidate.size)
    return str(dest)
