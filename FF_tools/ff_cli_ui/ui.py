"""Terminal prompts: numbered menus, text with a default, yes/no. Standard library only."""
import glob
import os
import sys
from typing import Callable, List, Optional, Sequence, Tuple

PAGE = 15

try:
    import readline            # Linux/macOS; on Windows only with the pyreadline3 package
except ImportError:
    readline = None


class Quit(Exception):
    """The user ended the program (q at a menu, Ctrl-C, end of input)."""


class Back(Exception):
    """The user asked to go one step back (b)."""


def _color_on() -> bool:
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return False
    if os.name == "nt":
        os.system("")      # turns on ANSI escapes in the Windows console
    return True


_COLOR = _color_on()


def style(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOR else text


def bold(text: str) -> str:
    return style(text, "1")


def dim(text: str) -> str:
    return style(text, "2")


def green(text: str) -> str:
    return style(text, "32")


def yellow(text: str) -> str:
    return style(text, "33")


def red(text: str) -> str:
    return style(text, "31")


def clear() -> None:
    """Start a step on an empty screen, so the menu is never pushed off the top."""
    if _COLOR:
        print("[2J[H", end="")


def heading(text: str) -> None:
    print()
    print(bold(f"== {text}"))


def _path_completer(dirs_only: bool) -> Callable[[str, int], Optional[str]]:
    """Tab completion of file system paths (~ allowed); a folder gets a trailing /."""
    def complete(text: str, state: int) -> Optional[str]:
        matches = []
        for p in sorted(glob.glob(os.path.expanduser(text) + "*")):
            if os.path.isdir(p):
                matches.append(p + "/")
            elif not dirs_only:
                matches.append(p)
        if text.startswith("~"):          # glob saw the expanded form: show what the user typed
            home = os.path.expanduser("~")
            matches = ["~" + m[len(home):] if m.startswith(home) else m for m in matches]
        return matches[state] if state < len(matches) else None
    return complete


def _read(prompt: str, complete: Optional[str] = None) -> str:
    """complete = "file" or "dir" turns on Tab completion of paths for this one answer."""
    if complete and readline:
        readline.set_completer(_path_completer(complete == "dir"))
        readline.set_completer_delims(" \t\n")
        readline.parse_and_bind("bind ^I rl_complete" if "libedit" in (readline.__doc__ or "") else "tab: complete")
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise Quit()
    finally:
        if complete and readline:
            readline.set_completer(None)


def choose(title: str, options: Sequence[Tuple[str, str]], *, default: Optional[int] = None, back: bool = True,
           extras: Sequence[Tuple[str, str]] = ()) -> object:
    """Numbered menu. options = (label, detail). extras = (key letter, label) lines that return the letter.

    Returns the 0-based index of an option, or the letter of an extra. b = back (raises Back), q = quit (raises Quit).
    More than PAGE options are shown a page at a time (n = next page, p = previous)."""
    page = 0
    pages = max(1, (len(options) + PAGE - 1) // PAGE)
    letters = {k.lower(): k for k, _ in extras}
    while True:
        print()
        print(title)
        lo, hi = page * PAGE, min(len(options), (page + 1) * PAGE)
        for i in range(lo, hi):
            label, detail = options[i]
            mark = " (default)" if default == i else ""
            print(f"  {bold(str(i + 1).rjust(3))}) {label}{mark}" + (f"   {dim(detail)}" if detail else ""))
        for key, label in extras:
            print(f"  {bold(key.rjust(3))}) {label}")
        hint = []
        if pages > 1:
            hint.append(f"page {page + 1}/{pages}: n = next, p = previous")
        if back:
            hint.append("b = back")
        hint.append("q = quit")
        print(dim("      " + ", ".join(hint)))
        answer = _read("> ").lower()
        if not answer and default is not None:
            return default
        if answer == "q":
            raise Quit()
        if answer == "b" and back:
            raise Back()
        if answer == "n" and page + 1 < pages:
            page += 1
            continue
        if answer == "p" and page > 0:
            page -= 1
            continue
        if answer in letters:
            return letters[answer]
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer) - 1
        print(red("  not a choice, type one of the numbers or letters above"))


def ask(prompt: str, default: Optional[str] = None, *, allow_empty: bool = False,
        validate: Optional[Callable[[str], Optional[str]]] = None, complete: Optional[str] = None) -> str:
    """Text input. validate(text) returns an error message, or None when the text is fine. b = back.
    complete = "file" or "dir": Tab completes paths."""
    while True:
        shown = f"{prompt} [{default}]" if default else prompt
        answer = _read(f"{shown}: ", complete)
        if answer.lower() == "b" and not default:
            raise Back()
        if not answer:
            if default is not None:
                answer = default
            elif not allow_empty:
                print(red("  an answer is needed (b = back)"))
                continue
        if validate:
            error = validate(answer)
            if error:
                print(red(f"  {error}"))
                continue
        return answer


def confirm(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = _read(f"{prompt} {suffix}: ").lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print(red("  type y or n"))
