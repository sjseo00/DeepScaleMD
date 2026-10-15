"""Terminal prompts: numbered menus, text with a default, yes/no. Standard library only."""
import os
import sys
from typing import Callable, List, Optional, Sequence, Tuple

PAGE = 15


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


def heading(text: str) -> None:
    print()
    print(bold(f"== {text}"))


def _read(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise Quit()


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
        validate: Optional[Callable[[str], Optional[str]]] = None) -> str:
    """Text input. validate(text) returns an error message, or None when the text is fine. b = back."""
    while True:
        shown = f"{prompt} [{default}]" if default else prompt
        answer = _read(f"{shown}: ")
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
