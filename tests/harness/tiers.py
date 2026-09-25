"""Which tier a test belongs to, decided by the folder its file lives in."""

from __future__ import annotations

import pathlib

#: tests/, the folder this package sits in.
TESTS = pathlib.Path(__file__).resolve().parents[1]

#: Every value --tier accepts. fast is the default, and the only tier CI runs.
TIERS = ("fast", "nightly", "weekly", "local", "quarantine")

#: A folder with one of these names puts every test under it in that tier.
TIER_DIRS = ("nightly", "weekly", "local", "quarantine")

#: The tiers each --tier value collects. The first three widen in turn. local and
#: quarantine collect only themselves: local tests need this machine's logins, Docker or
#: transcripts, and quarantined tests are known to be unreliable, so neither run says
#: anything about the fast tier.
SELECTS: dict[str, frozenset[str]] = {
    "fast": frozenset({"fast"}),
    "nightly": frozenset({"fast", "nightly"}),
    "weekly": frozenset({"fast", "nightly", "weekly"}),
    "local": frozenset({"local"}),
    "quarantine": frozenset({"quarantine"}),
}

#: Files that every folder needs whatever the tier. They are never left out.
_ALWAYS = ("__init__.py", "conftest.py")


def _parts(path: pathlib.Path) -> tuple[str, ...] | None:
    """`path` relative to tests/, as parts, or None when it is outside tests/."""
    try:
        return pathlib.Path(path).resolve().relative_to(TESTS).parts
    except ValueError:
        return None


def tier_of(path: pathlib.Path) -> str:
    """The tier of a test file or folder. Anything outside tests/, such as the doctests
    in memvara/, is fast."""
    parts = _parts(path)
    if parts is None:
        return "fast"
    if parts[:1] == ("live",):
        return "local"
    for part in parts:
        if part in TIER_DIRS:
            return part
    return "fast"


def ignored(path: pathlib.Path, option: str, *, is_dir: bool | None = None) -> bool:
    """Whether a run with --tier `option` leaves `path` out of collection.

    A folder is left out only when it is itself a tier folder, or tests/live, that the
    option does not select. Any other folder is entered, because a nightly or local
    folder can sit inside it. A file is left out when its tier is not selected, except
    __init__.py and conftest.py, which every folder needs.
    """
    wanted = SELECTS[option]
    if is_dir is None:
        is_dir = pathlib.Path(path).is_dir()
    if is_dir:
        parts = _parts(path)
        if not parts:
            return False
        if parts == ("live",):
            own: str | None = "local"
        else:
            own = parts[-1] if parts[-1] in TIER_DIRS else None
        return own is not None and own not in wanted
    if pathlib.Path(path).name in _ALWAYS:
        return False
    return tier_of(path) not in wanted
