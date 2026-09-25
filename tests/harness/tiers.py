"""Which tier a test belongs to, decided by the folder its file lives in."""

from __future__ import annotations

import pathlib

#: tests/, the folder this package sits in.
TESTS = pathlib.Path(__file__).resolve().parents[1]

#: Every value --tier accepts. fast is the default, and the only tier CI runs.
TIERS = ("fast", "nightly", "weekly", "local", "quarantine")

#: A folder with one of these names puts every test under it in that tier: every tier
#: except fast, which is where a test lives when no such folder is above it.
TIER_DIRS = tuple(tier for tier in TIERS if tier != "fast")

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

    Folders and files get their tier from the same rule, tier_of, so the two can never
    disagree. A folder is left out when its tier is one the option does not select. A
    fast folder is always entered, because a nightly or local folder can sit inside it.
    A file is left out when its tier is not selected, except __init__.py and
    conftest.py, which every folder needs.
    """
    wanted = SELECTS[option]
    if is_dir is None:
        is_dir = pathlib.Path(path).is_dir()
    tier = tier_of(path)
    if is_dir:
        return tier != "fast" and tier not in wanted
    if pathlib.Path(path).name in _ALWAYS:
        return False
    return tier not in wanted


#: The Hypothesis profile each tier runs under.
HYPOTHESIS_PROFILE_FOR: dict[str, str] = {
    "fast": "memvara-fast",
    "nightly": "memvara-nightly",
    "weekly": "memvara-weekly",
    "local": "memvara-nightly",
    "quarantine": "memvara-nightly",
}


def load_hypothesis_profile(option: str) -> None:
    """Register the suite's Hypothesis profiles, and load the one for --tier `option`.

    The fast profile is derandomized and keeps no example database, so a PR run is
    repeatable. The nightly and weekly profiles run far more examples. They keep what
    they find in ~/.cache/memvara-adversarial/hypothesis, so a failure found one night is
    tried first the next night. Hypothesis is imported here rather than at the top of
    the module, so this module stays importable in an environment without the dev extra.
    """
    try:
        from hypothesis import HealthCheck, settings  # noqa: PLC0415
        from hypothesis.database import DirectoryBasedExampleDatabase  # noqa: PLC0415
    except ImportError:
        return
    quiet = [HealthCheck.too_slow]
    settings.register_profile(
        "memvara-fast", max_examples=30, stateful_step_count=25, derandomize=True,
        database=None, deadline=None, print_blob=True, suppress_health_check=quiet)
    database = DirectoryBasedExampleDatabase(
        str(pathlib.Path.home() / ".cache" / "memvara-adversarial" / "hypothesis"))
    settings.register_profile(
        "memvara-nightly", max_examples=3000, stateful_step_count=100, database=database,
        deadline=None, print_blob=True, suppress_health_check=quiet)
    settings.register_profile(
        "memvara-weekly", parent=settings.get_profile("memvara-nightly"),
        max_examples=20000, stateful_step_count=200)
    settings.load_profile(HYPOTHESIS_PROFILE_FOR[option])
