"""The test suite's `--tier` option, registered where every pytest run can see it.

pytest reads this file on every run in this repository, whatever paths the run is given.
It reads tests/conftest.py only when a run includes tests/. Both hooks below lived there
at first, and two things went wrong:
- a run given only memvara/, such as `pytest memvara --tier local`, stopped with
  "unrecognized arguments: --tier", because pytest had not read the file that registers
  the option;
- the doctests in memvara/ were collected by every tier, because pytest asks a conftest
  file's `pytest_ignore_collect` only about paths inside that file's own folder.

tests/harness/tiers.py decides which tier a test is in and what each `--tier` value
collects, and docs/claude/testing.md describes the tiers. That module is loaded here from
its file, under a name of its own, rather than imported as `harness.tiers`: this file is
read before pytest puts tests/ on the import path, and a setting that put it there would
be lost in a run given another config file (`-c`), and would let a module under tests/
hide an installed package of the same name in every run.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import ModuleType
from typing import Any

import pytest


def _load_tiers() -> ModuleType:
    path = pathlib.Path(__file__).resolve().parent / "tests" / "harness" / "tiers.py"
    spec = importlib.util.spec_from_file_location("_memvara_test_tiers", path)
    assert spec is not None and spec.loader is not None, path
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tiers = _load_tiers()


def pytest_addoption(parser: Any) -> None:
    parser.addoption(
        "--tier", choices=tiers.TIERS, default="fast",
        help="which tier of tests to collect: fast (the default, and what CI runs), "
             "nightly, weekly, local or quarantine. See docs/claude/testing.md.")


def pytest_configure(config: Any) -> None:
    tiers.load_hypothesis_profile(config.getoption("--tier"))


def pytest_ignore_collect(collection_path: pathlib.Path, config: Any) -> bool | None:
    """Leave out every test whose tier --tier does not select.

    Returns True or None, never False. pytest stops at the first hook that returns a
    value, so returning False here would overrule --ignore and every other plugin's
    decision about the same path.
    """
    if tiers.ignored(collection_path, config.getoption("--tier")):
        if collection_path.is_dir():
            config.stash.setdefault(_LEFT_OUT, set()).add(collection_path)
        return True
    return None


#: The tier folders this run left out, for the line below.
_LEFT_OUT = pytest.StashKey[set]()


def pytest_report_collectionfinish(config: Any) -> str:
    """Say which tier ran and which tier folders it left out, on every run."""
    return tiers.collection_report(config.getoption("--tier"),
                                   config.stash.get(_LEFT_OUT, set()))
