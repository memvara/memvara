"""The adversarial suite's nightly run, as the package `nightly`.

The modules here are run as scripts, such as `python3 scripts/nightly/run.py`, and the
tests import them as `nightly.<module>` with scripts/ on the import path. A module run as
a script puts scripts/ on the path itself, so its siblings are loaded under the same names
either way, and no generic name such as `run` or `flakes` enters `sys.modules`.

Importing the package also puts tests/ on the import path, so these scripts use the
harness's `report` and `env` modules exactly as the tests do. The testing guide's section
"The nightly run" describes the whole run.
"""

from __future__ import annotations

import pathlib
import sys

#: The checkout these scripts belong to. This file is scripts/nightly/__init__.py.
REPO = pathlib.Path(__file__).resolve().parents[2]

if str(REPO / "tests") not in sys.path:
    sys.path.insert(0, str(REPO / "tests"))
