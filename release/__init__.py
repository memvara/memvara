"""Release tooling. Not part of the installed package.

`[tool.hatch.build.targets.wheel]` names `memvara` and only `memvara`, so nothing here
reaches a user's site-packages. It *is* in the sdist, on purpose and for the same reason
`tests/` is: a source distribution that cannot reproduce how the thing was released is
missing something a reader will want.

This file exists so the modules beside it can import each other by name — `python -m
release.bump_version` from the repository root — rather than by path arithmetic. The two
older scripts, `publish_pypi.py` and `publish_npm.py`, are standalone and import nothing
from here; they are the manual path and are left exactly as they were audited.
"""
