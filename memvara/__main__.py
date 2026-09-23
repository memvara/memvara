"""Entry point for `python3 -m memvara`, which runs the `memvara` console script.

Kept to one line for the reason `memvara/server/__main__.py` is: the work is in
`cli.main`, which takes its streams and environment as arguments and is therefore
testable without a subprocess.

This spelling matters beyond tidiness. The npm package `memvara` installs a `bin` of the
same name, so on a machine with both, whichever comes first on `PATH` wins. `python3 -m
memvara` always reaches this one, and `npx memvara` always reaches that one.
"""

from .cli import main

if __name__ == "__main__":  # pragma: no cover - exercised by running the module
    raise SystemExit(main())
