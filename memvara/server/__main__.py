"""Entry point for `python -m memvara.server`.

Kept to one line so there is nothing here to test: the work is in `cli.main`, which
takes its streams and environment as arguments and is therefore testable without a
subprocess.
"""

from .cli import main

if __name__ == "__main__":  # pragma: no cover - exercised by running the module
    raise SystemExit(main())
