"""One module per coding host. Each defines `HOST`, a `core.host.Host` record.

This package is the per-repository half of the split: `core/` is identical in every
plugin repository that vendors these hooks, and everything that differs between them
lives here -- the record itself, and the one line below that names it.
"""

from __future__ import annotations


def default():
    """The host this repository packages, for an invocation that named none.

    `run.py` is always given `--host`, so this answers the other caller: a bare
    `python3 hooks/recall.py`, which is how this plugin was invoked before there was a
    dispatcher and is how its tests still drive it. Sited here rather than in `core/`
    so that vendoring these hooks for another editor changes this file and no other.
    """
    from hosts.claude import HOST

    return HOST
