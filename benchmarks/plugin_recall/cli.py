"""`python -m benchmarks.plugin_recall --plugin <name-or-path>`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .cases import DEFAULT_CASES, CaseError, load
from .plugin import PluginError, discover
from .report import as_json, render
from .runner import run

EPILOG = """\
examples:
  python -m benchmarks.plugin_recall --plugin memvara
  python -m benchmarks.plugin_recall --plugin supermemory --verbose
  python -m benchmarks.plugin_recall --plugin ~/.claude/plugins/cache/x/y/1.0.0 --json

A plugin's prompt hook runs as a real subprocess against whatever store it is configured
with, so a run reflects that machine at that moment and is not reproducible across
machines. The synthetic corpus is store-independent by construction and is the part that
compares across plugins; hit cases are not, and say so.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.plugin_recall",
        description="Measure what a memory plugin injects into a model's context.",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plugin", required=True,
                        help="Installed plugin name, or a path to a plugin root.")
    parser.add_argument("--cases", type=Path, action="append", default=None,
                        help="Corpus file; repeatable. Defaults to the shipped synthetic "
                             "silence corpus.")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(),
                        help="Directory reported to the hook as the session's cwd. Some "
                             "plugins scope recall by project, so this changes results.")
    parser.add_argument("--timeout", type=float, default=None,
                        help="Override the hook's declared timeout, in seconds.")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                        help="Extra environment for the hook; repeatable. Use it to point "
                             "a plugin at a benchmark store rather than a real one, e.g. "
                             "MEMVARA_DB=/tmp/bench.db or "
                             "SUPERMEMORY_API_URL=http://localhost:6767.")
    parser.add_argument("--shared-session", action="store_true",
                        help="Run the whole corpus in one session. Measures marginal cost "
                             "in a session already underway; its hit rate is NOT "
                             "comparable with the default, because a plugin that declines "
                             "to re-inject a memory already in context scores as a miss.")
    parser.add_argument("--verbose", action="store_true", help="List every case.")
    parser.add_argument("--json", action="store_true", help="Emit the result as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = args.cases or [DEFAULT_CASES]
    try:
        cases = load(*paths)
        plugin = discover(args.plugin, timeout=args.timeout)
    except (CaseError, PluginError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if any(path != DEFAULT_CASES for path in paths):
        # Said once, on stderr, so it cannot be mistaken for part of the result. A hook is
        # a subprocess that may call whatever service its vendor chose, and a private
        # corpus is by definition the operator's real prompts -- running one against a
        # third-party plugin transmits those prompts to that third party. Nobody should
        # discover that from a network log.
        print(f"note: running non-default cases against {plugin.label()}. Its hook is a "
              "subprocess and may send these prompts to its vendor's service.",
              file=sys.stderr)

    extra_env = {}
    for pair in args.env:
        if "=" not in pair:
            print(f"error: --env {pair!r} is not KEY=VALUE", file=sys.stderr)
            return 2
        key, _, value = pair.partition("=")
        extra_env[key] = value

    result = run(plugin, cases, cwd=args.cwd.resolve(), extra_env=extra_env,
                 shared_session=args.shared_session)
    print(as_json(result) if args.json else render(result, verbose=args.verbose))
    return 0
