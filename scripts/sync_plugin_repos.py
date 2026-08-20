#!/usr/bin/env python3
"""Copy memvara/skills/memvara/ into a plugin-repo checkout.

Usage:
  python3 scripts/sync_plugin_repos.py /path/to/claude-memvara

Writes plugin/skills/memvara/ and skill.lock (library HEAD). Does not commit.
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SKILL_SRC = REPO / "memvara" / "skills" / "memvara"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dest", type=pathlib.Path, help="plugin repo root")
    args = parser.parse_args(argv)
    dest = args.dest.resolve()
    if not dest.is_dir():
        print(f"not a directory: {dest}", file=sys.stderr)
        return 2
    if not (SKILL_SRC / "SKILL.md").is_file():
        print(f"missing skill at {SKILL_SRC}", file=sys.stderr)
        return 2

    # Claude/Cursor/Codex/Grok/VS Code vendor under plugin/. OpenCode and
    # OpenClaw keep the tree at skills/memvara.
    if (dest / "skills" / "memvara").is_dir() and not (
        dest / "plugin" / ".mcp.json"
    ).is_file() and not (dest / "plugin" / "mcp.json").is_file():
        target = dest / "skills" / "memvara"
    else:
        target = dest / "plugin" / "skills" / "memvara"
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SKILL_SRC, target)

    sha = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    (dest / "skill.lock").write_text(
        "# The library commit whose memvara/skills/memvara/ tree this plugin vendors.\n"
        "# CI diffs plugin/skills/memvara against that SHA. skill-sync.yml updates this\n"
        "# file when it opens a PR.\n"
        "repo=memvara/memvara\n"
        f"sha={sha}\n"
        "path=memvara/skills/memvara\n",
        encoding="utf-8",
    )
    print(f"copied skill @ {sha} -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
