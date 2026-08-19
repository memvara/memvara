#!/usr/bin/env python3
"""Pack → hash → copy → publish the tarball, never the directory.

    python3 release/rehearse_npm.py

This is the integration rehearsal the Actions job cannot have a Test-npm equivalent
for. It talks to a local Verdaccio if Node can start one, and never to
registry.npmjs.org. Missing Node is a skip (exit 0), not a failure: the Python
matrix has no Node, and a rehearsal you have to disable to land a commit is a
rehearsal nobody runs.

The thing it is proving is the same boundary as publish-npm: the bytes that get
`npm publish`ed are a tarball whose SHA-256 was taken before it moved, not a
directory packed at the last second.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "npm" / "memvara"


def say(msg: str) -> None:
    print(f"  {msg}")


def have_node() -> bool:
    if shutil.which("node") is None or shutil.which("npm") is None:
        return False
    try:
        subprocess.run(
            ["node", "-e", "process.exit(0)"],
            check=True, capture_output=True, timeout=8,
        )
        return True
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def pack() -> Path:
    proc = subprocess.run(
        ["npm", "pack", "--json"],
        cwd=PKG,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    name = payload[0]["filename"] if isinstance(payload, list) else payload["filename"]
    tarball = PKG / name
    if not tarball.is_file():
        raise SystemExit(f"npm pack claimed {name} but it is not on disk")
    return tarball


def wait_port(port: int, timeout: float = 20.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.4):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def try_verdaccio(tarball: Path, version: str) -> bool:
    """Publish the tarball to a throwaway registry. False means 'could not start'."""
    port = 4873
    config = {
        "storage": "./storage",
        "listen": f"127.0.0.1:{port}",
        "auth": {"htpasswd": {"file": "./htpasswd", "max_users": -1}},
        "uplinks": {},
        "packages": {
            "@*/*": {"access": "$all", "publish": "$all", "unpublish": "$all"},
            "**": {"access": "$all", "publish": "$all", "unpublish": "$all"},
        },
        "logs": {"type": "stdout", "format": "pretty", "level": "error"},
    }
    with tempfile.TemporaryDirectory(prefix="memvara-verdaccio-") as tmp:
        tmp_path = Path(tmp)
        cfg = tmp_path / "config.yaml"
        # Minimal YAML by hand — we already have the dict, and PyYAML is not a
        # core dependency. Verdaccio accepts JSON for this file too.
        (tmp_path / "config.json").write_text(json.dumps(config))
        cfg.write_text(
            "storage: ./storage\n"
            f"listen: 127.0.0.1:{port}\n"
            "auth:\n"
            "  htpasswd:\n"
            "    file: ./htpasswd\n"
            "    max_users: -1\n"
            "uplinks: {}\n"
            "packages:\n"
            "  '@*/*':\n"
            "    access: $all\n"
            "    publish: $all\n"
            "    unpublish: $all\n"
            "  '**':\n"
            "    access: $all\n"
            "    publish: $all\n"
            "    unpublish: $all\n"
            "logs: {type: stdout, format: pretty, level: error}\n"
        )
        env = os.environ.copy()
        env["VERDACCIO_PUBLIC_URL"] = f"http://127.0.0.1:{port}/"
        try:
            proc = subprocess.Popen(
                ["npx", "--yes", "verdaccio", "--config", str(cfg)],
                cwd=tmp_path,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            say(f"verdaccio did not start ({exc}); pack/hash/dry-run already passed")
            return False
        try:
            if not wait_port(port):
                say("verdaccio never opened the port; pack/hash/dry-run already passed")
                return False
            registry = f"http://127.0.0.1:{port}/"
            # Any token is enough when max_users is -1; the CLI still wants a line.
            npmrc = tmp_path / "npmrc"
            npmrc.write_text(f"//{registry.split('://', 1)[1]}:_authToken=rehearsal\n")
            subprocess.run(
                [
                    "npm", "publish", str(tarball),
                    "--access", "public",
                    "--registry", registry,
                    "--userconfig", str(npmrc),
                ],
                check=True,
                cwd=tmp_path,
            )
            viewed = subprocess.check_output(
                ["npm", "view", f"memvara@{version}", "version", "--registry", registry],
                text=True,
                cwd=tmp_path,
            ).strip()
            if viewed != version:
                raise SystemExit(f"verdaccio has {viewed!r}, expected {version!r}")
            say(f"verdaccio has memvara@{viewed}")
            return True
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def main() -> int:
    if not have_node():
        say("Node/npm not on PATH — rehearsal skipped, not failed")
        return 0

    meta = json.loads((PKG / "package.json").read_text())
    version = meta["version"]
    say(f"packing npm/memvara {version}")

    tarball = pack()
    try:
        digest = sha256(tarball)
        say(f"sha256 {digest}")
        with tempfile.TemporaryDirectory(prefix="memvara-npm-bytes-") as tmp:
            copied = Path(tmp) / tarball.name
            shutil.copy2(tarball, copied)
            again = sha256(copied)
            if again != digest:
                raise SystemExit(f"copy changed the bytes: {digest} -> {again}")
            say("copy matches")
            subprocess.run(
                ["npm", "publish", str(copied), "--dry-run", "--access", "public"],
                check=True,
                cwd=tmp,
            )
            say("npm publish <tarball> --dry-run ok")
        try_verdaccio(tarball, version)
    finally:
        tarball.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
