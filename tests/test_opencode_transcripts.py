"""The OpenCode plugin's session transcripts stay private and do not outlive their capture.

OpenCode hands a plugin no transcript, so `plugin/hooks/js/opencode.mjs` writes one, a
whole conversation, for the capture hook to read. It wrote it to
`$TMPDIR/memvara-opencode/<session>.jsonl` with the default modes, and on Linux that is
usually the shared `/tmp`: 0755 and 0644 under the usual umask, readable by every account
on the machine, and kept for a day after the capture that needed it.

Each test loads the plugin in Node with a fake OpenCode client and a stub `python3` first
on PATH, so the capture the plugin starts runs nothing but records what it was handed.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import stat
import subprocess

import pytest

from harness.env import REPO, child_env

pytestmark = [
    pytest.mark.skipif(os.name != "posix", reason="file modes are POSIX"),
    pytest.mark.skipif(shutil.which("node") is None, reason="needs Node for the plugin"),
]

PLUGIN = REPO / "plugin" / "hooks" / "js" / "opencode.mjs"

#: One session going idle twice over the same reply, then once more after a new message,
#: waiting after each for the capture it started to finish.
DRIVER = r"""
import fs from "node:fs"
import path from "node:path"

const plugin = await import(process.argv[2])
const said = [
  { info: { role: "user" }, parts: [{ type: "text", text: "my bank PIN hint is the cat" }] },
  { info: { role: "assistant" }, parts: [{ type: "text", text: "Noted." }] },
]
const client = { session: { messages: async () => ({ data: said }) } }
const hooks = await plugin.MemvaraPlugin({ client, directory: process.cwd(),
                                           worktree: process.cwd() })
const idle = { event: { type: "session.idle", properties: { sessionID: "ses_probe" } } }
const log = path.join(process.env.HOME, "captures.jsonl")
const captures = () => fs.existsSync(log) ? fs.readFileSync(log, "utf8").trim().split("\n") : []
const settle = async (n) => {
  for (let i = 0; i < 100 && captures().length < n; i++) await new Promise((r) => setTimeout(r, 50))
  await new Promise((r) => setTimeout(r, 200))
}
await hooks.event(idle)
await settle(1)
await hooks.event(idle)
await settle(1)
said.push({ info: { role: "user" }, parts: [{ type: "text", text: "and I moved to Porto" }] })
await hooks.event(idle)
await settle(2)
console.log(JSON.stringify({ captures: captures().length }))
// The plugin's timeout for `session.messages` is a timer nothing clears, harmless in
// OpenCode's long-lived server and fifteen idle seconds here.
process.exit(0)
"""

#: The stub `python3` the shim spawns for capture: it records the payload it was handed,
#: and the transcript's mode and text as they were while it ran.
STUB = """#!/bin/sh
payload=$(cat)
path=$(printf '%s' "$payload" | sed -n 's/.*"transcript_path": *"\\([^"]*\\)".*/\\1/p')
# GNU stat first: on Linux, `stat -f` reports the file system instead, over several
# lines, and does not fail, so the BSD form must be the fallback.
mode=$(stat -c %a "$path" 2>/dev/null || stat -f %Lp "$path")
printf '{"path": "%s", "mode": "%s"}\\n' "$path" "$mode" >> "$HOME/captures.jsonl"
echo '{}'
"""


def run_plugin(tmp_path: pathlib.Path, *, umask: int = 0o022, driver: str = DRIVER,
               stub_script: str = STUB) -> tuple[pathlib.Path, dict]:
    home, shared_tmp, bin_dir = tmp_path / "home", tmp_path / "tmp", tmp_path / "bin"
    for directory in (home, shared_tmp, bin_dir):
        directory.mkdir(exist_ok=True)
    stub = bin_dir / "python3"
    stub.write_text(stub_script, encoding="utf-8")
    stub.chmod(0o755)
    (tmp_path / "driver.mjs").write_text(driver, encoding="utf-8")
    env = child_env(home, {"TMPDIR": str(shared_tmp), "PATH": f"{bin_dir}:/usr/bin:/bin"})
    old = os.umask(umask)
    try:
        done = subprocess.run([shutil.which("node"), str(tmp_path / "driver.mjs"),
                               str(PLUGIN)], env=env, cwd=str(tmp_path),
                              capture_output=True, text=True, timeout=60)
    finally:
        os.umask(old)
    assert done.returncode == 0, done.stderr
    return home, json.loads(done.stdout.strip().splitlines()[-1])


def records(home: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in
            (home / "captures.jsonl").read_text(encoding="utf-8").splitlines()]


def test_a_transcript_is_written_privately_under_the_users_own_directory(tmp_path):
    home, _ = run_plugin(tmp_path)
    (first, *_) = records(home)
    written = pathlib.Path(first["path"])
    assert written.parent == home / ".memvara" / ".hooks" / "opencode"
    assert first["mode"] == "600", "the transcript was readable by other accounts"
    for directory in (written.parent, written.parent.parent, home / ".memvara"):
        assert oct(stat.S_IMODE(directory.stat().st_mode)) == "0o700"
    assert not (tmp_path / "tmp" / "memvara-opencode").exists(), "written to the shared tmp"


def test_a_transcript_is_removed_once_its_capture_is_done(tmp_path):
    """Kept for a day before, whole, after the one capture that read it."""
    home, _ = run_plugin(tmp_path)
    for record in records(home):
        assert not pathlib.Path(record["path"]).exists(), "the transcript outlived capture"


def test_an_idle_event_over_the_same_reply_starts_no_second_capture(tmp_path):
    """With the transcript gone, the capture hook's own watermark cannot tell a repeated
    event from a new reply, so the plugin remembers what it last handed to capture."""
    home, result = run_plugin(tmp_path)
    assert result["captures"] == 2 and len(records(home)) == 2


def test_transcripts_an_earlier_version_left_in_the_shared_tmp_are_removed(tmp_path):
    old = tmp_path / "tmp" / "memvara-opencode"
    old.mkdir(parents=True)
    (old / "ses_old.jsonl").write_text('{"type": "user"}\n', encoding="utf-8")
    run_plugin(tmp_path)
    assert not (old / "ses_old.jsonl").exists(), "an old transcript was left readable"


#: One session going idle, then idle again after a new message while the capture the first
#: event started still has its transcript open and has not read it yet.
DRIVER_WHILE_READING = r"""
import fs from "node:fs"
import path from "node:path"

const plugin = await import(process.argv[2])
const said = [
  { info: { role: "user" }, parts: [{ type: "text", text: "my bank PIN hint is the cat" }] },
  { info: { role: "assistant" }, parts: [{ type: "text", text: "Noted." }] },
]
const client = { session: { messages: async () => ({ data: said }) } }
const hooks = await plugin.MemvaraPlugin({ client, directory: process.cwd(),
                                           worktree: process.cwd() })
const idle = { event: { type: "session.idle", properties: { sessionID: "ses_probe" } } }
const home = process.env.HOME
const log = path.join(home, "captures.jsonl")
const captures = () => fs.existsSync(log) ? fs.readFileSync(log, "utf8").trim().split("\n") : []
const until = async (done) => {
  for (let i = 0; i < 400 && !done(); i++) await new Promise((r) => setTimeout(r, 25))
}
await hooks.event(idle)
await until(() => fs.existsSync(path.join(home, "opened")))
said.push({ info: { role: "user" }, parts: [{ type: "text", text: "and I moved to Porto" }] })
await hooks.event(idle)
fs.writeFileSync(path.join(home, "written"), "")
await until(() => captures().length >= 2)
await new Promise((r) => setTimeout(r, 300))
console.log(JSON.stringify({ captures: captures().length }))
process.exit(0)
"""

#: A stub capture that opens its transcript, says so, waits until the driver has handed a
#: newer transcript to a second capture, and only then reads what it opened, recording how
#: many lines that was.
STUB_READING_LATE = """#!/bin/sh
payload=$(cat)
path=$(printf '%s' "$payload" | sed -n 's/.*"transcript_path": *"\\([^"]*\\)".*/\\1/p')
exec 3< "$path"
: > "$HOME/opened"
i=0
while [ ! -e "$HOME/written" ] && [ $i -lt 400 ]; do sleep 0.05; i=$((i+1)); done
lines=$(wc -l <&3 | tr -d ' ')
# GNU stat first: on Linux, `stat -f` reports the file system instead, over several
# lines, and does not fail, so the BSD form must be the fallback.
mode=$(stat -c %a "$path" 2>/dev/null || stat -f %Lp "$path")
printf '{"path": "%s", "mode": "%s", "lines": %s}\\n' "$path" "$mode" "$lines" >> "$HOME/captures.jsonl"
echo '{}'
"""


def test_a_capture_still_reading_keeps_the_transcript_it_was_handed(tmp_path):
    """The plugin wrote each new transcript over the session's file in place. A capture
    started by an earlier idle event can still be reading that file, and it then read a
    file truncated and rewritten under it: here it read the newer transcript's three lines
    instead of the two it was handed. A new transcript now goes to a private file of its
    own that is renamed over the old one, so a reader keeps the file it opened."""
    home, result = run_plugin(tmp_path, driver=DRIVER_WHILE_READING,
                              stub_script=STUB_READING_LATE)
    assert result["captures"] == 2
    read = records(home)
    assert sorted(r["lines"] for r in read) == [2, 3], "a capture read a rewritten file"
    assert [r["mode"] for r in read] == ["600", "600"]
    assert list((home / ".memvara" / ".hooks" / "opencode").iterdir()) == []


#: One session going idle, then idle again after its last message was replaced by one of
#: the same length, as an undone reply followed by a different one of the same length.
DRIVER_SAME_LENGTH = DRIVER.replace(
    """await hooks.event(idle)
await settle(1)
await hooks.event(idle)
await settle(1)
said.push({ info: { role: "user" }, parts: [{ type: "text", text: "and I moved to Porto" }] })
await hooks.event(idle)
await settle(2)""",
    """await hooks.event(idle)
await settle(1)
said[1] = { info: { role: "assistant" }, parts: [{ type: "text", text: "Got it" }] }
await hooks.event(idle)
await settle(2)""")


def test_a_new_transcript_as_long_as_the_last_one_is_captured(tmp_path):
    """The plugin remembered only the length of the transcript it last handed to capture,
    and started no capture for one of the same length. A different conversation that
    happens to be as long, here a reply replaced by another of six characters, was never
    captured. It now compares a digest of the content."""
    assert DRIVER_SAME_LENGTH != DRIVER
    home, result = run_plugin(tmp_path, driver=DRIVER_SAME_LENGTH)
    assert result["captures"] == 2 and len(records(home)) == 2
