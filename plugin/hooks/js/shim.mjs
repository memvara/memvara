/**
 * Spawn one Python hook and read its reply. Host-neutral: nothing here knows OpenCode.
 *
 * The two JavaScript hosts cannot run a shell hook, so this is how the same four bodies
 * reach them. It frames a call the way `run.py` already expects -- a JSON payload on
 * stdin, `run.py <hook> --host <id>` on argv -- and parses the flat object
 * `core/envelope._render_flat` prints. The host module beside this file decides what to
 * do with the result; this file only guarantees that a hook cannot hang, cannot throw at
 * its caller, and cannot fail a turn.
 *
 * Every failure resolves rather than rejecting. That is the same rule the Python entry
 * point follows and for the same reason: a hook that fails a prompt is worse than a hook
 * that does nothing. The difference is that here the caller is the host's own turn loop,
 * so a rejected promise would surface as a broken turn rather than a logged error.
 *
 * **`null` and `{}` mean different things and callers depend on it.** `{}` is "the hook
 * ran and had nothing to say"; `null` is "the hook did not run" -- it timed out, failed
 * to spawn, exited non-zero, or printed bytes that were not JSON. Collapsing the two
 * would make a transient failure indistinguishable from an empty store, and the caller
 * that records a session as already-started cannot then tell whether to try again.
 */

import { spawn } from "node:child_process"
import { randomBytes } from "node:crypto"
import fs from "node:fs"
import os from "node:os"
import path from "node:path"

const LOG_DIR = path.join(os.homedir(), ".memvara", ".hooks")

/**
 * Take every permission for group and others off `target`, as the Python hooks do
 * (`lib/private.py`). A file or directory an earlier version created with the default
 * modes, readable by every account on the machine, is repaired this way.
 */
function restrict(target) {
  const mode = fs.statSync(target).mode & 0o777
  if (mode & 0o077) fs.chmodSync(target, mode & 0o700)
}

/**
 * Create `dir` 0700, and take group and other permissions off it and every directory
 * above it up to `~/.memvara`, as `lib/private.py` does. Throws as `fs.mkdirSync` would.
 */
export function privateDir(dir) {
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 })
  const root = path.join(os.homedir(), ".memvara")
  for (let level = dir; level === root || level.startsWith(root + path.sep);
       level = path.dirname(level)) {
    restrict(level)
    if (level === root) break
  }
}

/**
 * Create `file` holding `text`, or replace it, readable by this account only.
 *
 * The text is written to a new file beside `file`, created 0600 and exclusively, so no
 * other process has it open, and that file is then renamed over `file`. A process that
 * already has `file` open keeps reading the whole of what it opened. Writing `file` in
 * place truncated it under such a reader: a capture still reading one transcript read a
 * newer one that was written over it. The renamed file keeps its 0600 mode.
 */
export function writePrivate(file, text) {
  const temporary = path.join(path.dirname(file),
    `.${path.basename(file)}.${process.pid}.${randomBytes(6).toString("hex")}.tmp`)
  fs.writeFileSync(temporary, text, { mode: 0o600, flag: "wx" })
  try {
    fs.renameSync(temporary, file)
  } catch (err) {
    try { fs.unlinkSync(temporary) } catch { /* already gone */ }
    throw err
  }
}

/**
 * One line in the hook log, and never a throw.
 *
 * Silence is not an option for a host with no operator-visible channel: `status_key` is
 * empty in the OpenCode record precisely because nothing this plugin says reaches the
 * screen, which makes this file the only account of itself it has. Wrapped because a
 * home directory that cannot be written to must not become a broken turn.
 *
 * The directories are 0700 and the file 0600, as every file the Python hooks write is.
 */
export function note(name, text) {
  try {
    privateDir(LOG_DIR)
    const file = path.join(LOG_DIR, `${name}.log`)
    fs.appendFileSync(file, `${new Date().toISOString()} ${text}\n`, { mode: 0o600 })
    restrict(file)
  } catch {
    /* a hook must never fail a turn */
  }
}

/**
 * Run one hook to completion and return its parsed reply, or `{}`.
 *
 * `timeoutMs` is enforced here because neither JavaScript host publishes a hook timeout
 * of its own. Without it an extraction that wedged would hold the turn open forever,
 * which on a host that awaits its hooks is indistinguishable from the client hanging.
 * The child is killed rather than abandoned so a wedged interpreter does not accumulate
 * one process per turn.
 */
export async function runHook({ hooksDir, hook, host, payload, timeoutMs = 10000 }) {
  const script = path.join(hooksDir, "run.py")
  if (!fs.existsSync(script)) {
    note("hooks", `skipped=no run.py at ${script} hook=${hook}`)
    return null
  }
  return await new Promise((resolve) => {
    let child
    try {
      child = spawn("python3", [script, hook, "--host", host], {
        stdio: ["pipe", "pipe", "pipe"],
      })
    } catch (err) {
      note("hooks", `failed hook=${hook} host=${host} spawn: ${String(err)}`)
      resolve(null)
      return
    }

    let out = ""
    let settled = false
    const finish = (value, why) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      if (why) note("hooks", why)
      resolve(value)
    }
    const timer = setTimeout(() => {
      try { child.kill("SIGKILL") } catch { /* already gone */ }
      finish(null, `timeout hook=${hook} host=${host} after=${timeoutMs}ms`)
    }, timeoutMs)

    // `setEncoding`, not `out += buffer`. Concatenating Buffers decodes each chunk on its
    // own, so a multi-byte character split across two chunks becomes U+FFFD -- silently,
    // because the surrounding JSON still parses.
    //
    // Belt-and-braces rather than a live fix, and worth saying so plainly: `run.py`
    // answers through `json.dumps`, which escapes non-ASCII by default, so an em dash
    // leaves as the seven ASCII bytes `\u2014` and there is nothing for a boundary to
    // split. Measured -- 16,371 bytes of real recall output, zero bytes above 127. A
    // review of this file asserted the corruption was live and was wrong: the synthetic
    // proof used JavaScript's `JSON.stringify`, which does NOT escape non-ASCII, so it
    // demonstrated a property of a payload this path never produces.
    //
    // Kept because it is the correct way to read text off a stream and costs nothing, and
    // because the ASCII property is an accident of a default rather than a promise. The
    // plugin repository pins it: a guard there fails if the renderer ever passes
    // `ensure_ascii=False`, which is the moment this line stops being defensive.
    child.stdout.setEncoding("utf8")
    child.stdout.on("data", (chunk) => { out += chunk })
    child.stderr.on("data", () => { /* the body logs its own reasons */ })
    child.on("error", (err) =>
      finish(null, `failed hook=${hook} host=${host} ${String(err)}`))
    child.on("close", (code) => {
      const text = out.trim()
      // A non-zero exit is not supposed to happen -- `run.py` returns 0 on every path it
      // knows about -- so when it does, it is the paths it does NOT know about: a broken
      // import in a half-copied tree, an interpreter that died after spawning. stderr is
      // discarded here on purpose (it can carry recalled text), which used to leave that
      // case with no evidence anywhere at all. On this host the log IS the account: the
      // record sets `status_key=""` because OpenCode gives a plugin no channel to the
      // screen, so a silent failure here is a plugin that recalls nothing and says so
      // nowhere.
      // `!settled` matters: the timeout path SIGKILLs the child, so `close` then fires
      // with a null code and this would log a second line calling our own kill a failure.
      // One event, one line, and the line that is already there says more.
      if (!settled && code !== 0) {
        note("hooks",
             `failed hook=${hook} host=${host} ` +
             `${code === null ? "killed by signal" : `exit=${code}`} bytes=${text.length}`)
      }
      if (!text) { finish(code === 0 ? {} : null); return }
      try {
        const parsed = JSON.parse(text)
        finish(parsed && typeof parsed === "object" ? parsed : {})
      } catch {
        // Bytes that are not JSON mean the body printed something unexpected. Reporting
        // the length rather than the text keeps a recalled memory out of the log file.
        finish(null, `unparsed hook=${hook} host=${host} bytes=${text.length}`)
      }
    })

    try {
      child.stdin.write(JSON.stringify(payload ?? {}))
      child.stdin.end()
    } catch (err) {
      finish(null, `failed hook=${hook} host=${host} stdin: ${String(err)}`)
    }
  })
}

/**
 * Start a hook and deliberately do not wait for it.
 *
 * Capture takes 12-14 seconds. Measured on opencode 1.18.20: an awaited handler holds the
 * turn open for exactly as long as it runs (8.016s for an 8s sleep), while an un-awaited
 * one returns in 1ms and its work still completes 8.1s later, because the plugin lives in
 * the host's persistent server process. So this is not fire-and-forget as a shortcut --
 * it is the only shape in which capture can exist here at all.
 */
export function runHookDetached(opts) {
  // Returned, though never awaited by a turn, so a caller can act when the hook is done.
  return runHook(opts).catch((err) =>
    note("hooks", `failed hook=${opts.hook} detached ${String(err)}`))
}
