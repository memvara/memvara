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
 * Every failure resolves to `{}` rather than rejecting. That is the same rule the Python
 * entry point follows and for the same reason: a hook that fails a prompt is worse than a
 * hook that does nothing. The difference is that here the caller is the host's own turn
 * loop, so a rejected promise would surface as a broken turn rather than a logged error.
 */

import { spawn } from "node:child_process"
import fs from "node:fs"
import os from "node:os"
import path from "node:path"

const LOG_DIR = path.join(os.homedir(), ".memvara", ".hooks")

/**
 * One line in the hook log, and never a throw.
 *
 * Silence is not an option for a host with no operator-visible channel: `status_key` is
 * empty in the OpenCode record precisely because nothing this plugin says reaches the
 * screen, which makes this file the only account of itself it has. Wrapped because a
 * home directory that cannot be written to must not become a broken turn.
 */
export function note(name, text) {
  try {
    fs.mkdirSync(LOG_DIR, { recursive: true })
    fs.appendFileSync(path.join(LOG_DIR, `${name}.log`),
      `${new Date().toISOString()} ${text}\n`)
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
    return {}
  }
  return await new Promise((resolve) => {
    let child
    try {
      child = spawn("python3", [script, hook, "--host", host], {
        stdio: ["pipe", "pipe", "pipe"],
      })
    } catch (err) {
      note("hooks", `failed hook=${hook} host=${host} spawn: ${String(err)}`)
      resolve({})
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
      finish({}, `timeout hook=${hook} host=${host} after=${timeoutMs}ms`)
    }, timeoutMs)

    child.stdout.on("data", (b) => { out += b })
    child.stderr.on("data", () => { /* the body logs its own reasons */ })
    child.on("error", (err) =>
      finish({}, `failed hook=${hook} host=${host} ${String(err)}`))
    child.on("close", () => {
      const text = out.trim()
      if (!text) { finish({}); return }
      try {
        const parsed = JSON.parse(text)
        finish(parsed && typeof parsed === "object" ? parsed : {})
      } catch {
        // Bytes that are not JSON mean the body printed something unexpected. Reporting
        // the length rather than the text keeps a recalled memory out of the log file.
        finish({}, `unparsed hook=${hook} host=${host} bytes=${text.length}`)
      }
    })

    try {
      child.stdin.write(JSON.stringify(payload ?? {}))
      child.stdin.end()
    } catch (err) {
      finish({}, `failed hook=${hook} host=${host} stdin: ${String(err)}`)
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
  runHook(opts).catch((err) =>
    note("hooks", `failed hook=${opts.hook} detached ${String(err)}`))
}
