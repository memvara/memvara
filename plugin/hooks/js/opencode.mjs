/**
 * The module OpenCode loads. Maps its in-process hooks onto the four canonical ones.
 *
 * OpenCode's plugin API is unlike every shell host: handlers receive typed objects and
 * inject by mutating them. So this file is the whole of the translation, and it is kept
 * as thin as the host allows -- the memory work itself is the same Python that runs
 * everywhere else, reached through `shim.mjs`.
 *
 * Three behaviours here are measurements rather than choices, each recorded in
 * `hosts/opencode.py` with the numbers:
 *
 * 1. A part pushed into `output.parts` MUST carry `id`, `sessionID` and `messageID`.
 *    Pushing `{type, text}` alone fails schema validation server-side and kills the whole
 *    turn with an opaque `UnknownError`, whose real cause appears only in opencode's own
 *    log as `invalid user part before save`.
 * 2. `capture` is never awaited. Awaiting holds the turn open for the full extraction.
 * 3. `session_start` runs on the first message of each session, because OpenCode has no
 *    session-start hook that can inject -- every once-per-session hook it offers is void.
 */

import { fileURLToPath } from "node:url"

import { note, runHook, runHookDetached } from "./shim.mjs"

const HOST = "opencode"
//: `fileURLToPath`, not `new URL(...).pathname`. The latter yields `/C:/Users/...` on
//: Windows -- a string that looks like a path, joins like a path, and resolves to
//: nothing, so every hook would degrade to "no run.py" on that platform and only there.
const HOOKS_DIR = fileURLToPath(new URL("..", import.meta.url))

/**
 * Sessions whose `session_start` has already run.
 *
 * In memory rather than on disk, and correct only because an OpenCode plugin is loaded
 * once into a server process that outlives the turn -- the same property that makes
 * detached capture work. If that ever stops being true this degrades to running
 * `session_start` more often, never to running it never -- and the entry is released
 * again when the hook did not run, so a transient failure degrades the same way rather
 * than suppressing the standing block for the rest of the session.
 */
const started = new Set()

/** One line, once per process, recording what `permission.ask` actually hands a hook. */
let askShapeLogged = false

const TIMEOUTS = { session_start: 20000, recall: 10000, capture: 120000,
                   approve: 5000, transcript: 15000 }

export const MemvaraPlugin = async ({ client, directory, worktree }) => {
  note("hooks", `opencode plugin loaded dir=${HOOKS_DIR}`)

  /** Materialise a transcript OpenCode never hands us, in the shape `lib.transcript` reads. */
  const writeTranscript = async (sessionID) => {
    try {
      // Bounded, like every other call out of this file. `runHook` enforces a timeout
      // because the host publishes none; this call had none at all, and it sits BEFORE
      // the detached capture -- so a wedged server would hold the event handler open
      // forever and the "capture cannot stall anything" property would not cover the
      // transcript step capture depends on.
      const res = await Promise.race([
        client.session.messages({ path: { id: sessionID } }),
        new Promise((_, reject) =>
          setTimeout(() => reject(new Error("session.messages timed out")),
                     TIMEOUTS.transcript)),
      ])
      const rows = res?.data ?? res ?? []
      const lines = []
      for (const row of rows) {
        const info = row?.info ?? row
        const role = info?.role
        if (role !== "user" && role !== "assistant") continue
        const content = (row?.parts ?? [])
          .filter((p) => p?.type === "text" && p.text)
          .map((p) => ({ type: "text", text: p.text }))
        if (content.length) lines.push(JSON.stringify({ type: role, message: { content } }))
      }
      if (!lines.length) return ""
      const fs = await import("node:fs")
      const os = await import("node:os")
      const path = await import("node:path")
      const dir = path.join(os.tmpdir(), "memvara-opencode")
      fs.mkdirSync(dir, { recursive: true })
      // One file per session, rewritten each turn, so this never grows within a session.
      // Across sessions it would grow without bound -- a machine that has run OpenCode
      // for a month would hold a file per session it ever opened, each the size of a
      // whole conversation. Pruned by age on write rather than deleted after capture,
      // because capture is detached and deleting under a running child is a race.
      const cutoff = Date.now() - 24 * 60 * 60 * 1000
      for (const name of fs.readdirSync(dir)) {
        try {
          const full = path.join(dir, name)
          if (fs.statSync(full).mtimeMs < cutoff) fs.unlinkSync(full)
        } catch { /* another turn pruned it first */ }
      }
      const file = path.join(dir, `${sessionID}.jsonl`)
      fs.writeFileSync(file, lines.join("\n") + "\n")
      return file
    } catch (err) {
      note("hooks", `transcript unavailable session=${sessionID} ${String(err)}`)
      return ""
    }
  }

  return {
    "chat.message": async (input, output) => {
      const sessionID = input?.sessionID ?? ""
      const messageID = output?.message?.id ?? input?.messageID ?? ""
      const prompt = (output?.parts ?? [])
        .filter((p) => p?.type === "text" && p.text)
        .map((p) => p.text)
        .join("\n")

      // Point 1 in this file's header is an invariant about the part we push, so it has
      // to be checked before the work rather than asserted in prose and then defaulted
      // away. Injecting is the whole point, but a part built from ids we do not have is
      // the one thing measured to take the entire turn down with an opaque error, and
      // one missed recall costs a turn's memories rather than the turn.
      if (!sessionID || !messageID) {
        note("hooks", `skipped=chat.message has no ids session=${!!sessionID} ` +
                      `message=${!!messageID}`)
        return
      }

      const payload = { session_id: sessionID, cwd: directory ?? worktree ?? "", prompt }
      const blocks = []

      if (!started.has(sessionID)) {
        // Claimed BEFORE the await so two messages racing into the same new session
        // cannot both run it, and released again if it did not run -- `null` from the
        // shim means exactly that, as opposed to `{}` for "ran, nothing stored yet".
        // Without the release a single 20s timeout on the first message of a session
        // would suppress the standing block for the rest of that session, silently.
        started.add(sessionID)
        const reply = await runHook({
          hooksDir: HOOKS_DIR, hook: "session_start", host: HOST,
          payload, timeoutMs: TIMEOUTS.session_start,
        })
        if (reply === null) started.delete(sessionID)
        else if (reply.additionalContext) blocks.push(reply.additionalContext)
      }

      const reply = await runHook({
        hooksDir: HOOKS_DIR, hook: "recall", host: HOST,
        payload, timeoutMs: TIMEOUTS.recall,
      })
      if (reply?.additionalContext) blocks.push(reply.additionalContext)
      if (!blocks.length) return

      // Every required key, for the reason in this file's header.
      output.parts.push({
        id: `prt_memvara_${Date.now().toString(36)}`,
        sessionID,
        messageID,
        type: "text",
        text: blocks.join("\n\n"),
      })
    },

    "permission.ask": async (input, output) => {
      // The one hook here whose input shape was NOT measured: a permission prompt never
      // fired during the spike, so which field carries the tool name is inferred from
      // the type definitions rather than from a receipt. The keys are logged once so the
      // first real invocation says what actually arrives, and the failure mode if the
      // guess is wrong is benign -- the match misses, nothing is auto-approved, and the
      // user is asked exactly as they are today.
      if (!askShapeLogged) {
        askShapeLogged = true
        note("hooks", `permission.ask keys=${Object.keys(input ?? {}).join(",")}`)
      }
      const tool = input?.type ?? input?.permission ?? input?.title ?? ""
      const reply = await runHook({
        hooksDir: HOOKS_DIR, hook: "approve", host: HOST,
        payload: { session_id: input?.sessionID ?? "", tool_name: String(tool) },
        timeoutMs: TIMEOUTS.approve,
      })
      // Only ever widens to "allow". A hook that could deny would be able to block a
      // tool call the user asked for, which is not what auto-approving reads is for.
      if (reply?.status === "allow") output.status = "allow"
    },

    event: async ({ event }) => {
      if (event?.type !== "session.idle") return
      const sessionID = event?.properties?.sessionID ?? event?.properties?.sessionId ?? ""
      if (!sessionID) return
      const transcript = await writeTranscript(sessionID)
      if (!transcript) return
      // Not awaited: see shim.runHookDetached.
      runHookDetached({
        hooksDir: HOOKS_DIR, hook: "capture", host: HOST,
        payload: { session_id: sessionID, cwd: directory ?? worktree ?? "",
                   transcript_path: transcript },
        timeoutMs: TIMEOUTS.capture,
      })
    },
  }
}

export default MemvaraPlugin
