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

import { note, runHook, runHookDetached } from "./shim.mjs"

const HOST = "opencode"
const HOOKS_DIR = new URL("..", import.meta.url).pathname

/**
 * Sessions whose `session_start` has already run.
 *
 * In memory rather than on disk, and correct only because an OpenCode plugin is loaded
 * once into a server process that outlives the turn -- the same property that makes
 * detached capture work. If that ever stops being true this degrades to running
 * `session_start` more often, never to running it never.
 */
const started = new Set()

const TIMEOUTS = { session_start: 20000, recall: 10000, capture: 120000, approve: 5000 }

export const MemvaraPlugin = async ({ client, directory, worktree }) => {
  note("hooks", `opencode plugin loaded dir=${HOOKS_DIR}`)

  /** Materialise a transcript OpenCode never hands us, in the shape `lib.transcript` reads. */
  const writeTranscript = async (sessionID) => {
    try {
      const res = await client.session.messages({ path: { id: sessionID } })
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

      const payload = { session_id: sessionID, cwd: directory ?? worktree ?? "", prompt }
      const blocks = []

      if (sessionID && !started.has(sessionID)) {
        started.add(sessionID)
        const reply = await runHook({
          hooksDir: HOOKS_DIR, hook: "session_start", host: HOST,
          payload, timeoutMs: TIMEOUTS.session_start,
        })
        if (reply.additionalContext) blocks.push(reply.additionalContext)
      }

      const reply = await runHook({
        hooksDir: HOOKS_DIR, hook: "recall", host: HOST,
        payload, timeoutMs: TIMEOUTS.recall,
      })
      if (reply.additionalContext) blocks.push(reply.additionalContext)
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
      const tool = input?.type ?? input?.permission ?? input?.title ?? ""
      const reply = await runHook({
        hooksDir: HOOKS_DIR, hook: "approve", host: HOST,
        payload: { session_id: input?.sessionID ?? "", tool_name: String(tool) },
        timeoutMs: TIMEOUTS.approve,
      })
      // Only ever widens to "allow". A hook that could deny would be able to block a
      // tool call the user asked for, which is not what auto-approving reads is for.
      if (reply.status === "allow") output.status = "allow"
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
