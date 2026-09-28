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

import { createHash } from "node:crypto"
import fs from "node:fs"
import os from "node:os"
import path from "node:path"
import { fileURLToPath } from "node:url"

import { note, privateDir, runHook, runHookDetached, writePrivate } from "./shim.mjs"

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

/**
 * Where a session's transcript is written for capture to read: the account's own hook
 * directory, 0700, with the file 0600 (`shim.privateDir`, `shim.writePrivate`). A whole
 * conversation goes into it. It used to be `$TMPDIR/memvara-opencode`, with the default
 * modes, and on Linux `$TMPDIR` is usually the shared `/tmp`, so every account on the
 * machine could read it, and a directory of that fixed name could be made there first by
 * another account.
 */
const TRANSCRIPTS = path.join(os.homedir(), ".memvara", ".hooks", "opencode")

/** An OpenCode session id, which becomes a file name. Anything else is not written. */
const SESSION_ID = /^[A-Za-z0-9_-]{1,200}$/

/**
 * For each session, a digest of the transcript last handed to capture, and how many
 * captures are still reading its file.
 *
 * The file is removed when the last capture reading it is done, so the capture hook's own
 * watermark, which it keys on the file, cannot tell a repeated idle event from a new
 * reply. This remembers instead: a transcript with exactly the content of the one last
 * captured holds no new reply, and starts no capture. The digest is of the content, not
 * its length, because a different conversation can be exactly as long, for example when a
 * reply is undone and another of the same length takes its place. In memory, for the
 * reason `started` is.
 */
const captured = new Map()
const reading = new Map()

/**
 * Remove the transcripts an earlier version left in the shared `$TMPDIR`, where every
 * account could read them and nothing prunes them any more. Only this account's own
 * files can be removed there, and anything that cannot is left.
 */
function removeOldTranscripts() {
  const old = path.join(os.tmpdir(), "memvara-opencode")
  let names = []
  try { names = fs.readdirSync(old) } catch { return }
  for (const name of names) {
    if (!name.endsWith(".jsonl")) continue
    try { fs.unlinkSync(path.join(old, name)) } catch { /* not ours to remove */ }
  }
  try { fs.rmdirSync(old) } catch { /* not empty, or not ours */ }
}

export const MemvaraPlugin = async ({ client, directory, worktree }) => {
  note("hooks", `opencode plugin loaded dir=${HOOKS_DIR}`)
  removeOldTranscripts()

  /** Materialise a transcript OpenCode never hands us, in the shape `lib.transcript` reads.
   * Returns its path, or `""` when there is nothing new for capture to read. */
  const writeTranscript = async (sessionID) => {
    if (!SESSION_ID.test(sessionID)) {
      note("hooks", "skipped=transcript: the session id is not a plain file name")
      return ""
    }
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
      const text = lines.join("\n") + "\n"
      const digest = createHash("sha256").update(text).digest("hex")
      if (digest === captured.get(sessionID)) return ""
      privateDir(TRANSCRIPTS)
      // One file per session, removed when the capture that reads it is done (`event`
      // below). A newer transcript replaces it by a rename (`shim.writePrivate`), so a
      // capture still reading the older one is not disturbed. A file left behind, by a
      // server that stopped while a capture ran, is pruned by age here.
      const cutoff = Date.now() - 24 * 60 * 60 * 1000
      for (const name of fs.readdirSync(TRANSCRIPTS)) {
        try {
          const full = path.join(TRANSCRIPTS, name)
          if (fs.statSync(full).mtimeMs < cutoff) fs.unlinkSync(full)
        } catch { /* another turn pruned it first */ }
      }
      const file = path.join(TRANSCRIPTS, `${sessionID}.jsonl`)
      writePrivate(file, text)
      captured.set(sessionID, digest)
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
      reading.set(sessionID, (reading.get(sessionID) ?? 0) + 1)
      // Not awaited: see shim.runHookDetached. The transcript is removed when the last
      // capture reading it is done, whether it finished, failed or was killed: it holds a
      // whole conversation, and nothing reads it afterwards.
      runHookDetached({
        hooksDir: HOOKS_DIR, hook: "capture", host: HOST,
        payload: { session_id: sessionID, cwd: directory ?? worktree ?? "",
                   transcript_path: transcript },
        timeoutMs: TIMEOUTS.capture,
      }).finally(() => {
        const left = (reading.get(sessionID) ?? 1) - 1
        if (left > 0) { reading.set(sessionID, left); return }
        reading.delete(sessionID)
        try { fs.unlinkSync(transcript) } catch { /* already gone */ }
      })
    },
  }
}

export default MemvaraPlugin
