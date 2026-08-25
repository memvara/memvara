/**
 * The stdio loop. Newline-delimited JSON in, newline-delimited JSON out.
 *
 * Framed to match `memvara/server/mcp.py`'s `handle_line`: one line in, at most one line
 * out, and a notification — no `id`, or an explicit null one — is never answered. A
 * client that speaks to the Python server over stdio and to this bridge should not be
 * able to tell which it got.
 *
 * The rule that matters most here is that **a failed request must become a JSON-RPC
 * error, not a dead process.** A bridge that exits on one bad response takes the whole
 * session with it, and the client sees a closed pipe rather than a reason — so every
 * throw below is converted to an error object addressed to the id that caused it. The
 * only fatal condition is stdin closing, which is the client leaving.
 */

"use strict";

const PARSE_ERROR = -32700;
const INTERNAL_ERROR = -32603;

function errorFor(id, code, message) {
  return { jsonrpc: "2.0", id: id ?? null, error: { code, message } };
}

/** JSON-RPC says a notification has no id; an explicit null is not addressable either. */
function isRequest(message) {
  return (
    message !== null &&
    typeof message === "object" &&
    !Array.isArray(message) &&
    message.id !== undefined &&
    message.id !== null
  );
}

/**
 * Run until `stdin` ends.
 *
 * `onFatal` is called for conditions that are about the *connection* rather than about
 * one message — a rejected credential, say. It gets to decide whether to keep going,
 * because "the token expired" is recoverable in a way that "the request was malformed"
 * is not.
 */
async function run({ stdin, stdout, stderr, transport, onUnauthorized = null }) {
  let buffer = "";
  const write = (message) => stdout.write(JSON.stringify(message) + "\n");

  const handleLine = async (line) => {
    if (!line.trim()) return;

    let message;
    try {
      message = JSON.parse(line);
    } catch {
      // No id is recoverable from an unparseable line, so this is addressed to null —
      // which is exactly what the Python server does with the same input.
      write(errorFor(null, PARSE_ERROR, "line is not valid JSON"));
      return;
    }

    const wantsReply = isRequest(message);

    try {
      const responses = await transport.send(message);
      for (const response of responses) write(response);
    } catch (err) {
      if (err.name === "Unauthorized" && onUnauthorized) {
        const recovered = await onUnauthorized(err);
        if (recovered) {
          try {
            for (const response of await transport.send(message)) write(response);
            return;
          } catch (again) {
            err.message = again.message;
          }
        }
      }
      stderr.write(`memvara: ${err.message}\n`);
      if (wantsReply) write(errorFor(message.id, INTERNAL_ERROR, err.message));
    }
  };

  // Lines are handled strictly in order. Concurrency here would let two responses race
  // onto stdout interleaved, and a client reading line-delimited JSON would see one
  // corrupt line rather than two good ones.
  let chain = Promise.resolve();
  stdin.setEncoding("utf8");
  for await (const chunk of stdin) {
    buffer += chunk;
    let newline;
    while ((newline = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, newline);
      buffer = buffer.slice(newline + 1);
      chain = chain.then(() => handleLine(line));
    }
  }
  chain = chain.then(() => handleLine(buffer));
  await chain;
}

module.exports = { run, isRequest, errorFor, PARSE_ERROR, INTERNAL_ERROR };
