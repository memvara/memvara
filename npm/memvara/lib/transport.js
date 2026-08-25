/**
 * Streamable HTTP, client half. One JSON-RPC message goes out; zero or more come back.
 *
 * The endpoint may answer a POST three different ways and all three are normal:
 *
 *   202/204 with no body   — a notification was accepted; there is nothing to reply.
 *   application/json       — exactly one response object.
 *   text/event-stream      — one or more responses, as SSE `data:` events, then close.
 *
 * Collapsing those into "return the messages" is this module's whole job, so the stdio
 * loop above it never learns which shape arrived.
 */

"use strict";

/** A 401 that carries what the server said, so the caller can name the remedy. */
class Unauthorized extends Error {
  constructor(message, { wwwAuthenticate = null } = {}) {
    super(message);
    this.name = "Unauthorized";
    this.wwwAuthenticate = wwwAuthenticate;
  }
}

/**
 * Split an SSE byte stream into events.
 *
 * Written as a fed-buffer rather than a line splitter because **a chunk boundary is not
 * an event boundary**: `data: {"jsonrpc"` and `":"2.0",...}\n\n` can and do arrive as two
 * reads, and a parser that treats each chunk as whole loses the message with no error at
 * all. The buffer keeps the tail until a blank line proves an event is complete.
 */
function createSseParser() {
  let buffer = "";
  return {
    feed(chunk) {
      buffer += chunk;
      const events = [];
      let split;
      // \n\n ends an event; \r\n\r\n is the same thing from a server that speaks CRLF.
      while ((split = buffer.search(/\r?\n\r?\n/)) !== -1) {
        const raw = buffer.slice(0, split);
        buffer = buffer.slice(split + buffer.match(/\r?\n\r?\n/)[0].length);
        const data = raw
          .split(/\r?\n/)
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trim())
          .join("\n");
        if (data) events.push(data);
      }
      return events;
    },
    /** Anything left when the stream closed without a trailing blank line. */
    flush() {
      const raw = buffer;
      buffer = "";
      const data = raw
        .split(/\r?\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim())
        .join("\n");
      return data ? [data] : [];
    },
  };
}

class HttpTransport {
  constructor({ endpoint, token, fetchImpl = globalThis.fetch, userAgent }) {
    this.endpoint = endpoint;
    this.token = token;
    this.fetch = fetchImpl;
    this.userAgent = userAgent;
    /** Set from the first response that carries one; sent on everything after. */
    this.sessionId = null;
  }

  headers() {
    const h = {
      "content-type": "application/json",
      // Both, because the server picks: JSON for a single reply, SSE when it wants to
      // stream. Sending only one of them is how a client gets a 406 it cannot explain.
      accept: "application/json, text/event-stream",
      "user-agent": this.userAgent,
    };
    if (this.token) h.authorization = `Bearer ${this.token}`;
    if (this.sessionId) h["mcp-session-id"] = this.sessionId;
    return h;
  }

  /** Send one message. Returns an array of decoded JSON-RPC messages, possibly empty. */
  async send(message) {
    const response = await this.fetch(this.endpoint, {
      method: "POST",
      headers: this.headers(),
      body: JSON.stringify(message),
    });

    // Captured before any status check: the server assigns a session on `initialize`
    // and every later request has to carry it, including if this one turns out to be
    // an error we retry.
    const session = response.headers.get("mcp-session-id");
    if (session) this.sessionId = session;

    if (response.status === 401) {
      throw new Unauthorized("the server rejected this credential", {
        wwwAuthenticate: response.headers.get("www-authenticate"),
      });
    }

    if (response.status === 202 || response.status === 204) return [];

    if (!response.ok) {
      const body = await response.text().catch(() => "");
      throw new Error(
        `${this.endpoint} answered HTTP ${response.status}${body ? `: ${body.slice(0, 400)}` : ""}`,
      );
    }

    const contentType = (response.headers.get("content-type") || "").toLowerCase();

    if (contentType.includes("text/event-stream")) {
      return await this.#readStream(response);
    }

    const text = await response.text();
    if (!text.trim()) return [];
    const decoded = JSON.parse(text);
    return Array.isArray(decoded) ? decoded : [decoded];
  }

  async #readStream(response) {
    const parser = createSseParser();
    const messages = [];
    const collect = (payloads) => {
      for (const payload of payloads) {
        try {
          messages.push(JSON.parse(payload));
        } catch {
          // A `data:` frame that is not JSON is not ours — SSE comments and keep-alive
          // pings look like this. Dropping it is right; failing the request is not.
        }
      }
    };

    const decoder = new TextDecoder();
    for await (const chunk of response.body) {
      collect(parser.feed(decoder.decode(chunk, { stream: true })));
    }
    collect(parser.feed(decoder.decode()));
    collect(parser.flush());
    return messages;
  }
}

module.exports = { HttpTransport, Unauthorized, createSseParser };
