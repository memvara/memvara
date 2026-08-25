"use strict";

const { test } = require("node:test");
const assert = require("node:assert");

const { HttpTransport, Unauthorized, createSseParser } = require("../lib/transport.js");

/** A `fetch` stand-in returning a Response built from parts. */
function reply({ status = 200, headers = {}, body = "" }) {
  return async () => new Response(body, { status, headers });
}

test("an SSE event split across chunks is still one event", () => {
  // The bug this exists for loses a message with no error at all: a parser that treats
  // each chunk as whole silently drops a response the server did send.
  const parser = createSseParser();
  assert.deepEqual(parser.feed('data: {"jsonrpc"'), []);
  assert.deepEqual(parser.feed(':"2.0","id":7}\n\n'), ['{"jsonrpc":"2.0","id":7}']);
});

test("CRLF and multiple events in one chunk both parse", () => {
  const parser = createSseParser();
  assert.deepEqual(parser.feed('data: {"a":1}\r\n\r\ndata: {"b":2}\n\n'), [
    '{"a":1}',
    '{"b":2}',
  ]);
});

test("a json reply becomes exactly one message", async () => {
  const t = new HttpTransport({
    endpoint: "https://x/mcp",
    token: "k",
    userAgent: "t",
    fetchImpl: reply({
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ jsonrpc: "2.0", id: 1, result: {} }),
    }),
  });
  const got = await t.send({ jsonrpc: "2.0", id: 1, method: "initialize" });
  assert.equal(got.length, 1);
  assert.equal(got[0].id, 1);
});

test("an event-stream reply yields every message in it", async () => {
  const t = new HttpTransport({
    endpoint: "https://x/mcp",
    token: "k",
    userAgent: "t",
    fetchImpl: reply({
      headers: { "content-type": "text/event-stream" },
      body: 'data: {"jsonrpc":"2.0","id":1,"result":{}}\n\n' +
            'data: {"jsonrpc":"2.0","method":"notifications/message"}\n\n',
    }),
  });
  const got = await t.send({ jsonrpc: "2.0", id: 1, method: "x" });
  assert.equal(got.length, 2);
});

test("202 means accepted with nothing to say", async () => {
  const t = new HttpTransport({
    endpoint: "https://x/mcp", token: "k", userAgent: "t",
    fetchImpl: reply({ status: 202 }),
  });
  assert.deepEqual(await t.send({ jsonrpc: "2.0", method: "notifications/initialized" }), []);
});

test("the session id is captured and then sent on every later request", async () => {
  const seen = [];
  let first = true;
  const fetchImpl = async (url, init) => {
    seen.push(init.headers["mcp-session-id"]);
    const headers = { "content-type": "application/json" };
    if (first) {
      first = false;
      headers["mcp-session-id"] = "sess-42";
    }
    return new Response(JSON.stringify({ jsonrpc: "2.0", id: 1, result: {} }), { headers });
  };
  const t = new HttpTransport({ endpoint: "https://x/mcp", token: "k", userAgent: "t", fetchImpl });
  await t.send({ jsonrpc: "2.0", id: 1, method: "initialize" });
  await t.send({ jsonrpc: "2.0", id: 2, method: "tools/list" });
  assert.equal(seen[0], undefined, "nothing to send before the server assigns one");
  assert.equal(seen[1], "sess-42", "and it must ride every request after");
  assert.equal(t.sessionId, "sess-42");
});

test("401 raises Unauthorized carrying what the server said", async () => {
  const t = new HttpTransport({
    endpoint: "https://x/mcp", token: "bad", userAgent: "t",
    fetchImpl: reply({
      status: 401,
      headers: { "www-authenticate": 'Bearer resource_metadata="https://x/.well-known/r"' },
      body: '{"error":"unauthorized"}',
    }),
  });
  await assert.rejects(
    () => t.send({ jsonrpc: "2.0", id: 1, method: "x" }),
    (err) => {
      assert.ok(err instanceof Unauthorized);
      assert.match(err.wwwAuthenticate, /resource_metadata/);
      return true;
    },
  );
});

test("the Authorization and User-Agent headers are both set", async () => {
  let captured = null;
  const t = new HttpTransport({
    endpoint: "https://x/mcp", token: "sk-1", userAgent: "memvara-npm/0.1.0",
    fetchImpl: async (url, init) => {
      captured = init.headers;
      return new Response("{}", { headers: { "content-type": "application/json" } });
    },
  });
  await t.send({ jsonrpc: "2.0", id: 1, method: "x" });
  assert.equal(captured.authorization, "Bearer sk-1");
  // Not cosmetic: app.memvara.dev is behind Cloudflare, which 1010s some default
  // client user-agents at the edge — a 403 that never reaches the application.
  assert.equal(captured["user-agent"], "memvara-npm/0.1.0");
  assert.match(captured.accept, /application\/json/);
  assert.match(captured.accept, /text\/event-stream/);
});
