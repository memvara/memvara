"use strict";

/**
 * The bridge, end to end, against a real HTTP server and through a real spawn of the
 * real binary.
 *
 * Not `run()` called in-process. `release/rehearse_npm.py` was green through the
 * `npm publish` bug that cost a release, because it passed absolute `Path` objects while
 * the workflow passed a relative one — the rehearsal and the thing it rehearsed were
 * never running the same invocation. So this spawns `bin/memvara.js` the way `npx` will,
 * with argv and env and stdio as the client provides them.
 */

const { test } = require("node:test");
const assert = require("node:assert");
const http = require("node:http");
const path = require("node:path");
const { spawn } = require("node:child_process");

const BIN = path.join(__dirname, "..", "bin", "memvara.js");

const TOOLS = [
  "memory_add", "memory_remember", "memory_recall", "memory_search",
  "memory_neighborhood", "memory_paths", "memory_since", "memory_history",
  "memory_why", "memory_forget", "memory_end", "memory_stats",
];

/** A minimal MCP endpoint: enough to be answered, not a mock of one. */
function startServer({ requireToken = "sk-test" } = {}) {
  const seen = [];
  const server = http.createServer((req, res) => {
    let body = "";
    req.on("data", (c) => (body += c));
    req.on("end", () => {
      seen.push({ url: req.url, headers: req.headers, body });
      if (req.headers.authorization !== `Bearer ${requireToken}`) {
        res.writeHead(401, {
          "content-type": "application/json",
          "www-authenticate": 'Bearer resource_metadata="http://x/.well-known/r"',
        });
        res.end('{"error":"unauthorized"}');
        return;
      }
      const message = JSON.parse(body);
      if (message.id === undefined || message.id === null) {
        res.writeHead(202).end();
        return;
      }
      const headers = { "content-type": "application/json" };
      if (message.method === "initialize") headers["mcp-session-id"] = "sess-integration";
      let result = {};
      if (message.method === "initialize") {
        result = { protocolVersion: "2025-06-18", serverInfo: { name: "memvara", version: "t" } };
      } else if (message.method === "tools/list") {
        result = { tools: TOOLS.map((name) => ({ name, description: name, inputSchema: {} })) };
      }
      res.writeHead(200, headers).end(JSON.stringify({ jsonrpc: "2.0", id: message.id, result }));
    });
  });
  return new Promise((res) => server.listen(0, "127.0.0.1", () => res({ server, seen })));
}

/** Drive the spawned bridge: write lines, collect stdout lines, close, wait for exit. */
function drive(bin, args, env, lines) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [bin, ...args], {
      env: { ...process.env, ...env },
      stdio: ["pipe", "pipe", "pipe"],
    });
    let out = "";
    let err = "";
    child.stdout.on("data", (c) => (out += c));
    child.stderr.on("data", (c) => (err += c));
    child.on("error", reject);
    child.on("close", (code) =>
      resolve({
        code,
        stderr: err,
        messages: out.split("\n").filter(Boolean).map((l) => JSON.parse(l)),
      }),
    );
    for (const line of lines) child.stdin.write(JSON.stringify(line) + "\n");
    child.stdin.end();
  });
}

test("initialize and tools/list round-trip through the spawned binary", async (t) => {
  const { server, seen } = await startServer();
  t.after(() => server.close());
  const { port } = server.address();

  const { code, messages } = await drive(
    BIN,
    ["--server", `http://127.0.0.1:${port}`],
    { MEMVARA_API_KEY: "sk-test" },
    [
      { jsonrpc: "2.0", id: 1, method: "initialize", params: {} },
      { jsonrpc: "2.0", method: "notifications/initialized" },
      { jsonrpc: "2.0", id: 2, method: "tools/list", params: {} },
    ],
  );

  assert.equal(code, 0, "stdin closing is the client leaving, which is a clean exit");
  assert.equal(messages.length, 2, "a notification is never answered");
  assert.equal(messages[0].id, 1);
  assert.deepEqual(messages[1].result.tools.map((x) => x.name), TOOLS);

  // The endpoint is the server plus /mcp, and the session the server assigned on
  // initialize rides every later request.
  assert.ok(seen.every((r) => r.url === "/mcp"));
  assert.equal(seen[0].headers["mcp-session-id"], undefined);
  assert.equal(seen[2].headers["mcp-session-id"], "sess-integration");
});

test("a rejected credential explains which one, and does not kill the session", async (t) => {
  const { server } = await startServer({ requireToken: "sk-right" });
  t.after(() => server.close());
  const { port } = server.address();

  const { code, stderr, messages } = await drive(
    BIN,
    ["--server", `http://127.0.0.1:${port}`],
    { MEMVARA_API_KEY: "sk-wrong" },
    [{ jsonrpc: "2.0", id: 1, method: "initialize", params: {} }],
  );

  assert.equal(code, 0);
  assert.match(stderr, /rejected the credential from MEMVARA_API_KEY/);
  // The client asked a question; it gets an answer rather than a closed pipe.
  assert.equal(messages.length, 1);
  assert.equal(messages[0].id, 1);
  assert.ok(messages[0].error, "a failed request becomes a JSON-RPC error, not an exit");
});

test("an unparseable line is answered, addressed to null", async (t) => {
  const { server } = await startServer();
  t.after(() => server.close());
  const { port } = server.address();

  const child = spawn(process.execPath, [BIN, "--server", `http://127.0.0.1:${port}`], {
    env: { ...process.env, MEMVARA_API_KEY: "sk-test" },
    stdio: ["pipe", "pipe", "pipe"],
  });
  let out = "";
  child.stdout.on("data", (c) => (out += c));
  child.stdin.write("{not json\n");
  child.stdin.end();
  const code = await new Promise((r) => child.on("close", r));

  assert.equal(code, 0);
  const messages = out.split("\n").filter(Boolean).map((l) => JSON.parse(l));
  assert.equal(messages[0].id, null);
  assert.equal(messages[0].error.code, -32700);
});

test("with no credential it tries to sign in, and says why when it cannot", async (t) => {
  // Pointed at a server with no OAuth metadata, deliberately. An earlier version of this
  // test passed no --server, so it reached app.memvara.dev for real, registered a client
  // and opened a browser — a unit test that authenticates against production is a broken
  // test whatever it asserts. Discovery failing here is the fast, offline, local path
  // through the same code.
  const server = http.createServer((req, res) => res.writeHead(404).end());
  await new Promise((r) => server.listen(0, "127.0.0.1", r));
  t.after(() => server.close());
  const { port } = server.address();

  const emptyHome = path.join(__dirname, "no-such-home");
  const { code, stderr } = await drive(
    BIN,
    ["--server", `http://127.0.0.1:${port}`],
    { MEMVARA_API_KEY: "", HOME: emptyHome, USERPROFILE: emptyHome },
    [],
  );
  assert.equal(code, 2, "an unusable credential path is a usage error, not a crash");
  assert.match(stderr, /no credential found; signing in/);
  assert.match(stderr, /sign-in failed/);
  // Both fallbacks are named, because the reader has to be told a way forward.
  assert.match(stderr, /MEMVARA_API_KEY/);
  assert.match(stderr, /memvara-mcp login/);
});

test("`logout` is safe when there is nothing cached", async (t) => {
  const emptyHome = path.join(__dirname, "no-such-home");
  const { code, stderr } = await drive(BIN, ["logout"], { HOME: emptyHome, USERPROFILE: emptyHome }, []);
  assert.equal(code, 0);
  assert.match(stderr, /nothing to remove/);
});
