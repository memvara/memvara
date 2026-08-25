/**
 * OAuth, for the reader this package actually exists for.
 *
 * Without this the bridge tells a JavaScript developer with no Python to go and run
 * `memvara-mcp login` — a Python command — which would make "you do not need Python"
 * false in the one case it was written to serve. So the browser flow is not a nicety
 * here; it is the difference between the premise holding and not.
 *
 * The shape is the MCP authorization spec's, which is plain OAuth 2.1: discover the
 * authorization server from the protected resource's own metadata, register a public
 * client dynamically, then `authorization_code` with PKCE S256 and a loopback redirect.
 * `app.memvara.dev` advertises every piece of that at
 * `/.well-known/oauth-authorization-server`, so nothing below is invented for it.
 *
 * Two deliberate refusals, both borrowed from `memvara/server/login.py`, which solved
 * this problem once already on the Python side:
 *
 *   * **`state` is checked before the code is spent.** Anything on the machine can
 *     connect to `127.0.0.1`, so a callback arriving with the wrong state is somebody
 *     else's browser, or an attacker's, and the code in it is not ours to redeem.
 *   * **The token file is created at 0600 before any content lands**, not chmod'd after.
 *     A process that dies between the two would otherwise leave a readable token.
 */

"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");
const { spawn } = require("node:child_process");

const { OAUTH_PATH } = require("./creds.js");

const base64url = (buf) => buf.toString("base64url");

function pkce() {
  const verifier = base64url(crypto.randomBytes(32));
  const challenge = base64url(crypto.createHash("sha256").update(verifier).digest());
  return { verifier, challenge };
}

async function json(url, init, userAgent) {
  const response = await fetch(url, {
    ...init,
    headers: { accept: "application/json", "user-agent": userAgent, ...(init?.headers || {}) },
  });
  const text = await response.text();
  if (!response.ok) {
    throw new Error(`${url} answered HTTP ${response.status}${text ? `: ${text.slice(0, 300)}` : ""}`);
  }
  return text ? JSON.parse(text) : {};
}

/** Ask the resource where its authorization server is, rather than assuming. */
async function discover(serverUrl, userAgent) {
  const base = serverUrl.replace(/\/+$/, "");
  const meta = await json(`${base}/.well-known/oauth-authorization-server`, {}, userAgent);
  for (const field of ["authorization_endpoint", "token_endpoint"]) {
    if (!meta[field]) throw new Error(`${base} advertises no ${field}`);
  }
  return meta;
}

function openBrowser(url) {
  // Best effort by design. A sandbox, a headless box or a machine with no browser is a
  // normal environment, not an error — the URL is printed either way and typing it works.
  const cmd = process.platform === "darwin" ? "open"
    : process.platform === "win32" ? "cmd" : "xdg-open";
  const args = process.platform === "win32" ? ["/c", "start", "", url] : [url];
  try {
    spawn(cmd, args, { stdio: "ignore", detached: true }).unref();
    return true;
  } catch {
    return false;
  }
}

function saveTokens(tokens, serverUrl) {
  const dir = path.dirname(OAUTH_PATH);
  fs.mkdirSync(dir, { recursive: true, mode: 0o700 });
  // Created empty at 0600 first; content only after the mode is right.
  fs.closeSync(fs.openSync(OAUTH_PATH, "w", 0o600));
  fs.chmodSync(OAUTH_PATH, 0o600);
  fs.writeFileSync(
    OAUTH_PATH,
    JSON.stringify(
      {
        access_token: tokens.access_token,
        refresh_token: tokens.refresh_token || null,
        expires_at: tokens.expires_in ? Date.now() + tokens.expires_in * 1000 : null,
        server_url: serverUrl,
        client_id: tokens.client_id || null,
      },
      null,
      2,
    ) + "\n",
    { encoding: "utf8", mode: 0o600 },
  );
  return OAUTH_PATH;
}

/** Full browser flow. Returns `{access_token, ...}` and writes the cache. */
async function login({ serverUrl, userAgent, stderr = process.stderr }) {
  const meta = await discover(serverUrl, userAgent);
  const resource = `${serverUrl.replace(/\/+$/, "")}/mcp`;

  let port;
  const callback = new Promise((resolve, reject) => {
    // A ceiling, for the same reason `memvara/server/login.py` keeps one: without it a
    // user who closes the tab leaves this process waiting forever, and the MCP client
    // that spawned it just shows a server that never started. Failing with a reason
    // after five minutes is strictly better than hanging with none.
    const deadline = setTimeout(() => {
      server.close();
      reject(new Error("timed out after 5 minutes waiting for the browser to come back"));
    }, 5 * 60 * 1000);
    deadline.unref();
    const server = http.createServer((req, res) => {
      const url = new URL(req.url, "http://127.0.0.1");
      if (url.pathname !== "/callback") return void res.writeHead(404).end();
      res.writeHead(200, { "content-type": "text/html; charset=utf-8" });
      res.end("<!doctype html><meta charset=utf-8><title>memvara</title>" +
              "<body style=\"font:16px system-ui;padding:3rem\">" +
              "<h1>Signed in.</h1><p>Close this tab and return to your terminal.</p>");
      clearTimeout(deadline);
      server.close();
      resolve(Object.fromEntries(url.searchParams));
    });
    server.on("error", (err) => {
      clearTimeout(deadline);
      reject(err);
    });
    server.listen(0, "127.0.0.1", () => {
      port = server.address().port;
      startAuthorize().catch((err) => {
        clearTimeout(deadline);
        server.close();
        reject(err);
      });
    });
  });

  let verifier;
  let state;
  let clientId;

  async function startAuthorize() {
    const redirectUri = `http://127.0.0.1:${port}/callback`;
    if (!meta.registration_endpoint) {
      throw new Error(
        `${serverUrl} advertises no registration_endpoint, so this client cannot register ` +
          "itself. Set MEMVARA_API_KEY instead, or run: memvara-mcp login --project NAME",
      );
    }
    const registered = await json(
      meta.registration_endpoint,
      {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          client_name: "memvara (npx)",
          redirect_uris: [redirectUri],
          grant_types: ["authorization_code", "refresh_token"],
          response_types: ["code"],
          token_endpoint_auth_method: "none",
        }),
      },
      userAgent,
    );
    clientId = registered.client_id;
    if (!clientId) throw new Error("the registration endpoint returned no client_id");

    const challenge = pkce();
    verifier = challenge.verifier;
    state = base64url(crypto.randomBytes(16));

    const authorize = new URL(meta.authorization_endpoint);
    authorize.searchParams.set("response_type", "code");
    authorize.searchParams.set("client_id", clientId);
    authorize.searchParams.set("redirect_uri", redirectUri);
    authorize.searchParams.set("code_challenge", challenge.challenge);
    authorize.searchParams.set("code_challenge_method", "S256");
    authorize.searchParams.set("state", state);
    // RFC 8707. Names which resource the token is for, so a token minted here cannot be
    // replayed against a different one on the same authorization server.
    authorize.searchParams.set("resource", resource);

    stderr.write(`memvara: opening ${authorize.origin}${authorize.pathname} to sign in\n`);
    if (!openBrowser(authorize.href)) {
      stderr.write("memvara: could not open a browser. Open this URL yourself:\n");
    }
    stderr.write(`  ${authorize.href}\n`);
  }

  const params = await callback;
  if (params.error) {
    throw new Error(`sign-in was refused: ${params.error_description || params.error}`);
  }
  if (params.state !== state) {
    // Anything on this machine can reach 127.0.0.1. A mismatched state is somebody
    // else's callback and its code is not ours to redeem.
    throw new Error("the callback's state did not match the request; refusing to use it");
  }
  if (!params.code) throw new Error("the callback carried no authorization code");

  const body = new URLSearchParams({
    grant_type: "authorization_code",
    code: params.code,
    redirect_uri: `http://127.0.0.1:${port}/callback`,
    client_id: clientId,
    code_verifier: verifier,
    resource,
  });
  const tokens = await json(
    meta.token_endpoint,
    { method: "POST", headers: { "content-type": "application/x-www-form-urlencoded" }, body },
    userAgent,
  );
  if (!tokens.access_token) throw new Error("the token endpoint returned no access_token");
  const file = saveTokens({ ...tokens, client_id: clientId }, serverUrl);
  stderr.write(`memvara: signed in; token cached in ${file}\n`);
  return tokens;
}

/** Spend a refresh token. Returns the new tokens, or null if it cannot be refreshed. */
async function refresh({ serverUrl, userAgent, refreshToken, clientId }) {
  if (!refreshToken || !clientId) return null;
  try {
    const meta = await discover(serverUrl, userAgent);
    const tokens = await json(
      meta.token_endpoint,
      {
        method: "POST",
        headers: { "content-type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({
          grant_type: "refresh_token",
          refresh_token: refreshToken,
          client_id: clientId,
          resource: `${serverUrl.replace(/\/+$/, "")}/mcp`,
        }),
      },
      userAgent,
    );
    if (!tokens.access_token) return null;
    saveTokens({ ...tokens, refresh_token: tokens.refresh_token || refreshToken, client_id: clientId }, serverUrl);
    return tokens;
  } catch {
    // A refresh that fails is not an error to report on its own — the caller falls back
    // to a full sign-in, which is a better outcome than a stack trace about a grant.
    return null;
  }
}

function logout() {
  try {
    fs.unlinkSync(OAUTH_PATH);
    return true;
  } catch {
    return false;
  }
}

module.exports = { login, refresh, logout, discover, pkce, saveTokens };
