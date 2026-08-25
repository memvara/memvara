/**
 * Where a bearer token comes from, in the order it is looked for.
 *
 * The order is the product decision, not a detail: `MEMVARA_API_KEY` first so a
 * container or a CI job can hand one in without touching disk, then the file
 * `memvara-mcp login` already writes, then OAuth. Reaching OAuth means the first two
 * missed, and the point of checking them first is that a developer who has already
 * logged in with the Python CLI gets a working bridge with **no browser at all** —
 * which is the whole reason this package exists rather than `mcp-remote`.
 *
 * `~/.memvara/credentials.json` is read and never written. `memvara/server/config.py`
 * owns that schema — `{api_key, project, server_url}` — and an OAuth token pair does not
 * fit in it. Writing our shape into their file would break `ServerConfig.from_env` for
 * the Python server on the same machine, silently, at the next start.
 */

"use strict";

const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const DEFAULT_SERVER = "https://app.memvara.dev";

/** The file `memvara-mcp login` writes. Read-only from here. */
const CREDENTIALS_PATH = path.join(os.homedir(), ".memvara", "credentials.json");

/** Ours. Separate file precisely so the one above keeps its schema. */
const OAUTH_PATH = path.join(os.homedir(), ".memvara", "oauth.json");

function readJson(file) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch {
    // Missing, unreadable, or not JSON are the same answer here: there is no credential
    // at this source, try the next one. A corrupt file must not be fatal — the next
    // source may well succeed, and failing the whole bridge over it would be worse.
    return null;
  }
}

/**
 * Resolve a token and the server it belongs to.
 *
 * Returns `{token, serverUrl, source}` or `{token: null, serverUrl, source: null}`.
 * `source` is carried so the error message on a 401 can name *which* credential was
 * rejected — "the key in MEMVARA_API_KEY" and "the key in ~/.memvara/credentials.json"
 * need different remedies, and a bridge that says only "unauthorized" makes the reader
 * guess which one it even tried.
 */
function resolve({ env = process.env, serverOverride = null } = {}) {
  const explicitServer =
    serverOverride || (env.MEMVARA_SERVER_URL || "").trim() || null;

  const fromEnv = (env.MEMVARA_API_KEY || "").trim();
  if (fromEnv) {
    return {
      token: fromEnv,
      serverUrl: explicitServer || DEFAULT_SERVER,
      source: "MEMVARA_API_KEY",
    };
  }

  const creds = readJson(CREDENTIALS_PATH);
  if (creds && typeof creds.api_key === "string" && creds.api_key.trim()) {
    return {
      token: creds.api_key.trim(),
      // An explicit override beats the file, but the file beats the default: a key
      // minted against a self-hosted console is worthless pointed at app.memvara.dev,
      // and `login` records which server issued it for exactly this reason.
      serverUrl:
        explicitServer ||
        (typeof creds.server_url === "string" && creds.server_url.trim()) ||
        DEFAULT_SERVER,
      source: CREDENTIALS_PATH,
    };
  }

  const oauth = readJson(OAUTH_PATH);
  if (oauth && typeof oauth.access_token === "string" && oauth.access_token.trim()) {
    return {
      token: oauth.access_token.trim(),
      serverUrl:
        explicitServer ||
        (typeof oauth.server_url === "string" && oauth.server_url.trim()) ||
        DEFAULT_SERVER,
      source: OAUTH_PATH,
      refreshToken: typeof oauth.refresh_token === "string" ? oauth.refresh_token : null,
      expiresAt: typeof oauth.expires_at === "number" ? oauth.expires_at : null,
    };
  }

  return { token: null, serverUrl: explicitServer || DEFAULT_SERVER, source: null };
}

module.exports = { resolve, DEFAULT_SERVER, CREDENTIALS_PATH, OAUTH_PATH };
