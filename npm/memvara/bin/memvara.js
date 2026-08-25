#!/usr/bin/env node
/**
 * `npx memvara` — connect a stdio MCP client to the hosted memvara server.
 *
 * Zero configuration is the point. There is no URL to paste and no client to register,
 * and when `MEMVARA_API_KEY` or the file `memvara-mcp login` already wrote is present,
 * **no browser opens at all**. A client with remote-MCP support does not need this
 * program — it can talk to the endpoint directly, and `README.md` says so rather than
 * letting someone install a bridge they had no use for. This is for the clients that
 * only speak stdio.
 *
 * Everything printed for a human goes to stderr, always. stdout is the protocol.
 */

"use strict";

const { resolve, DEFAULT_SERVER, CREDENTIALS_PATH, OAUTH_PATH } = require("../lib/creds.js");
const oauth = require("../lib/oauth.js");
const { HttpTransport } = require("../lib/transport.js");
const { run } = require("../lib/bridge.js");
const { version } = require("../package.json");

const USAGE = `memvara — bridge a stdio MCP client to the hosted memvara server

  npx memvara [--server URL]
  npx memvara login    sign in with a browser and cache the token
  npx memvara logout   forget the cached token

Options
  --server URL   the console to connect to (default ${DEFAULT_SERVER})
  --version      print the version and exit
  --help         print this and exit

Credentials, in the order they are looked for:
  1. MEMVARA_API_KEY
  2. ${CREDENTIALS_PATH}   (written by: memvara-mcp login --project NAME)
  3. ${OAUTH_PATH}   (written by: npx memvara login)

With none of the three present, the bridge signs you in with a browser on first run.
No Python required.

If your MCP client can talk to a remote server itself, you do not need this program —
point it at ${DEFAULT_SERVER}/mcp and let it do the OAuth.
`;

/** The client_id the cached sign-in registered, needed to spend its refresh token. */
function readClientId() {
  try {
    return JSON.parse(require("node:fs").readFileSync(OAUTH_PATH, "utf8")).client_id || null;
  } catch {
    return null;
  }
}

function parseArgs(argv) {
  const options = { server: null, command: null };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (i === 0 && (arg === "login" || arg === "logout")) {
      options.command = arg;
      continue;
    }
    if (arg === "--help" || arg === "-h") return { help: true };
    if (arg === "--version" || arg === "-v") return { version: true };
    if (arg === "--server") {
      options.server = argv[i + 1];
      if (!options.server) return { error: "--server needs a URL" };
      i += 1;
      continue;
    }
    if (arg.startsWith("--server=")) {
      options.server = arg.slice("--server=".length);
      continue;
    }
    return { error: `unexpected argument ${JSON.stringify(arg)}` };
  }
  return options;
}

async function main(argv = process.argv.slice(2), env = process.env) {
  const options = parseArgs(argv);
  if (options.help) {
    process.stderr.write(USAGE);
    return 0;
  }
  if (options.version) {
    process.stdout.write(`${version}\n`);
    return 0;
  }
  if (options.error) {
    process.stderr.write(`memvara: ${options.error}\n\n${USAGE}`);
    return 2;
  }

  const userAgent = `memvara-npm/${version} node/${process.versions.node}`;

  if (options.command === "logout") {
    process.stderr.write(
      oauth.logout()
        ? `memvara: removed ${OAUTH_PATH}\n`
        : `memvara: nothing to remove at ${OAUTH_PATH}\n`,
    );
    return 0;
  }

  let credential = resolve({ env, serverOverride: options.server });

  if (options.command === "login" || !credential.token) {
    // Signing in rather than telling a JavaScript developer to go and run a Python
    // command. Telling them that would make "you do not need Python" false in the one
    // case this package was written for.
    if (!credential.token && options.command !== "login") {
      process.stderr.write("memvara: no credential found; signing in.\n");
    }
    try {
      await oauth.login({ serverUrl: credential.serverUrl, userAgent });
    } catch (err) {
      process.stderr.write(
        `memvara: sign-in failed: ${err.message}\n` +
          "  Set MEMVARA_API_KEY instead, or run:  memvara-mcp login --project NAME\n" +
          `  which writes ${CREDENTIALS_PATH}.\n`,
      );
      return 2;
    }
    credential = resolve({ env, serverOverride: options.server });
    if (options.command === "login") return 0;
  }

  const endpoint = `${credential.serverUrl.replace(/\/+$/, "")}/mcp`;
  const transport = new HttpTransport({
    endpoint,
    token: credential.token,
    // Named, and not left to Node's default: `app.memvara.dev` sits behind Cloudflare,
    // which answers some default client user-agents with a 1010 at the edge — a 403 that
    // never reaches the application and says nothing about the client's *name* being the
    // problem. Measured on this host with Python's urllib; not a risk worth inheriting.
    userAgent,
  });

  process.stderr.write(
    `memvara ${version} → ${endpoint} (credential: ${credential.source})\n`,
  );

  await run({
    stdin: process.stdin,
    stdout: process.stdout,
    stderr: process.stderr,
    transport,
    onUnauthorized: async (err) => {
      // Refresh once and retry once. A cached access token expiring mid-session is
      // ordinary; making the user restart their editor over it is not. Only the OAuth
      // credential is refreshable — an API key that is rejected is simply wrong.
      if (credential.source === OAUTH_PATH && credential.refreshToken) {
        const renewed = await oauth.refresh({
          serverUrl: credential.serverUrl,
          userAgent,
          refreshToken: credential.refreshToken,
          clientId: readClientId(),
        });
        if (renewed && renewed.access_token) {
          transport.token = renewed.access_token;
          credential = resolve({ env, serverOverride: options.server });
          process.stderr.write("memvara: token refreshed\n");
          return true;
        }
      }
      process.stderr.write(
        `memvara: ${endpoint} rejected the credential from ${credential.source}.\n` +
          (credential.source === "MEMVARA_API_KEY"
            ? "  Check the value, or unset it to fall back to a cached sign-in.\n"
            : `  Run: npx memvara login  to sign in again.\n`) +
          (err.wwwAuthenticate ? `  server said: ${err.wwwAuthenticate}\n` : ""),
      );
      return false;
    },
  });
  return 0;
}

if (require.main === module) {
  main().then(
    (code) => process.exit(code),
    (err) => {
      process.stderr.write(`memvara: ${err && err.stack ? err.stack : err}\n`);
      process.exit(1);
    },
  );
}

module.exports = { main, parseArgs, USAGE };
