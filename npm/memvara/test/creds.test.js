"use strict";

const { test } = require("node:test");
const assert = require("node:assert");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const { resolve, DEFAULT_SERVER } = require("../lib/creds.js");

test("MEMVARA_API_KEY wins, and names itself as the source", () => {
  const got = resolve({ env: { MEMVARA_API_KEY: "sk-env" } });
  assert.equal(got.token, "sk-env");
  assert.equal(got.source, "MEMVARA_API_KEY");
  assert.equal(got.serverUrl, DEFAULT_SERVER);
});

test("--server beats MEMVARA_SERVER_URL beats the default", () => {
  assert.equal(
    resolve({ env: { MEMVARA_API_KEY: "k", MEMVARA_SERVER_URL: "https://env" },
              serverOverride: "https://flag" }).serverUrl,
    "https://flag",
  );
  assert.equal(
    resolve({ env: { MEMVARA_API_KEY: "k", MEMVARA_SERVER_URL: "https://env" } }).serverUrl,
    "https://env",
  );
});

test("a whitespace-only key is not a key", () => {
  // Otherwise `MEMVARA_API_KEY=` in a .env file resolves to a credential, and the
  // bridge sends `Authorization: Bearer ` and reports a 401 the reader cannot explain.
  const got = resolve({ env: { MEMVARA_API_KEY: "   " } });
  assert.notEqual(got.source, "MEMVARA_API_KEY");
});

test("with no credential anywhere, token is null and the server still resolves", () => {
  // HOME is redirected so this does not read the developer's real login file.
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "memvara-creds-"));
  const realHome = process.env.HOME;
  const realUserProfile = process.env.USERPROFILE;
  try {
    process.env.HOME = tmp;
    process.env.USERPROFILE = tmp;
    delete require.cache[require.resolve("../lib/creds.js")];
    const fresh = require("../lib/creds.js");
    const got = fresh.resolve({ env: {} });
    assert.equal(got.token, null);
    assert.equal(got.source, null);
    assert.equal(got.serverUrl, DEFAULT_SERVER);
  } finally {
    process.env.HOME = realHome;
    if (realUserProfile === undefined) delete process.env.USERPROFILE;
    else process.env.USERPROFILE = realUserProfile;
    delete require.cache[require.resolve("../lib/creds.js")];
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test("a corrupt credentials file is skipped, not fatal", () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "memvara-creds-"));
  const realHome = process.env.HOME;
  const realUserProfile = process.env.USERPROFILE;
  try {
    process.env.HOME = tmp;
    process.env.USERPROFILE = tmp;
    fs.mkdirSync(path.join(tmp, ".memvara"));
    fs.writeFileSync(path.join(tmp, ".memvara", "credentials.json"), "{not json");
    delete require.cache[require.resolve("../lib/creds.js")];
    const fresh = require("../lib/creds.js");
    assert.doesNotThrow(() => fresh.resolve({ env: {} }));
    assert.equal(fresh.resolve({ env: {} }).token, null);
    // ...and an env key still works with the same broken file on disk.
    assert.equal(fresh.resolve({ env: { MEMVARA_API_KEY: "k" } }).token, "k");
  } finally {
    process.env.HOME = realHome;
    if (realUserProfile === undefined) delete process.env.USERPROFILE;
    else process.env.USERPROFILE = realUserProfile;
    delete require.cache[require.resolve("../lib/creds.js")];
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test("the login file's server_url is used, so a self-hosted key is not sent to the default", () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "memvara-creds-"));
  const realHome = process.env.HOME;
  const realUserProfile = process.env.USERPROFILE;
  try {
    process.env.HOME = tmp;
    process.env.USERPROFILE = tmp;
    fs.mkdirSync(path.join(tmp, ".memvara"));
    fs.writeFileSync(
      path.join(tmp, ".memvara", "credentials.json"),
      JSON.stringify({ api_key: "sk-file", project: "p", server_url: "https://self.hosted" }),
    );
    delete require.cache[require.resolve("../lib/creds.js")];
    const fresh = require("../lib/creds.js");
    const got = fresh.resolve({ env: {} });
    assert.equal(got.token, "sk-file");
    assert.equal(got.serverUrl, "https://self.hosted");
  } finally {
    process.env.HOME = realHome;
    if (realUserProfile === undefined) delete process.env.USERPROFILE;
    else process.env.USERPROFILE = realUserProfile;
    delete require.cache[require.resolve("../lib/creds.js")];
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});
