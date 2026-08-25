/**
 * This package is a command, not a library.
 *
 * `require("memvara")` is almost certainly not what you wanted — run `npx memvara`, or
 * point your MCP client at the `memvara` binary this package installs. The export below
 * exists to say so at the moment somebody looks, and it keeps the property the name
 * reservation had before it: **it does not throw.** A module that throws on import
 * breaks a bundler's module graph and turns "wrong package" into a build failure several
 * layers from its cause. Returning something inspectable puts the explanation in the
 * place the reader is already standing.
 */

"use strict";

module.exports = {
  /** Always false. This package exposes no programmatic API; it ships a CLI. */
  isLibrary: false,
  /** What to run instead. */
  cli: "npx memvara",
  notice:
    "memvara is a command, not a library. `npx memvara` bridges a stdio MCP client to " +
    "the hosted memvara server; there is no JavaScript API to import. If your MCP " +
    "client speaks to remote servers itself, skip this package and point it at " +
    "https://app.memvara.dev/mcp. The memory engine is Python: `pip install memvara`.",
  python: "pip install memvara",
  homepage: "https://github.com/memvara/memvara",
};
