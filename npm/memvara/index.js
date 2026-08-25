/**
 * A name reservation, not a client. Importing this gives you a notice and nothing else.
 *
 * Deliberately: no side effects, and **it does not throw**. A placeholder that throws on
 * import breaks a bundler's module graph and turns "you installed the wrong thing" into a
 * build failure several layers from its cause. Returning something inspectable lets
 * whoever got here by accident read why in the place they are already looking.
 *
 * The notice changed in 0.0.2 and the reason is worth stating: 0.0.1 told a JavaScript
 * reader the library is Python and stopped there, which reads as "come back later". It is
 * not the situation. There is no JS client, and there is also nothing to wait for — the
 * Python package ships an MCP server, and MCP is the interface a JavaScript agent already
 * speaks. The sentence that was missing is the one that says so.
 */

"use strict";

const NOTICE =
  "memvara is a bitemporal memory layer for AI agents. This npm package is a name " +
  "reservation only — there is no JavaScript client, and this module exposes no " +
  "functionality. The library is Python: `pip install memvara`. You do not need a " +
  "JavaScript client to use it from a JavaScript agent: that same install provides " +
  "`memvara-mcp`, a JSON-RPC 2.0 MCP server over stdio, and the hosted console at " +
  "https://app.memvara.dev serves MCP over HTTP. If what you want is a JS client for " +
  "the REST API, say so on the issue tracker — the number of people who ask is most " +
  "of the answer.";

module.exports = {
  /** Always false. Check this rather than feature-detecting an API that is not here. */
  implemented: false,
  notice: NOTICE,
  python: "pip install memvara",
  homepage: "https://github.com/memvara/memvara",
};
