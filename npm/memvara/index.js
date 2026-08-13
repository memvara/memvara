/**
 * A name reservation, not a client. Importing this gives you a notice and nothing else.
 *
 * Deliberately: no side effects, and **it does not throw**. A placeholder that throws on
 * import breaks a bundler's module graph and turns "you installed the wrong thing" into a
 * build failure several layers from its cause. Returning something inspectable lets
 * whoever got here by accident read why in the place they are already looking.
 */

"use strict";

const NOTICE =
  "memvara is a bitemporal memory layer for AI agents. This npm package is a name " +
  "reservation only — there is no JavaScript client yet, and this module exposes no " +
  "functionality. The library is Python: `pip install memvara`. If you are looking for " +
  "a JS client for the REST API, it does not exist; say so on the issue tracker and it " +
  "will help decide whether to build one.";

module.exports = {
  /** Always false. Check this rather than feature-detecting an API that is not here. */
  implemented: false,
  notice: NOTICE,
  python: "pip install memvara",
  homepage: "https://github.com/memvara/memvara",
};
