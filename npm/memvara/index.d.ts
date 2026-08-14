/**
 * Types for the placeholder.
 *
 * What actually protects a caller is that this type has four properties and no methods,
 * so any attempt to use the package as a client is a compile error:
 * `memvara.recall(...)` is TS2339, "Property 'recall' does not exist". That is the
 * earliest point at which anyone can be told the package is empty.
 *
 * `implemented` is the literal `false` rather than `boolean` as a secondary signal: it
 * shows as `false` on hover, and anything assuming it could be true fails — `const ok:
 * true = memvara.implemented` is TS2322.
 *
 * It does **not** make `if (memvara.implemented) { ... }` an error, which an earlier
 * version of this comment claimed. The branch narrows to `never` and is unreachable, but
 * TypeScript does not report type-dead branches, and `allowUnreachableCode: false` does
 * not change that — it is a syntactic check. Verified against tsc 5, not assumed.
 */
declare const memvara: {
  readonly implemented: false;
  readonly notice: string;
  readonly python: string;
  readonly homepage: string;
};

export = memvara;
