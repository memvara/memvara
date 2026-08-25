/**
 * Types for the module export, which is a signpost rather than an API.
 *
 * The shape has no methods on purpose, so using this package as a client is a compile
 * error at the earliest possible moment: `memvara.recall(...)` is TS2339, "Property
 * 'recall' does not exist". `isLibrary` is the literal `false` rather than `boolean` as a
 * secondary signal — it reads as `false` on hover, and `const ok: true =
 * memvara.isLibrary` is TS2322.
 *
 * It does **not** make `if (memvara.isLibrary) { ... }` an error. The branch narrows to
 * `never` and is unreachable, but TypeScript does not report type-dead branches and
 * `allowUnreachableCode: false` does not change that — it is a syntactic check.
 * Verified against tsc 5 rather than assumed.
 */
declare const memvara: {
  readonly isLibrary: false;
  readonly cli: string;
  readonly notice: string;
  readonly python: string;
  readonly homepage: string;
};

export = memvara;
