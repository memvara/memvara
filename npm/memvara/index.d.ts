/**
 * Types for the placeholder. `implemented` is a literal `false` rather than `boolean`
 * on purpose: a TypeScript caller who writes `if (memvara.implemented) { ... }` gets the
 * branch narrowed to `never` and finds out at compile time, which is the earliest anyone
 * can be told that this package does nothing.
 */
declare const memvara: {
  readonly implemented: false;
  readonly notice: string;
  readonly python: string;
  readonly homepage: string;
};

export = memvara;
