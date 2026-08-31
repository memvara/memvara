"""Tools that build a plugin repository's files from this tree. Never imported by a hook.

Nothing under `tools/` runs on the per-prompt path, and nothing here may be imported from
one: the budget there is ~30ms of interpreter startup plus whatever a body costs, and a
generator has no business inside it. It ships in the vendored copy because the tree is
vendored whole, by a plain copy with no path rewriting and no exclusions -- an exclusion
list is one more thing that can quietly stop covering a file.
"""
