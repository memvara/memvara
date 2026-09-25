"""Support code for the adversarial test suite. Nothing in this package is a test.

docs/claude/testing.md explains the suite. Every module here must be safe to import with
no side effects, because `--doctest-modules` imports each one while pytest collects.
"""
