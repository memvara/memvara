"""Test doubles for the services a memvara client talks to. Nothing in this package is a
test, and importing it has no side effects.

Each fake stands in for one thing a client reaches over a network or starts as a
program, and each is checked in tests/adversarial/fakes/ by driving it with the real
client code it stands in for. A fake that drifts from what its client sends or reads
therefore fails its own tests before it can mislead any other test.

* `fake_v1.FakeV1` is the hosted `/v1` REST API that `memvara.remote` calls, answered by
  a real local `Memvara`.
* `hosted_mcp.FakeHostedMcp` is the hosted `/mcp` endpoint that the plugin's hooks and
  the npm bridge call, answered by the real MCP server code.
* `openai_compat.FakeOpenAI` is an OpenAI-compatible chat-completions endpoint that
  returns scripted replies.
* `cli.FakeClis` writes executables named `claude` and `codex` for the capture hook, and
  `cli.HangingClis` writes a `claude` and a `codex` that never answer.

`_http` holds what the three HTTP fakes share. Every server here listens on 127.0.0.1
only, so nothing reaches the network. docs/claude/testing.md explains how to use them.
"""
