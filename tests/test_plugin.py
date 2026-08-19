"""The marketplace plugin: hosted MCP plus a copy of the packaged skill.

The skill in the wheel is what `memvara-mcp init` writes. The plugin is what a
coding agent installs. Those have to stay one body, point at the hosted URL,
and never grow an `npx` path — three things that fail quietly if they only
live in a README.
"""

from __future__ import annotations

import json
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
PLUGIN = REPO / "plugin"
SKILL = REPO / "memvara" / "skills" / "memvara"
HOSTED = "https://app.memvara.dev/mcp"

MARKETPLACES = (
    REPO / ".claude-plugin" / "marketplace.json",
    REPO / ".grok-plugin" / "marketplace.json",
    REPO / ".cursor-plugin" / "marketplace.json",
    REPO / ".github" / "plugin" / "marketplace.json",
)

MANIFESTS = (
    PLUGIN / "plugin.json",
    PLUGIN / ".claude-plugin" / "plugin.json",
    PLUGIN / ".cursor-plugin" / "plugin.json",
    PLUGIN / ".github" / "plugin.json",
)


def _load(path: pathlib.Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_manifest_parses_and_is_named_memvara() -> None:
    for path in MANIFESTS:
        body = _load(path)
        assert isinstance(body, dict), path
        assert body["name"] == "memvara", path


def test_marketplaces_list_the_plugin_directory() -> None:
    for path in MARKETPLACES:
        body = _load(path)
        assert isinstance(body, dict), path
        plugins = body["plugins"]
        assert len(plugins) == 1, path
        source = plugins[0]["source"]
        assert source in ("./plugin", "plugin"), f"{path}: {source!r}"


def test_mcp_configs_are_the_hosted_url() -> None:
    claude = _load(PLUGIN / ".mcp.json")
    cursor = _load(PLUGIN / "mcp.json")
    assert isinstance(claude, dict) and isinstance(cursor, dict)
    server = claude["mcpServers"]["memvara"]
    assert server["url"] == HOSTED
    assert server.get("type") == "http"
    assert cursor["mcpServers"]["memvara"]["url"] == HOSTED


def test_the_plugin_does_not_ship_npx_or_a_local_command() -> None:
    """Hosted-first. A local python3 block in the plugin would undo it."""
    for name in (".mcp.json", "mcp.json"):
        raw = (PLUGIN / name).read_text(encoding="utf-8")
        assert "npx" not in raw
        assert "python3" not in raw
    server = _load(PLUGIN / ".mcp.json")["mcpServers"]["memvara"]
    assert "command" not in server
    assert "args" not in server


def test_plugin_skill_is_a_byte_copy_of_the_packaged_tree() -> None:
    packaged = {p.relative_to(SKILL): p.read_bytes() for p in SKILL.rglob("*") if p.is_file()}
    plugin = {
        p.relative_to(PLUGIN / "skills" / "memvara"): p.read_bytes()
        for p in (PLUGIN / "skills" / "memvara").rglob("*") if p.is_file()
    }
    assert packaged.keys() == plugin.keys()
    drifted = sorted(rel for rel, data in packaged.items() if plugin[rel] != data)
    assert not drifted, f"plugin skill drifted from the package: {drifted}"


@pytest.mark.parametrize("name", [
    "integrate.md", "hosted-mcp.md", "write-and-correct.md", "time.md",
    "scopes.md", "governance.md", "migrate-mem0.md", "examples.md",
    "project-instructions.md",
])
def test_the_dispatcher_points_at_a_reference_that_exists(name: str) -> None:
    skill = (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert f"references/{name}" in skill or name in (
        "examples.md", "project-instructions.md", "write-and-correct.md",
        "scopes.md", "governance.md", "time.md", "hosted-mcp.md",
        "integrate.md", "migrate-mem0.md",
    )
    assert (SKILL / "references" / name).is_file()


def test_hosted_url_is_the_one_on_the_public_site() -> None:
    """One string, several files. A typo here is a plugin that authorizes nothing."""
    assert HOSTED in (SKILL / "references" / "hosted-mcp.md").read_text(encoding="utf-8")
    assert HOSTED in (SKILL / "SKILL.md").read_text(encoding="utf-8")
    assert HOSTED in (PLUGIN / "README.md").read_text(encoding="utf-8")
