"""The tool descriptions name only tools and arguments that exist.

A model reads these descriptions once and cannot check them. A description that names a
tool or an argument the server does not have sends the model to a call that fails, so
every name in them is checked here against the tool table, in every configuration a
server can be started in.
"""

from __future__ import annotations

from memvara.server.config import FEATURE_DEFAULTS
from memvara.server.tools import FEATURE_ARGUMENTS, TOOLS

from .surface import configurations, served, table, texts


def test_every_switch_has_a_configuration() -> None:
    """Every feature switch gets a server of its own, so no switch goes unchecked."""
    labels = [configuration.label for configuration in configurations()]

    assert len(labels) == len(set(labels)), labels
    assert {"default", "read-only", "anchored"} <= set(labels)
    for feature in FEATURE_DEFAULTS:
        assert len([label for label in labels if label.split()[0] == feature]) == 1, feature
    assert "profile off" in labels, "a feature that is on by default is switched off"
    assert "agentic_extraction on" in labels, "a feature that is off by default is switched on"


def test_each_configuration_serves_what_its_switch_says() -> None:
    """Each configuration is a real, different server, so a check run over all of them
    is not the default server checked twenty-five times."""
    owned_tools = [tool for tool in TOOLS if tool.feature]
    for configuration in configurations():
        tools = {tool["name"]: tool for tool in served(configuration)}
        for owned in owned_tools:
            hidden = (owned.feature in configuration.features_off
                      or (configuration.read_only and owned.writes))
            assert (owned.name in tools) is not hidden, (configuration.label, owned.name)
        for feature, removed in FEATURE_ARGUMENTS.items():
            if feature in configuration.features_off:
                for tool in tools.values():
                    assert not set(removed) & set(tool["inputSchema"]["properties"]), (
                        configuration.label, tool["name"])
        if configuration.read_only:
            assert all(tool["annotations"]["readOnlyHint"] for tool in tools.values())
        for tool in tools.values():
            anchored = tool["inputSchema"]["properties"].get("anchored")
            if anchored is not None:
                assert anchored["default"] is configuration.anchored, (
                    configuration.label, tool["name"])

    by_label = {configuration.label: configuration for configuration in configurations()}
    assert len(served(by_label["default"])) == len(TOOLS)
    assert "memory_profile" not in {t["name"] for t in served(by_label["profile off"])}
    recall = {t["name"]: t for t in served(by_label["synthesis off"])}["memory_recall"]
    assert "synthesize" not in recall["inputSchema"]["properties"]
    assert "memory_remember" not in {t["name"] for t in served(by_label["read-only"])}


def test_the_table_keeps_every_argument_whatever_the_switches() -> None:
    """The table is what a tool can take on some server, so arguments a switch removes
    are still in it."""
    assert {"synthesize", "query_rewrite", "anchored"} <= table()["memory_recall"]
    assert "reason" in table()["memory_forget"]
    assert set(table()) == {tool.name for tool in TOOLS}


def test_texts_labels_each_description_with_where_it_came_from() -> None:
    tool = {"name": "t", "description": "d",
            "inputSchema": {"properties": {"a": {"description": "x"}, "b": {}}}}

    assert texts(tool) == [("t", "d"), ("t.a", "x"), ("t.b", "")]
