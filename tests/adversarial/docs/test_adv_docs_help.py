"""Every option a command line accepts is in its help, and the help names only what the
configuration reads.

A person runs `--help` to find out why a client says the server failed, and copies what
it says into a settings file. A variable the help names and the configuration never reads
is one they would set to no effect, and an option the code accepts and the help leaves
out is one nobody can find.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any, Callable, Mapping

import pytest

from harness import known_bugs
from memvara.server.config import FEATURES, ServerConfig

from .commandline import (CommandLine, command_lines, config_reads, console_scripts,
                          default_off, false_claims, features, hides, nonexistent_features,
                          read_script, read_subcommand, reads_in, removes, subcommands,
                          undocumented, unread_variables, variable_defaults, variables)
from .planted import parse as parse_elsewhere
from .surface import configurations, served


# -- planted commands, read by the same code as the real ones ---------------------------

PLANTED_USAGE = """\
planted — a command for these tests.

  --name NAME   who to greet.
"""

_PLANTED_OPTIONS = ("--name", "--loud")


def planted(argv: list[str], *, stdout: Any = None) -> int:
    if "--help" in argv or "-h" in argv:
        print(PLANTED_USAGE, file=stdout)
        return 0
    return 0 if _planted_parse(argv) else 2


def _planted_parse(argv: list[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    rest = list(argv)
    while rest:
        name = rest.pop(0)
        if name not in _PLANTED_OPTIONS:
            raise ValueError(name)
        found[name] = rest.pop(0) if rest else ""
    return found


def planted_wrapper(argv: list[str], **options: Any) -> int:
    return planted(argv, **options)


PLANTED_MAIN_USAGE = """\
planted-main — the one command is greet.
"""


def planted_main(argv: list[str] | None = None, *, stdout: Any = None) -> int:
    args = list(argv or [])
    if args and args[0] == "greet":
        return planted_wrapper(args[1:], stdout=stdout)
    if args in (["--help"], ["-h"], ["help"]):
        print(PLANTED_MAIN_USAGE, file=stdout)
        return 0
    if args == ["--verbose"]:
        return 0
    return 2


PLANTED_ELSEWHERE_USAGE = """\
planted-elsewhere — a command whose options are parsed in another module.

  --city NAME   where to look.
"""


def planted_elsewhere(argv: list[str], *, stdout: Any = None) -> int:
    if "--help" in argv or "-h" in argv:
        print(PLANTED_ELSEWHERE_USAGE, file=stdout)
        return 0
    return 0 if parse_elsewhere(argv) else 2


#: Where the next planted command looks its handler up when it runs. No reader of the
#: source can follow a call through it.
_HANDLERS: dict[str, Callable[[list[str]], int]] = {}


def planted_opaque(argv: list[str], *, stdout: Any = None) -> int:
    if "--help" in argv or "-h" in argv:
        print(PLANTED_USAGE, file=stdout)
        return 0
    return _HANDLERS["run"](argv)


PLANTED_KEY = "MEMVARA_PLANTED_KEY"
_PLANTED_PREFIX = "MEMVARA_PLANTED_SWITCH_"


def planted_config(env: Mapping[str, str]) -> list[str]:
    env.get("MEMVARA_PLANTED")
    env.get(PLANTED_KEY)
    return [name for name in env if name.startswith(_PLANTED_PREFIX)]


PLANTED_HELP = """\
Configured by environment:

  MEMVARA_MODE        'local' (default) or 'cloud'.
  MEMVARA_TENANT      isolation boundary. Default 'default'.
  MEMVARA_USER        who. Unset means the whole tenant.
  MEMVARA_NOT_READ    a variable nothing reads.
  MEMVARA_FEATURE_<NAME>
                     '0' switches one off. Every feature is on by default except
                     AGENTIC_EXTRACTION and EXTRACTION_CHUNKS. PROFILE=0 hides
                     memory_profile, LINKS=0 hides memory_why, and END_REASON=0 removes
                     the reason and until_reason arguments. PROFLE=0 is accepted, and
                     the OS keychain is read.
  MEMVARA_READ_W_GRAPH  weight. Unset means 0.0, the leg off.

Client configuration follows.
"""


def test_an_option_the_help_does_not_name_is_reported() -> None:
    command = read_subcommand("planted", planted)
    assert command.accepted == {"--help", "-h", "--name", "--loud"}
    assert command.help_words == {"--help", "-h"}
    assert undocumented(command) == ["--loud"]


def test_a_wrapper_is_followed_to_the_help_it_hands_over_to() -> None:
    command = read_subcommand("planted-main greet", planted_wrapper)
    assert command.function is planted
    assert command.help == PLANTED_USAGE
    assert undocumented(command) == ["--loud"]


def test_options_parsed_in_another_module_are_read() -> None:
    """A command that hands its arguments to a parser in another module accepts that
    parser's options. Reading only the command's own module would find none of them, and
    the check would pass on nothing."""
    command = read_subcommand("planted-elsewhere", planted_elsewhere)
    assert command.accepted == {"--help", "-h", "--city", "--quiet"}
    assert undocumented(command) == ["--quiet"]


def test_arguments_handed_to_code_that_cannot_be_read_are_refused() -> None:
    """A command that hands its arguments to something the reader cannot follow may accept
    options nobody can list, so the reader refuses instead of reporting none."""
    with pytest.raises(LookupError, match="planted_opaque"):
        read_subcommand("planted-opaque", planted_opaque)


def test_a_script_is_read_from_its_dispatch() -> None:
    """The words that print the help, `-h` and `help` among them, are found from the code
    that prints it. So the help need not list them, and no list kept by hand can hide
    another word."""
    script = read_script("planted-main", planted_main)
    assert script.accepted == {"greet", "--help", "-h", "help", "--verbose"}
    assert script.help_words == {"--help", "-h", "help"}
    assert undocumented(script) == ["--verbose"]
    assert subcommands(planted_main) == {"greet": planted_wrapper}


def test_a_variable_the_configuration_does_not_read_is_reported() -> None:
    assert reads_in(sys.modules[__name__], ("one",)) == {
        "MEMVARA_PLANTED", "MEMVARA_PLANTED_KEY", "MEMVARA_PLANTED_SWITCH_ONE"}
    assert {"MEMVARA_NOT_READ", "MEMVARA_FEATURE_"} <= variables(PLANTED_HELP)
    assert unread_variables(PLANTED_HELP, config_reads()) == ["MEMVARA_NOT_READ"]


def test_the_configuration_reader_resolves_names_and_the_feature_prefix() -> None:
    """`MEMVARA_DB_KEY` is read through the name `KEY_ENV`, and every feature variable
    through one prefix, so both would be missed by a search for string literals."""
    reads = config_reads()
    assert {"MEMVARA_DB", "MEMVARA_DB_KEY", "MEMVARA_FEATURE_PROFILE"} <= reads
    assert "MEMVARA_NOT_READ" not in reads


def test_a_feature_that_does_not_exist_is_reported() -> None:
    assert features(PLANTED_HELP) == {"AGENTIC_EXTRACTION", "EXTRACTION_CHUNKS", "PROFILE",
                                      "LINKS", "END_REASON", "PROFLE"}
    assert nonexistent_features(PLANTED_HELP, FEATURES) == ["PROFLE"]


def test_a_claim_that_a_feature_hides_what_it_does_not_is_reported() -> None:
    assert hides(PLANTED_HELP) == [("profile", ("memory_profile",)),
                                   ("links", ("memory_why",))]
    assert removes(PLANTED_HELP) == [("end_reason", ("reason", "until_reason"))]
    everything = {"memory_profile", "memory_why", "memory_link"}
    listed = {"default": everything, "profile off": everything - {"memory_profile"},
              "links off": everything - {"memory_link"}, "end_reason off": everything}
    arguments = {"reason", "until_reason", "k"}
    taken = {"default": arguments, "profile off": arguments, "links off": arguments,
             "end_reason off": {"k"}}
    assert false_claims(PLANTED_HELP, listed, taken) == [("links", "memory_why")]


def test_a_claim_about_a_feature_off_by_default_is_checked_with_it_switched_on() -> None:
    """A feature that is off by default has no server with it switched off, only one with it
    switched on. Its claim holds when that server has the name and the default does not."""
    help = ("  MEMVARA_FEATURE_<NAME>\n"
            "                     AGENTIC_EXTRACTION=0 hides memory_think, and NOPE=0 hides\n"
            "                     memory_recall.\n")
    listed = {"default": {"memory_recall"},
              "agentic_extraction on": {"memory_recall", "memory_think"}}
    taken = {"default": {"k"}, "agentic_extraction on": {"k"}}
    assert false_claims(help, listed, taken) == [("nope", "memory_recall")]
    both = {"default": {"memory_recall", "memory_think"},
            "agentic_extraction on": {"memory_recall", "memory_think"}}
    assert false_claims(help, both, taken) == [("agentic_extraction", "memory_think"),
                                               ("nope", "memory_recall")]


def test_the_defaults_a_help_states_are_read() -> None:
    assert variable_defaults(PLANTED_HELP) == [("MEMVARA_MODE", "local"),
                                               ("MEMVARA_TENANT", "default"),
                                               ("MEMVARA_READ_W_GRAPH", "0.0")]
    assert default_off(PLANTED_HELP) == {"agentic_extraction", "extraction_chunks"}
    assert default_off("Every feature is on by default.") == frozenset()
    assert default_off("This help says nothing about features.") is None


# -- the real command lines --------------------------------------------------------------

SUBCOMMANDS = [command for command in command_lines() if not command.top]
SCRIPTS = [command for command in command_lines() if command.top]


def _helps() -> list[str]:
    """Each distinct help text a command line prints."""
    return list(dict.fromkeys(command.help for command in command_lines()))


def test_every_command_line_is_found() -> None:
    """The checks below pass on nothing if the reader finds nothing, so every console
    script must be found, each with at least one subcommand."""
    scripts = console_scripts()
    names = {command.name for command in SCRIPTS}
    assert set(scripts) - {"python -m memvara.server"} <= names
    assert scripts["python -m memvara.server"] is scripts["memvara-mcp"], (
        "python -m memvara.server must run memvara-mcp's code, because that is what the "
        "checks read")
    for name in names:
        assert any(command.name.startswith(name + " ") for command in SUBCOMMANDS), name


@pytest.mark.parametrize("command", SUBCOMMANDS, ids=[c.name for c in SUBCOMMANDS])
def test_every_option_a_subcommand_accepts_is_in_its_help(command: CommandLine) -> None:
    assert not undocumented(command), (
        f"{command.name} accepts {undocumented(command)}, and its help names none of them")


def test_every_variable_a_help_names_is_read_by_the_configuration() -> None:
    problems = [f"a help names {name}, and memvara/server/config.py never reads it"
                for help in _helps() for name in unread_variables(help, config_reads())]
    assert not problems, "\n".join(problems)
    assert any(variables(help) for help in _helps()), "no variable was read in any help"


def test_every_feature_the_help_names_exists() -> None:
    problems = [f"a help names the feature {name}, which does not exist"
                for help in _helps() for name in nonexistent_features(help, FEATURES)]
    assert not problems, "\n".join(problems)
    assert any(features(help) for help in _helps()), "no feature was read in any help"


def test_what_the_help_says_a_feature_hides_is_hidden() -> None:
    listed = {c.label: {tool["name"] for tool in served(c)} for c in configurations()}
    taken = {c.label: {argument for tool in served(c)
                       for argument in tool["inputSchema"]["properties"]}
             for c in configurations()}
    problems = [f"a help says switching {feature} off takes away {name}, and it does not"
                for help in _helps() for feature, name in false_claims(help, listed, taken)]
    assert not problems, "\n".join(problems)
    assert any(hides(help) and removes(help) for help in _helps()), "no claim was read"


def test_a_variable_set_to_its_stated_default_changes_nothing(tmp_path: pathlib.Path) -> None:
    base = {"MEMVARA_DB": ":memory:"}
    unset = ServerConfig.from_env(base, cwd=str(tmp_path))
    stated = [pair for help in _helps() for pair in variable_defaults(help)]
    problems = [f"a help says {variable} defaults to {value!r}, and setting it to that "
                "changes the configuration" for variable, value in stated
                if ServerConfig.from_env({**base, variable: value}, cwd=str(tmp_path)) != unset]
    assert not problems, "\n".join(problems)
    assert stated, "no stated default was read in any help"


def test_a_feature_set_to_its_stated_default_changes_nothing(tmp_path: pathlib.Path) -> None:
    base = {"MEMVARA_DB": ":memory:"}
    unset = ServerConfig.from_env(base, cwd=str(tmp_path))
    problems = []
    for help in _helps():
        off = default_off(help)
        if off is None:
            continue
        prefixes = [name for name in variables(help) if name.endswith("_")]
        assert prefixes, "a help says which features are on by default and names no prefix"
        for feature in FEATURES:
            variable = f"{prefixes[0]}{feature.upper()}"
            value = "0" if feature in off else "1"
            if ServerConfig.from_env({**base, variable: value}, cwd=str(tmp_path)) != unset:
                problems.append(f"the help says {feature} is {'off' if value == '0' else 'on'}"
                                f" by default, and {variable}={value} changes the "
                                "configuration")
    assert not problems, "\n".join(problems)
    assert any(default_off(help) is not None for help in _helps()), (
        "no help says which features are on by default")


#: The drift #297 pins: both console scripts accept --version, and neither help names it.
UNNAMED = ["--version"]


@pytest.mark.parametrize("command", [
    pytest.param(command, id=command.name, marks=[known_bugs.xfail("B21")])
    for command in command_lines() if command.top])
def test_every_word_a_console_script_accepts_is_in_its_help(command: CommandLine) -> None:
    words = undocumented(command)
    if words == UNNAMED:
        raise known_bugs.Reproduced(f"{command.name} accepts {words}, and its help names "
                                    "none of them")
    assert not words, f"{command.name} accepts {words}, and its help names none of them"
