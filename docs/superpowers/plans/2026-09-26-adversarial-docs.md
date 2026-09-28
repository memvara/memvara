# Documentation that must match the code (A8, fast half) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fail the fast tier whenever the text a person or a model reads about memvara names something the code does not have: a command-line option missing from its help, an environment variable the configuration never reads, a tool or argument that does not exist, or a default stated in words that differs from the schema.

**Architecture:**
- The checks read the truth from the code, never from a list kept by hand. Tool names and arguments come from `TOOLS` and from the `tools/list` answer of a real `MemvaraMCPServer`. Command lines come from `pyproject.toml` and from the dispatch code, read with `ast`. The variables the configuration reads come from the `ast` of `memvara/server/config.py`.
- The mentions come from parsing the text: the tool descriptions, the server's `INSTRUCTIONS`, the packaged skill (`SKILL.md` and its `references/`), and every `--help` text.
- Each parser and check is a plain function in a helper module under `tests/adversarial/docs/`. Every check is first shown catching a planted fault, and then run on the real code.

**Tech Stack:** Python 3.10–3.13, pytest, the standard library's `ast`, `inspect` and `re`, and memvara's in-process `MemvaraMCPServer` over an in-memory store with `HashingEmbedder` and `NullLLM`.

**Spec:** `docs/superpowers/specs/2026-09-25-adversarial-test-suite-design.md`: the A8 row of the Phase 2 table (the fast half only) and done criterion 10, "The docs contract holds".

## Global Constraints

- Everything runs offline: stores from `tests/harness/stores.py`, no API key, no network.
- Parse the mentions from the text and read the truth from the code. No test keeps a hand list of what the text mentions.
- Do not edit `memvara/server/tools.py`, the packaged skill under `memvara/skills/memvara/`, or anything else under `memvara/`. Drift is a bug to report, not a file to fix here.
- A test that fails because of drift is not committed. The drift is reported to the maintainer with a reproduction, and the maintainer files the issue and lands the test as a strict expected failure. The check function it calls is committed, and is proven on a planted fault.
- The tests run on Python 3.10–3.13 on Linux, macOS and Windows. So nothing may need `tomllib` (3.11+), which rules out loading the predicate packs, and the parsers must accept Windows line endings.
- The files live in `tests/adversarial/docs/`, with an `__init__.py`. Test files are named `test_adv_docs_*.py`. The fast-tier budget for all of A8 is about 2 seconds.
- `tests/harness/checklist.py` does not exist on `main` yet, so no test carries a `covers` mark.
- Write plainly: every sentence must be understood on its first reading.

## What each check reads

| Check | The text | The truth |
|---|---|---|
| Every option a command line accepts is in its help | each `*USAGE` text a command prints | the dispatch and option-parsing code, read with `ast` |
| Every variable a help text names is one the configuration reads | every help text | `env.get(...)` calls in `config.py`, read with `ast`, plus `MEMVARA_FEATURE_<NAME>` for each feature |
| Every feature a help text names exists | the `MEMVARA_FEATURE_<NAME>` paragraph | `FEATURES` |
| A help text's "X=0 hides tool T" and "X=0 removes arguments A" claims hold | the server's help | `tools/list` with that feature off |
| A variable set to the value its help calls the default changes nothing | the server's help | `ServerConfig.from_env` |
| Every tool name in tool text is a tool | tool and argument descriptions, `INSTRUCTIONS` | `TOOLS` |
| Every snake_case identifier in tool text resolves | the same | tools, arguments, built-in predicates and their aliases |
| "tool with argument" names an argument of that tool | the same | `TOOLS` |
| A default stated in words equals the declared one | argument descriptions, in every switch configuration | `inputSchema` |
| Every tool, call and tied argument in the skill exists | `SKILL.md` and `references/*.md` | `TOOLS` |

## How the parse recognises a mention

This is the part a reviewer should check hardest, because a loose rule reports ordinary English and a strict rule misses renamed arguments.

- **A tool name** is `memory_` followed by lower-case words joined by `_`. `memory_*` is a wildcard, not a name.
- **An identifier** is a snake_case word: lower-case letters and digits in two or more parts joined by `_`. A word preceded by `.` belongs to a dotted library path and is left out.
- **An identifier resolves** when it is a tool, an argument of any tool, or a built-in predicate or one of its aliases. Otherwise it resolves only as example data: inside quotes, parentheses or braces, or in the list that follows "like", "such as" or "e.g.". A word followed by a colon labels the list after it, as "in snake_case:" does.
- **An argument is named** by a snake_case identifier, by a word written as `word=value`, by one of the tool's own arguments written in single quotes, or by a run of two or more of the tool's own arguments joined by commas, "and", "or" or "/", as in "ranked and synthesize". A single plain word such as "reason" or "query" is not a mention, because it is usually English.
- **A tie** says which tool an argument belongs to. In tool text: "`tool` with (the same) `argument`", "`tool` and `argument`", and "`tool`'s `argument`". In the skill, the same phrases written with backticks, plus "`argument` on `tool`", "`tool` takes `a` and `b`", "`tool` (optional `argument`)", "`tool` and `tool` both take ...", and "the ... argument is `x`".
- **A call** in the skill is `memory_x(` followed by arguments, read with `ast`. A call after a `#` on the same line is a comment, not an example.
- **A stated default** is a sentence that starts "Default" or "Defaults to", or a phrase "V is the (useful) default", "V by default" or "at the default V", where V is `true`, `false`, a number or a quoted string. "Defaults to now" and other phrases that are not a literal describe a value computed at call time. In a tool's own description, "'argument', default V" states that argument's default.

## Review Focus

1. **English words that are also argument names.** "a false reason", "the query", "the text" must not count as naming `reason`, `query` or `text`. Otherwise the `end_reason` switch, which removes `reason`, reports prose about stored reasons. Task 3 pins it on planted text and on the real `end_reason` configuration.
2. **Windows line endings.** A skill file checked out with CRLF must parse the same as with LF. Task 5 pins it by parsing a CRLF copy.
3. **A boolean default compared with a number.** `True == 1` in Python, so a naive comparison would accept "Default true" against a declared `1`. `0` and `0.0` must still compare equal. Task 4 pins both.
4. **A call spread over several lines, with a parenthesis inside a string.** The call must be read whole, not cut at the first `)`. Task 5 pins it.
5. **An alias that prints the help.** `-h`, `--help` and `help` print the help itself, so the help does not have to list them. They are found from the code (the branch that prints a `*USAGE` text), not from a list, so a new alias cannot hide an undocumented option. Task 6 pins it with a planted command.

---

### Task 1: The plan

**Files:**
- Create: `docs/superpowers/plans/2026-09-26-adversarial-docs.md`

- [ ] **Step 1: Commit the plan.**

```bash
git add docs/superpowers/plans/2026-09-26-adversarial-docs.md
git commit -m "Plan the adversarial suite's documentation checks"
```

### Task 2: The configurations a server can be started in

**Files:**
- Create: `tests/adversarial/docs/__init__.py`, `tests/adversarial/docs/surface.py`
- Test: `tests/adversarial/docs/test_adv_docs_tools.py`

**Interfaces:**
- Produces, in `surface.py`:
  - `Configuration(label: str, features_off: frozenset[str], read_only: bool, anchored: bool)`, a frozen dataclass.
  - `configurations() -> tuple[Configuration, ...]`: the default server; one per feature in `FEATURE_DEFAULTS`, switched away from its default (label `"<feature> off"` or `"<feature> on"`); `"read-only"`; `"anchored"`.
  - `served(configuration) -> tuple[dict[str, Any], ...]`: the `tools` of a `tools/list` answer from `MemvaraMCPServer(stores.memory(), user="u", ...)`, cached per configuration.
  - `table() -> dict[str, frozenset[str]]`: each tool in `TOOLS` and all of its arguments, whatever the switches.
  - `texts(tool: dict) -> list[tuple[str, str]]`: `("<tool>", description)` and `("<tool>.<argument>", description)` for one served tool.

- [ ] **Step 1: Write the failing tests.**
  - `test_every_switch_has_a_configuration`: the labels are `default`, one per feature, `read-only` and `anchored`, with no repeats.
  - `test_each_configuration_serves_what_its_switch_says`: a feature that owns tools (`Tool.feature`) hides them; a feature in `FEATURE_ARGUMENTS` removes its arguments from every tool; read-only lists no tool whose `readOnlyHint` is false; anchored declares `anchored` with default true. This proves the later checks run on real, different configurations.
- [ ] **Step 2: Run them and watch them fail on the missing module.**
- [ ] **Step 3: Write `surface.py`.**
- [ ] **Step 4: Run them and watch them pass.**
- [ ] **Step 5: Commit** `__init__.py`, `surface.py` and the test file.

### Task 3: Tool names and argument names in the tool descriptions

**Files:**
- Create: `tests/adversarial/docs/mentions.py`
- Modify: `tests/adversarial/docs/test_adv_docs_tools.py`, `docs/claude/testing.md`

**Interfaces:**
- Consumes: `surface.configurations`, `surface.served`, `surface.table`, `surface.texts`.
- Produces, in `mentions.py`:
  - `Mention(word: str, kind: str, start: int)`, where `kind` is `"snake"`, `"assigned"`, `"quoted"` or `"listed"`.
  - `tool_names(text) -> list[str]`, `mentions(text, own: Collection[str]) -> list[Mention]`, `ties(text) -> list[tuple[str, str]]`.
  - `predicates() -> frozenset[str]`: built-in predicate names and aliases, from `BUILTIN_PREDICATES`.
  - The checks, each returning a list of problems as sentences, empty when all is well:
    - `unknown_tools(text, table) -> list[str]`;
    - `unresolved_identifiers(text, table) -> list[str]`;
    - `wrong_ties(text, table) -> list[str]`;
    - `unserved_arguments(text, own: Collection[str], served: Collection[str]) -> list[str]`, for the arguments of the text's own tool: every mention of one of `own` must be in `served`.

- [ ] **Step 1: Write the failing tests on planted text.** One per rule in "How the parse recognises a mention", each shown both catching its fault and leaving the right text alone:
  - a misspelled tool name is reported, and `memory_*` is not;
  - a renamed argument ("send valid_from instead") is reported, and a predicate, an alias, a quoted id, a parenthesised example, a "like" list and a "snake_case:" label are not;
  - "memory_search with include_episodes" is reported, because that argument belongs to `memory_recall`;
  - with `synthesize` removed, "ranked and synthesize" and "(query_rewrite)" are reported, and "a false reason" is not (Review Focus 1).
- [ ] **Step 2: Run them and watch them fail on the missing module.**
- [ ] **Step 3: Write `mentions.py`.**
- [ ] **Step 4: Write the real-data tests,** each over every configuration, collecting every problem before asserting so one failure lists them all:
  - `test_every_tool_the_descriptions_name_exists`;
  - `test_every_identifier_in_the_descriptions_resolves`;
  - `test_a_tool_named_with_an_argument_takes_it`;
  - the same three over `INSTRUCTIONS`, in the same tests.
  - `test_a_tools_own_argument_named_in_its_text_is_served`: run it. If it fails, take it out of the file and report the drift with a reproduction.
- [ ] **Step 5: Run the file and watch the committed tests pass.**
- [ ] **Step 6: Add the section "Documentation that must match the code" to `docs/claude/testing.md`,** just before its last line, describing the checks so far.
- [ ] **Step 7: Commit** `mentions.py`, the test file and `testing.md`.

### Task 4: Defaults stated in words

**Files:**
- Create: `tests/adversarial/docs/defaults.py`
- Test: `tests/adversarial/docs/test_adv_docs_defaults.py`
- Modify: `docs/claude/testing.md`

**Interfaces:**
- Consumes: `surface.configurations`, `surface.served`.
- Produces, in `defaults.py`:
  - `Stated(argument: str, value: object, literal: bool, words: str)`, a frozen dataclass. `value` is `None` when `literal` is false.
  - `stated(description, argument) -> list[Stated]` for an argument's own description, and `stated_in_tool(description) -> list[Stated]` for "'argument', default V" in a tool's description.
  - `same(stated_value, declared) -> bool`: a boolean equals only a boolean, a number equals a number of equal value, and a string equals the same string.
  - `conflicts(tool: dict) -> list[str]`: a literal that differs from the declared default, and a value computed at call time on an argument that declares a constant default.
  - `undeclared(tool: dict) -> list[str]`: a literal stated in words on an argument that declares no default.

- [ ] **Step 1: Write the failing tests on planted text:** each phrase in "How the parse recognises a mention" is read, with its value; "the default order", "The default is not a harmless approximation" and "at the default and 41%" state nothing; `same(True, 1)` is false and `same(0, 0.0)` is true (Review Focus 3); a planted schema whose default differs from its words is reported by `conflicts`, and one with no default by `undeclared`.
- [ ] **Step 2: Run them and watch them fail on the missing module.**
- [ ] **Step 3: Write `defaults.py`.**
- [ ] **Step 4: Write the real-data tests,** over every configuration, so that the anchored server's "Default true on this server" is checked against its own schema:
  - `test_a_default_stated_in_words_equals_the_declared_default`;
  - `test_the_real_descriptions_state_defaults`: at least the known number are parsed, so that a parser that stopped reading would fail rather than pass on nothing.
  - `test_a_default_stated_in_words_is_declared_in_the_schema`: run it. If it fails, take it out of the file and report the drift with a reproduction.
- [ ] **Step 5: Run the file and watch the committed tests pass.**
- [ ] **Step 6: Add a paragraph on defaults to the `testing.md` section.**
- [ ] **Step 7: Commit** `defaults.py`, the test file and `testing.md`.

### Task 5: The packaged skill

**Files:**
- Create: `tests/adversarial/docs/skill.py`
- Test: `tests/adversarial/docs/test_adv_docs_skill.py`
- Modify: `docs/claude/testing.md`

**Interfaces:**
- Consumes: `mentions.tool_names`, `mentions.unknown_tools`, `surface.table`.
- Produces, in `skill.py`:
  - `skill_files() -> list[pathlib.Path]`: `SKILL.md`, then `references/*.md` in name order.
  - `Call(tool: str, keywords: tuple[str, ...], line: int)`; `calls(text) -> tuple[list[Call], list[str]]`, the calls and the ones that could not be read.
  - `ties(text) -> list[tuple[str, str]]`, the backtick forms.
  - `called_arguments(text) -> list[str]`: identifiers the text calls an argument.
  - The checks: `wrong_calls(text, table)`, `wrong_ties(text, table)`, `unknown_arguments(text, table)`, each returning a list of problems.

- [ ] **Step 1: Write the failing tests on planted text:** a call with an unknown argument is reported; the commented-out `memory_standing(query=...)` is not; a call over several lines with `")"` inside a string is read whole (Review Focus 4); every tie form is read; "`memory_search` takes `as_of`, not `known_at`" ties only `as_of`; "pass `true_since` on the new fact (or close it with `memory_end` and `at`)" ties only `at`; a CRLF copy of a real file gives the same result (Review Focus 2).
- [ ] **Step 2: Run them and watch them fail on the missing module.**
- [ ] **Step 3: Write `skill.py`.**
- [ ] **Step 4: Write the real-data tests:**
  - `test_every_tool_the_skill_names_exists`;
  - `test_every_call_in_the_skill_names_real_arguments`, which also asserts that the calls were found, so a parser that found none fails;
  - `test_an_argument_the_skill_ties_to_a_tool_belongs_to_it`;
  - `test_what_the_skill_calls_an_argument_is_one`.
- [ ] **Step 5: Run the file and watch it pass,** or take out a failing test and report the drift.
- [ ] **Step 6: Add a paragraph on the skill to the `testing.md` section.**
- [ ] **Step 7: Commit** `skill.py`, the test file and `testing.md`.

### Task 6: The command lines and their help

**Files:**
- Create: `tests/adversarial/docs/commandline.py`
- Test: `tests/adversarial/docs/test_adv_docs_help.py`
- Modify: `docs/claude/testing.md`

**Interfaces:**
- Consumes: `surface.configurations`, `surface.served`.
- Produces, in `commandline.py`:
  - `CommandLine(name: str, function: Callable, accepted: frozenset[str], help_words: frozenset[str], help: str, top: bool)`.
  - `console_scripts() -> dict[str, Callable]`: the `[project.scripts]` entries, read from `pyproject.toml` with a regular expression (no `tomllib` on 3.10), plus `python -m memvara.server`.
  - `command_lines() -> list[CommandLine]`: each console script, and each subcommand its dispatch code reaches. A subcommand is a branch `args[0] == "word"` or a table `args[0] in TABLE`. The help is the `*USAGE` text a branch prints; a wrapper with no help branch is followed to the function it returns a call to.
  - `undocumented(command) -> list[str]`: accepted words that are neither help words nor in the help text.
  - `config_reads() -> frozenset[str]`: every `MEMVARA_*` name `config.py` passes to `env.get`, with names such as `KEY_ENV` resolved, plus `<prefix><FEATURE>` for the prefix `config.py` scans with `startswith`.
  - `variables(help) -> set[str]`, `features(help) -> set[str]`, `hides(help) -> list[tuple[str, tuple[str, ...]]]`, `removes(help) -> list[tuple[str, tuple[str, ...]]]`, `variable_defaults(help) -> list[tuple[str, str]]` and `default_off(help) -> frozenset[str]`.
- The accepted words are:
  - for a console script: every string its dispatch compares `args` with, including the members of a table it tests `args[0] in`;
  - for a subcommand: every string constant that is exactly an option (`-x` or `--name`) in the command's own function, in the functions it passes `argv` to, and in the module-level tuples those functions read.

- [ ] **Step 1: Write the failing tests on planted commands** defined in the test file, so `inspect.getsource` can read them: a command with an option its help does not name is reported; its `-h` and `help` aliases are not (Review Focus 5); a wrapper is followed to its help; a planted help naming `MEMVARA_NOT_READ` is reported, as are a feature that does not exist and a claim that a feature hides a tool it does not hide.
- [ ] **Step 2: Run them and watch them fail on the missing module.**
- [ ] **Step 3: Write `commandline.py`.**
- [ ] **Step 4: Write the real-data tests:**
  - `test_every_command_line_is_found`: at least the two console scripts and six subcommands, so a reader that found nothing fails;
  - `test_every_option_a_subcommand_accepts_is_in_its_help`, one test per subcommand;
  - `test_every_variable_a_help_names_is_read_by_the_configuration`;
  - `test_every_feature_the_help_names_exists`;
  - `test_what_the_help_says_a_feature_hides_is_hidden`;
  - `test_a_variable_set_to_its_stated_default_changes_nothing`, and the same for each feature with the default the help states;
  - `test_every_word_a_console_script_accepts_is_in_its_help`: run it. If it fails, take it out of the file and report the drift with a reproduction.
- [ ] **Step 5: Run the file and watch the committed tests pass.**
- [ ] **Step 6: Add a paragraph on the command lines to the `testing.md` section.**
- [ ] **Step 7: Commit** `commandline.py`, the test file and `testing.md`.

### Task 7: Verification

- [ ] **Step 1: Run each new test file 20 times in a row** and record the result.
- [ ] **Step 2: Run the full gate as two commands with a private coverage file,** and check that coverage of `memvara/` is still 100%.
- [ ] **Step 3: Run `mypy -p memvara`, and `mypy tests/harness` with and without `--ignore-missing-imports`.** Also type-check `tests/adversarial/docs` with `MYPYPATH=tests`.
- [ ] **Step 4: Measure the fast-tier time of `tests/adversarial/docs`** with `--durations`, against the budget of about 2 seconds.

## What changed during implementation

- **The checks return names, not sentences.** Each check returns the offending names, or `(tool, argument)` pairs, and the test writes the sentence with the configuration or page it came from. The planted tests can then compare against hand-written values instead of message wording.
- **`commandline.py` has a few more public functions** than Task 6 lists: `read_script`, `read_subcommand`, `subcommands` and `reads_in`, and the checks `unread_variables`, `nonexistent_features` and `false_claims`. The planted commands and the planted configuration reader go through the same code as the real ones. `command_lines()` returns a cached tuple rather than a list.
- **The guards against a parser that reads nothing do not count.** `test_every_command_line_is_found` requires every script in `pyproject.toml`, each with at least one subcommand, and `test_the_real_descriptions_state_defaults` requires each kind of statement at least once. A fixed number would fail when a subcommand or a sentence is removed on purpose.
- **Three tests failed on drift and are not committed**, as Global Constraints says: `test_a_tools_own_argument_named_in_its_text_is_served`, `test_a_default_stated_in_words_is_declared_in_the_schema` and `test_every_word_a_console_script_accepts_is_in_its_help`. They went to the maintainer with a reproduction for each. The checks they call are committed and proven on planted faults.
- **The final review found one parser bug**, and it is fixed: a default that ended a sentence, as in "at the default 0.5.", was skipped without a word. `test_a_default_that_ends_a_sentence_is_still_read` pins it.
- **The full gate ran last,** after the review, so its result describes the final code.

## What the code review changed

The pull request's code review found these, and each is fixed with a test that failed first. Where one contradicts a rule earlier in this plan, this section is the current rule.

- **A tie in tool text.** After "and" or a possessive, only a snake_case word ties an argument to a tool. "memory_end's predicate" is the predicate of a fact, not an argument.
- **The options a subcommand accepts** are read in whatever module the function it hands its argument list to lives, not only its own. A hand-over that cannot be followed raises `LookupError` instead of reporting no options.
- **A help's claim about a feature that is off by default** is checked against the server with it switched on and the default server, because no "<feature> off" server exists for it.
- **The pins for #296 and #297** absorb only their exact symptom: the exact `(argument, words)` pairs, and only the `memvara` and `memvara-mcp` scripts.
- **The connection instructions** are checked on every server for an argument its switches removed from every tool it lists.
- **One shared list of served tools**, `surface.every_tool()`, feeds the description and default checks, what never changes within a run is cached, and the planted tool table lives in `tests/adversarial/docs/planted.py`.
