"""Promises the *distribution* makes, as opposed to promises the code makes.

Everything here is a claim printed on the tin — "numpy and nothing else", "the annotations
reach your type checker", "`pip install 'memvara[openai]'` gets you an OpenAI backend" —
and every one of them can break without a single line of library code changing. An extra
gets declared and no adapter is written behind it. A name is added to `__all__` and never
imported. A subpackage is created and the wheel quietly does not ship it. None of that is
visible from inside the package, which is why it needs its own file.

The bug that motivates the extras section is not hypothetical: `memvara[openai]` was
declared in `pyproject.toml` from the first commit and `memvara/llm/openai.py` did not
exist until Phase 5, so for the whole of waves 1–3 installing that extra bought you a
dependency and no adapter.

`.github/workflows/ci.yml`'s `offline` job is the prior art for the dependency half — it
installs the bare package and imports every module — and it is the *stronger* test,
because it checks a real install rather than a source tree. What is here is the part that
can run with no network and no install, so it fails on the developer's machine at the
moment the mistake is made rather than eight minutes later in CI.
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys
import zipfile
from typing import Iterable

import pytest

import memvara

REPO = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = REPO / "memvara"
PYPROJECT = REPO / "pyproject.toml"


# -- reading pyproject.toml ---------------------------------------------------------


def _toml_table(table: str) -> dict[str, str | list[str]]:
    """The `key = "value"` and `key = [...]` entries of one top-level TOML table.

    Hand-parsed rather than handed to `tomllib`, which arrived in 3.11 while
    `requires-python` promises 3.10. A packaging test that skips on one of the four
    interpreters the package claims to support is exactly the kind of half-enforcement
    the CI matrix exists to end. It understands only the shapes `pyproject.toml` actually
    uses — one scalar per key, and an array written on one line or across several — and
    `test_the_hand_parse_of_pyproject_agrees_with_tomllib` pins it against the real parser
    everywhere the real parser exists.

    The multi-line array arrived with `keywords` and `classifiers`, which are fifteen
    entries each and unreadable on one line. Handled by accumulating until the closing
    bracket rather than by a second code path, so the two spellings cannot diverge.
    """
    found: dict[str, str | list[str]] = {}
    inside = False
    key_open: str | None = None
    buffer = ""

    def items(body: str) -> list[str]:
        return [item.strip().strip("\"'") for item in body.split(",") if item.strip()]

    def uncommented(text: str) -> str:
        """`text` with a trailing `# ...` removed, unless the `#` is inside a string.

        Needed once arrays could span lines. Two ways the naive version was wrong, and
        both produce a silently *wrong* table rather than an error: a comment line whose
        prose happens to end in `]` closed the array early, and an inline comment after an
        entry became an entry of its own. Quote-aware because a `#` inside a value is
        data — no entry here contains one today, and a parser that depends on that is a
        parser that breaks on the day one does.
        """
        quote = ""
        for i, ch in enumerate(text):
            if quote:
                if ch == quote:
                    quote = ""
            elif ch in "\"'":
                quote = ch
            elif ch == "#":
                return text[:i]
        return text

    for raw in PYPROJECT.read_text(encoding="utf-8").splitlines():
        line = uncommented(raw).strip()
        if key_open is not None:
            # Inside a multi-line array. Whatever survives comment-stripping is entries;
            # a line that was only a comment leaves nothing, and cannot close the array.
            buffer += line
            if line.endswith("]"):
                found[key_open] = items(buffer.rstrip("]"))
                key_open, buffer = None, ""
            continue
        if line.startswith("["):
            inside = line == f"[{table}]"
            continue
        if not inside or not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if value == "[":
            key_open, buffer = key.strip(), ""
        elif value.startswith("[") and value.endswith("]"):
            found[key.strip()] = items(value[1:-1])
        elif value.startswith(('"', "'")):
            found[key.strip()] = value.strip("\"'")
    return found


EXTRAS = {name: value for name, value in _toml_table("project.optional-dependencies").items()
          if isinstance(value, list)}


# -- what each extra is for ---------------------------------------------------------
#
# An extra falls into exactly one of three kinds, and they fail in different ways, which
# is the whole reason for classifying them rather than listing them.
#
# An **adapter** extra buys code. Installing it must make something work that did not
# work before, and *not* installing it must produce an error that names the extra —
# because the person reading that traceback has no other way to learn that the fix is one
# `pip install` away. `ModuleNotFoundError: No module named 'x'` is not that error.
ADAPTER_EXTRAS = {"anthropic", "openai", "local-embed", "rerank",
                  "langchain", "llama-index", "crewai", "langgraph", "cloud"}
# A **reserved** extra buys nothing yet and says so. `http` names the REST layer's
# dependencies before the REST layer exists. That is defensible — it fixes the dependency
# set publicly before anything depends on it — and it is one letter away from the
# `memvara[openai]` bug, so the test below checks that nothing in the package secretly
# imports these. The day the REST layer lands, that test fails and this line has to move.
RESERVED_EXTRAS = {"http"}
# A **tooling** extra is for working on memvara, not with it. Nothing imports these from
# library code, and nothing should.
TOOLING_EXTRAS = {"dev", "bench"}


def _construct_anthropic_llm() -> None:
    from memvara.llm.anthropic import AnthropicLLM

    AnthropicLLM()


def _construct_openai_llm() -> None:
    from memvara.llm.openai import OpenAILLM

    OpenAILLM()


def _construct_local_embedder() -> None:
    from memvara.embed.local import LocalEmbedder

    LocalEmbedder()


def _construct_cross_encoder_reranker() -> None:
    from memvara.rerank.cross import CrossEncoderReranker

    CrossEncoderReranker()


def _construct_remote_store() -> None:
    from memvara.store.remote import RemoteStore

    RemoteStore(base_url="https://app.memvara.dev", api_key="k")


# The adapters are lazy attributes on their own modules, so naming the class is the
# shortest thing that needs the SDK — there is no constructor to reach without it.
def _resolve_langchain_history() -> None:
    from memvara.integrations import langchain

    langchain.MemvaraChatMessageHistory


def _resolve_llamaindex_block() -> None:
    from memvara.integrations import llamaindex

    llamaindex.MemvaraMemoryBlock


def _resolve_crewai_storage() -> None:
    from memvara.integrations import crewai

    crewai.MemvaraStorage


def _resolve_langgraph_store() -> None:
    from memvara.integrations import langgraph

    langgraph.MemvaraStore


#: extra -> (the module its SDK provides, the shortest call that needs it). The module
#: name is here rather than derived from the requirement because a distribution name and
#: an import name are not the same string — `sentence-transformers` imports as
#: `sentence_transformers`, and no rule turns one into the other reliably.
ADAPTERS = {
    "anthropic": ("anthropic", _construct_anthropic_llm),
    "openai": ("openai", _construct_openai_llm),
    "local-embed": ("sentence_transformers", _construct_local_embedder),
    # Same distribution as `local-embed`, different class behind it. Two extras may name
    # one SDK; what the rule below forbids is an extra with *no* code behind it.
    "rerank": ("sentence_transformers", _construct_cross_encoder_reranker),
    "langchain": ("langchain_core", _resolve_langchain_history),
    "llama-index": ("llama_index", _resolve_llamaindex_block),
    "crewai": ("crewai", _resolve_crewai_storage),
    # The import name is `langgraph`, but the *distribution* that provides
    # `langgraph.store.base` is `langgraph-checkpoint` — the `langgraph` wheel has no
    # `store/` in it at all. Exactly the distribution-name-is-not-import-name trap this
    # mapping exists for.
    "langgraph": ("langgraph", _resolve_langgraph_store),
    "cloud": ("httpx", _construct_remote_store),
}

#: The subset of `ADAPTERS` whose SDK name never appears in an `import` statement,
#: because `memvara.integrations._common.require()` reaches it through
#: `importlib.import_module` of a string. They are deliberately invisible to the static
#: walk below and are covered by the runtime one instead — listing them here keeps that
#: exemption explicit, so a framework that *does* get statically imported one day fails
#: the static test rather than quietly joining the exempt set.
DYNAMIC_SDKS = {"langchain_core", "llama_index", "crewai", "langgraph"}


# -- static import graph ------------------------------------------------------------


def _module_trees() -> list[ast.Module]:
    return [ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
            for p in sorted(PACKAGE.rglob("*.py"))]


def _absolute_imports(nodes: Iterable[ast.AST]) -> set[str]:
    """Top-level package names imported by these statements, relative imports excluded."""
    names: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def _is_type_checking_guard(node: ast.stmt) -> bool:
    """`if TYPE_CHECKING:` (bare or `typing.TYPE_CHECKING`) — the one `if` whose body a
    real interpreter never runs, so a static "what executes on import" walk has to treat
    it differently from every other conditional import.
    """
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return ((isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
            or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"))


def _import_time_imports(body: list[ast.stmt]) -> set[str]:
    """What executes on `import memvara.<module>`.

    Descends into `if`, `try` and `with` — a conditional import at module scope still runs
    at import time — and stops at every `def` and `class`, which is precisely the line the
    optional backends are hiding behind. The one `if` this does not descend into is
    `if TYPE_CHECKING:`: that branch is `False` at runtime by construction (that is the
    entire point of `TYPE_CHECKING`), so an SDK imported there only for annotations —
    `RemoteStore` importing `httpx` under it, exactly like `mypy` needs — never actually
    executes on import and must not be flagged as though it did.
    """
    names = _absolute_imports(body)
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if _is_type_checking_guard(node):
            names |= _import_time_imports(node.orelse)
            continue
        for field in ("body", "orelse", "finalbody"):
            names |= _import_time_imports(getattr(node, field, None) or [])
        for handler in getattr(node, "handlers", None) or []:
            names |= _import_time_imports(handler.body)
    return names


#: Standard library on the versions that have it, absent on the ones that do not, so
#: `sys.stdlib_module_names` disagrees with itself across the support matrix. `tomllib`
#: landed in 3.11 and this package supports 3.10, where it would otherwise be reported as
#: an undeclared third-party SDK — a false positive that says the opposite of the truth,
#: since the reason it is imported lazily is precisely so 3.10 never reaches it.
_STDLIB_SINCE_311 = {"tomllib"}


def _third_party(names: set[str]) -> set[str]:
    known = set(sys.stdlib_module_names) | _STDLIB_SINCE_311
    return {n for n in names if n != "memvara" and n not in known}


# -- the built wheel ----------------------------------------------------------------
#
# `dist/` is gitignored, so these are release-time gates rather than CI gates: they arm
# themselves the moment `python3 -m build --wheel` has run, which is the step
# `docs/RELEASING.md` puts immediately before them.

WHEELS = sorted((REPO / "dist").glob(f"memvara-{memvara.__version__}-*.whl"))
needs_wheel = pytest.mark.skipif(
    not WHEELS, reason=f"no dist/memvara-{memvara.__version__}-*.whl; run python3 -m build --wheel")


def _wheel_names() -> set[str]:
    names: set[str] = set()
    for wheel in WHEELS:
        with zipfile.ZipFile(wheel) as archive:
            names |= set(archive.namelist())
    return names


# -- the typing marker ---------------------------------------------------------------


def test_the_py_typed_marker_sits_in_the_top_level_package_directory() -> None:
    """Without this file every annotation in the library is invisible to a type checker.

    PEP 561 says an installed package's inline types are only honoured when a `py.typed`
    marker sits beside its `__init__.py`; mypy's answer without it is "module is installed,
    but missing library stubs or py.typed marker" and every memvara symbol becomes `Any`.
    The annotations were all already written, so this one empty file is the entire
    difference between a thoroughly annotated library and an untyped one at the call site.

    Top level only, and deliberately not repeated in the subpackages: one marker covers
    `memvara.store`, `memvara.llm` and the rest.
    """
    marker = PACKAGE / "py.typed"
    assert marker.is_file(), f"{marker} is missing; every annotation in memvara stops here"
    assert [p for p in PACKAGE.rglob("py.typed") if p.parent != PACKAGE] == [], (
        "a marker inside a subpackage claims that subpackage is separately distributed, "
        "which none of them are")


def test_the_marker_is_empty_because_the_word_partial_in_it_would_change_its_meaning() -> None:
    """`partial\\n` is the one string PEP 561 reads out of this file, and it means less.

    A marker containing `partial` tells the checker the inline annotations are incomplete
    and that it should keep looking for a stub package to fill the gaps — so a stray line
    of explanatory prose starting with that word would silently downgrade the promise.
    Zero bytes is the only content with no second reading.
    """
    assert (PACKAGE / "py.typed").read_bytes() == b""


@needs_wheel
def test_the_marker_survives_the_trip_into_the_wheel() -> None:
    """A marker in the source tree that the build drops is worth nothing to an installer.

    Hatchling includes it today by virtue of `packages = ["memvara"]` sweeping the whole
    directory, which means the guarantee rests on the marker never matching a VCS ignore
    rule. That is a thin thread to hang the library's entire typing story on, and this is
    the test that notices when it snaps.
    """
    assert "memvara/py.typed" in _wheel_names()


@needs_wheel
def test_the_wheel_carries_every_module_in_the_package() -> None:
    """A new subpackage that the build does not pick up fails as an ImportError on install.

    It cannot fail any earlier: the source tree keeps working for everyone who has the
    repository, and only someone installing the wheel ever sees the missing module.
    """
    expected = {f"memvara/{p.relative_to(PACKAGE).as_posix()}" for p in PACKAGE.rglob("*.py")}
    assert expected <= _wheel_names(), (
        "modules in the tree and not in dist/. Either the build is dropping them or the "
        "wheel predates them — rebuild with `python3 -m build --wheel` and rerun before "
        "reading this as a packaging bug")


# -- the dependency floor ------------------------------------------------------------


def test_the_core_declares_exactly_one_runtime_dependency() -> None:
    """Core requiring numpy and nothing else is a headline claim, not a default.

    Pinned by exact equality rather than by counting, so adding a dependency means
    editing the sentence in the README and this line together instead of one of them.
    """
    assert _toml_table("project")["dependencies"] == ["numpy>=1.24"]


def test_nothing_but_numpy_is_imported_while_the_package_is_being_imported() -> None:
    """`import memvara` must not touch an optional SDK, on any path, in any module.

    Read statically so the answer does not depend on what happens to be installed on the
    machine running the suite: a developer with `openai` in their environment gets the
    same verdict as the empty CI runner. The scan descends into `if` and `try` at module
    scope — a guarded import still executes on import — and stops at `def` and `class`,
    which is where the optional backends are supposed to be.
    """
    at_import_time: set[str] = set()
    for tree in _module_trees():
        at_import_time |= _import_time_imports(tree.body)
    assert _third_party(at_import_time) == {"numpy"}


def test_the_only_sdks_the_package_names_anywhere_are_the_ones_an_extra_installs() -> None:
    """An import of something no extra declares is a dependency nobody agreed to ship.

    This is the check that keeps a reserved extra honest in the other direction too: no
    module imports `fastapi`, `uvicorn` or `pydantic`, so `memvara[http]` really is
    reserved rather than half-wired.
    """
    anywhere: set[str] = set()
    for tree in _module_trees():
        anywhere |= _absolute_imports(ast.walk(tree))
    named = {module for module, _ in ADAPTERS.values()} - DYNAMIC_SDKS
    assert _third_party(anywhere) == {"numpy"} | named, (
        "an SDK named in the source with no extra declaring it, or an extra whose SDK is "
        "no longer imported. Note that a framework reached through "
        "`memvara.integrations._common.require()` is invisible here — it is an "
        "`importlib.import_module` of a string — so those are covered by the runtime "
        "walk below instead.")


def test_every_module_imports_cleanly_in_a_process_that_has_only_numpy() -> None:
    """The runtime half of the check above, and it catches what a source scan cannot.

    A module-level `importlib.import_module(...)`, a decorator that reaches for an SDK, a
    PEP 562 `__getattr__` that fires during import of something else — none of those are
    an `Import` node, and all of them break the offline install. A subprocess because the
    question is what a *fresh* interpreter does, and by the time this file runs pytest has
    already imported half the package.
    """
    probe = (
        "import importlib, pkgutil, sys, memvara\n"
        "bad = []\n"
        "for m in pkgutil.walk_packages(memvara.__path__, 'memvara.'):\n"
        "    try:\n"
        "        importlib.import_module(m.name)\n"
        "    except ImportError as exc:\n"
        "        bad.append((m.name, str(exc)))\n"
        "leaked = sorted({'anthropic', 'openai', 'sentence_transformers', 'fastapi',\n"
        "                 'uvicorn', 'pydantic'} & set(sys.modules))\n"
        "print(bad or '', leaked or '', sep='|')\n"
    )
    done = subprocess.run([sys.executable, "-c", probe], cwd=REPO, check=False,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    failed, leaked = done.stdout.strip().split("|")
    assert not failed, f"modules that would not import on a core install: {failed}"
    assert not leaked, f"importing memvara pulled in an optional SDK: {leaked}"


# -- extras and the adapters behind them ----------------------------------------------


def test_every_declared_extra_is_classified_so_a_new_one_cannot_ship_unexamined() -> None:
    """Declaring an extra is a promise; this is the place the promise gets made explicitly.

    `memvara[openai]` shipped for three waves as a dependency with no adapter behind it
    precisely because nothing anywhere had to say what the extra was *for*. Adding one to
    `pyproject.toml` now fails the suite until someone writes down which of the three
    kinds it is.
    """
    assert set(EXTRAS) == ADAPTER_EXTRAS | RESERVED_EXTRAS | TOOLING_EXTRAS, (
        "a new extra in pyproject.toml: add it to ADAPTER_EXTRAS (and to ADAPTERS, with "
        "the module and the shortest call that needs it), RESERVED_EXTRAS, or "
        "TOOLING_EXTRAS above. Failing here is the point — it is the moment someone has "
        "to say what the extra buys.")


def test_every_adapter_extra_has_a_call_that_actually_needs_its_sdk() -> None:
    """The `memvara[openai]` bug, stated as a rule: an adapter extra must have an adapter."""
    assert set(ADAPTERS) == ADAPTER_EXTRAS


@pytest.mark.parametrize("extra", [
    "anthropic",
    "openai",
    "local-embed",
    "rerank",
])
def test_an_adapter_whose_sdk_is_absent_raises_an_error_naming_the_extra(
        extra: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing optional dependency has to say which `pip install` fixes it.

    The audience for this message is someone who has never read `pyproject.toml` and has
    no reason to guess that the extra is spelled `local-embed` rather than
    `sentence-transformers`. A bare `ModuleNotFoundError` sends them to the SDK's own
    install page, where they install it globally and never learn the extra exists.

    `None` in `sys.modules` is CPython's own "this import is blocked" sentinel, used here
    so the test asserts the same thing whether or not the SDK happens to be installed on
    the machine running it.
    """
    module, construct = ADAPTERS[extra]
    monkeypatch.setitem(sys.modules, module, None)
    with pytest.raises(ImportError) as caught:
        construct()
    assert f"memvara[{extra}]" in str(caught.value)


def test_the_version_is_the_same_string_in_both_places_that_state_it() -> None:
    """`pyproject.toml` names the version for the installer, `__init__.py` for the program.

    Nothing keeps them equal. When they drift, `pip show memvara` and
    `memvara-mcp --version` disagree, and the bug report you get back quotes the one that
    is wrong. The two-place bump is the first line of `docs/RELEASING.md` for this reason.
    """
    assert _toml_table("project")["version"] == memvara.__version__


@needs_wheel
def test_the_built_wheel_is_the_version_the_package_reports() -> None:
    """A wheel built before the version bump installs as the old release under a new name.

    Only reachable once something has been built, which is exactly when it matters: this
    is the gate between `python3 -m build` and `twine upload`.
    """
    for wheel in WHEELS:
        with zipfile.ZipFile(wheel) as archive:
            metadata = archive.read(f"memvara-{memvara.__version__}.dist-info/METADATA")
        assert f"Version: {memvara.__version__}".encode() in metadata


def test_the_console_script_points_at_something_importable() -> None:
    """`memvara-mcp` is generated by the installer and never executed by the test suite.

    So a typo in the entry point survives every test in this repository and surfaces as an
    ImportError on the user's first launch — from a wrapper script they did not write and
    cannot easily read.
    """
    scripts = _toml_table("project.scripts")
    target = scripts["memvara-mcp"]
    assert isinstance(target, str)
    module, _, attribute = target.partition(":")
    imported = __import__(module, fromlist=[attribute])
    assert callable(getattr(imported, attribute))


# -- the public surface ----------------------------------------------------------------


def _lazy_exports() -> set[str]:
    """Names `memvara.__getattr__` will hand out, read from its source.

    Read rather than listed because a fourth backend added to that function and forgotten
    here would make the test below vacuous, which is the failure mode of every hardcoded
    inventory.

    Both spellings the hook uses are read: `name == "X"` and `name in ("X", "Y")`. The
    remote errors arrive eleven at a time and are written as the second, and a parser that
    only understood the first would have gone quiet about all eleven while still passing.
    """
    tree = ast.parse((PACKAGE / "__init__.py").read_text(encoding="utf-8"))
    getattr_fn = next(n for n in tree.body
                      if isinstance(n, ast.FunctionDef) and n.name == "__getattr__")
    compared = [c for node in ast.walk(getattr_fn) if isinstance(node, ast.Compare)
                for c in node.comparators]
    return {n.value for c in compared for n in ast.walk(c)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}


def test_every_name_in_dunder_all_can_actually_be_imported() -> None:
    """`__all__` is what `from memvara import *` binds, and it is not checked by anything.

    A name listed here and not exported is an `AttributeError` at the star-import, which
    no test in this repository would otherwise reach — the rest of the suite imports the
    symbols it needs by name.
    """
    missing = [name for name in memvara.__all__ if not hasattr(memvara, name)]
    assert not missing


def test_dunder_all_lists_nothing_twice() -> None:
    """Duplicates are how two people adding an export to the same list both succeed."""
    assert len(memvara.__all__) == len(set(memvara.__all__))


def test_a_star_import_of_memvara_needs_no_optional_sdk() -> None:
    """`from memvara import *` resolves every lazy export, including the hosted backends.

    That makes `__all__` the one place where adding a lazily-imported name can break the
    core install: the star import calls `__getattr__` for it, and if that adapter imports
    its SDK at module scope rather than inside a function, the two-package install stops
    being able to `import *` at all. In a subprocess because a star import into this
    module's namespace would be a mess to undo.
    """
    done = subprocess.run(
        [sys.executable, "-c", "exec('from memvara import *')\nprint('ok')"],
        cwd=REPO, check=False, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


def test_every_eagerly_imported_public_name_is_exported() -> None:
    """A symbol imported into `memvara/__init__.py` and left out of `__all__` is a trap.

    It works for anyone who writes `from memvara import X`, is absent for anyone who writes
    `from memvara import *`, and the difference only shows up in someone else's code.
    """
    tree = ast.parse((PACKAGE / "__init__.py").read_text(encoding="utf-8"))
    eager = {alias.asname or alias.name
             for node in tree.body if isinstance(node, ast.ImportFrom)
             for alias in node.names}
    assert eager <= set(memvara.__all__)


def test_every_lazily_exported_backend_is_listed_in_dunder_all() -> None:
    """Two backends behind one `__getattr__` should not disagree about being public.

    `__all__` is the inventory a reader consults and the list `from memvara import *`
    obeys, so a backend missing from it is documented nowhere and star-imports to nothing,
    while its sibling does both.
    """
    assert _lazy_exports() <= set(memvara.__all__)


# -- the parser this file leans on -----------------------------------------------------


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib arrived in 3.11")
def test_the_hand_parse_agrees_with_tomllib_on_comment_shapes_this_file_lacks(
        tmp_path: pathlib.Path) -> None:
    """The agreement test above reads `pyproject.toml`, which contains none of these.

    So the quote-aware comment stripping the multi-line array reader depends on was
    never exercised by it, and is skipped entirely on 3.10 — the one interpreter the hand
    parser exists to serve. Three shapes, each of which the first version got wrong and
    each of which produces a silently *wrong* table rather than an error: a comment line
    whose prose ends in a bracket, an inline comment after an entry, and a `#` inside a
    string, which must survive.
    """
    import tomllib

    probe = tmp_path / "pyproject.toml"
    probe.write_text(
        '[project]\n'
        'name = "probe"\n'
        'keywords = [\n'
        '    # a comment whose prose ends in a bracket [like this]\n'
        '    "alpha",\n'
        '    "beta",   # an inline note\n'
        '    "gam#ma",\n'
        ']\n'
        'description = "has a # inside a string"\n', encoding="utf-8")

    global PYPROJECT
    original, PYPROJECT = PYPROJECT, probe
    try:
        got = _toml_table("project")
    finally:
        PYPROJECT = original

    want = {k: v for k, v in tomllib.loads(probe.read_text(encoding="utf-8"))["project"].items()
            if isinstance(v, (str, list))}
    assert got == want


@pytest.mark.skipif(sys.version_info < (3, 11), reason="tomllib arrived in 3.11")
def test_the_hand_parse_of_pyproject_agrees_with_tomllib() -> None:
    """The reader above is only trustworthy if something checks it against a real parser.

    Every other test in this file reads `pyproject.toml` through it, so a reader that
    quietly returned `{}` would turn each of them into an assertion about nothing.
    """
    import tomllib

    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    for table, expected in (
        ("project.optional-dependencies", data["project"]["optional-dependencies"]),
        ("project.scripts", data["project"]["scripts"]),
        ("project", {k: v for k, v in data["project"].items()
                     if isinstance(v, (str, list))}),
    ):
        assert _toml_table(table) == expected
    assert EXTRAS, "an empty extras table would make the classification tests vacuous"
