"""Tests for the plugin-recall harness.

Every guard here was sabotaged before it was believed: the corresponding defect was
introduced, the test was watched going red, and only then was the defect removed. That is
the house rule, and it earned its place on this harness in particular -- its first live run
reported a perfect score for a plugin that was answering `recall failed` to every prompt.
"""

from __future__ import annotations

import json
import os
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.plugin_recall.cases import CaseError, load
from benchmarks.plugin_recall.plugin import PluginError, discover, invoke
from benchmarks.plugin_recall.report import render, summary
from benchmarks.plugin_recall.runner import run
from benchmarks.plugin_recall.seed import emit_cases, facts

#: A plugin is a directory with `hooks/hooks.json` and a command. Nothing else is required,
#: which is the point of the contract -- so the fake is a shell script.
HOOK = """#!/bin/sh
cat > /dev/null
{body}
"""


def make_plugin(root: Path, body: str, *, name: str = "fake", version: str = "1.0.0") -> Path:
    (root / "hooks").mkdir(parents=True, exist_ok=True)
    (root / ".claude-plugin").mkdir(parents=True, exist_ok=True)
    (root / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({"name": name, "version": version}))
    script = root / "hooks" / "hook.sh"
    script.write_text(HOOK.format(body=body))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    (root / "hooks" / "hooks.json").write_text(json.dumps({
        "hooks": {"UserPromptSubmit": [
            {"hooks": [{"type": "command",
                        "command": 'sh "${CLAUDE_PLUGIN_ROOT}/hooks/hook.sh"',
                        "timeout": 10}]}]}}))
    return root


SPEAKS = ('printf \'{"hookSpecificOutput":{"hookEventName":"UserPromptSubmit",'
          '"additionalContext":"- Marco owns the billing service"}}\'')
SILENT = 'printf \'{"continue":true}\''
BROKEN = 'printf \'{"systemMessage":"recall failed"}\''


class DiscoveryTests(unittest.TestCase):
    def test_a_bare_name_is_never_a_relative_directory(self):
        """The first live run resolved `--plugin memvara` to a library package that
        happened to sit in the working directory. It failed loudly only because that
        package has no `hooks/`; a directory that did have one would have been graded
        silently as the wrong software."""
        with TemporaryDirectory() as tmp:
            make_plugin(Path(tmp) / "ghost", SILENT)
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with self.assertRaises(PluginError):
                    discover("ghost")
            finally:
                os.chdir(cwd)

    def test_a_plugin_with_no_prompt_hook_is_refused_not_scored(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "p"
            (root / "hooks").mkdir(parents=True)
            (root / "hooks" / "hooks.json").write_text(json.dumps({"hooks": {}}))
            with self.assertRaises(PluginError) as caught:
                discover(str(root))
            self.assertIn("no UserPromptSubmit", str(caught.exception).replace(
                "declares no UserPromptSubmit", "no UserPromptSubmit"))

    def test_two_prompt_hooks_are_refused_rather_than_half_scored(self):
        with TemporaryDirectory() as tmp:
            root = make_plugin(Path(tmp) / "p", SILENT)
            declared = json.loads((root / "hooks" / "hooks.json").read_text())
            entry = declared["hooks"]["UserPromptSubmit"][0]["hooks"][0]
            declared["hooks"]["UserPromptSubmit"][0]["hooks"].append(dict(entry))
            (root / "hooks" / "hooks.json").write_text(json.dumps(declared))
            with self.assertRaises(PluginError):
                discover(str(root))


class ScoringTests(unittest.TestCase):
    def _run(self, body, cases_path):
        with TemporaryDirectory() as tmp:
            root = make_plugin(Path(tmp) / "p", body)
            return run(discover(str(root)), load(cases_path), cwd=Path(tmp))

    def test_a_broken_plugin_does_not_score_a_perfect_silence_run(self):
        """The degenerate case, and the only reason this harness has a `validated` flag.

        A plugin that injects nothing is indistinguishable from one with nothing to say, so
        a silence-only corpus hands a dead hook 100%. Withholding the score is the whole
        guard: delete `validated` from `Result.rate` and this test goes green at 100%,
        which is exactly the number that must never be printed."""
        from benchmarks.plugin_recall.cases import DEFAULT_CASES

        result = self._run(BROKEN, DEFAULT_CASES)
        self.assertFalse(result.validated)
        self.assertIsNone(result.rate("silence"))
        self.assertIn("UNVALIDATED", render(result))

    def test_a_plugin_that_speaks_is_scored_normally(self):
        from benchmarks.plugin_recall.cases import DEFAULT_CASES

        result = self._run(SPEAKS, DEFAULT_CASES)
        self.assertTrue(result.validated)
        # It injects on every prompt, and every prompt here is one it should have stayed
        # quiet on.
        self.assertEqual(result.rate("silence"), 0.0)

    def test_a_broken_plugin_is_not_given_a_zero_hit_rate_either(self):
        """The guard covered silence and not hits, so the same run that withheld the
        silence score as UNVALIDATED still printed `hit rate 0.0%` -- a verdict on
        retrieval quality, published about software that never answered anything."""
        with TemporaryDirectory() as tmp:
            hits = Path(tmp) / "hits.jsonl"
            hits.write_text(json.dumps({
                "id": "h-1", "kind": "hit", "prompt": "who owns billing?",
                "why": "seeded", "expect": ["Marco"]}) + "\n")
            root = make_plugin(Path(tmp) / "p", BROKEN)
            result = run(discover(str(root)), load(hits), cwd=Path(tmp))
            self.assertFalse(result.validated)
            self.assertIsNone(result.rate("hit"))
            self.assertNotIn("0.0%", render(result))

    def test_hit_cases_are_unavailable_rather_than_zero_when_none_are_loaded(self):
        """`None`, never `0.0`. A plugin that was never asked a question has not failed
        it, and a report that prints 0% publishes an accusation it did not test."""
        from benchmarks.plugin_recall.cases import DEFAULT_CASES

        result = self._run(SPEAKS, DEFAULT_CASES)
        self.assertIsNone(result.rate("hit"))
        self.assertIsNone(result.balanced)
        self.assertIn("unavailable", render(result))


class CorpusTests(unittest.TestCase):
    def _write(self, tmp, rows):
        path = Path(tmp) / "cases.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return path

    def test_a_hit_case_without_patterns_is_refused(self):
        """Such a case can never fail, so it inflates the denominator with a free point."""
        with TemporaryDirectory() as tmp:
            path = self._write(tmp, [{"id": "x", "kind": "hit", "prompt": "p", "why": "w"}])
            with self.assertRaises(CaseError):
                load(path)

    def test_a_silence_case_with_patterns_is_refused(self):
        with TemporaryDirectory() as tmp:
            path = self._write(tmp, [{"id": "x", "kind": "silence", "prompt": "p",
                                      "why": "w", "expect": ["a"]}])
            with self.assertRaises(CaseError):
                load(path)

    def test_duplicate_ids_across_files_are_refused(self):
        """A private corpus and a public one colliding is how a published silence number
        quietly starts including somebody's private case."""
        with TemporaryDirectory() as tmp:
            row = {"id": "dup", "kind": "silence", "prompt": "p", "why": "w"}
            a, b = Path(tmp) / "a.jsonl", Path(tmp) / "b.jsonl"
            a.write_text(json.dumps(row) + "\n")
            b.write_text(json.dumps(row) + "\n")
            with self.assertRaises(CaseError):
                load(a, b)

    def test_the_shipped_corpus_loads_and_is_all_silence(self):
        from benchmarks.plugin_recall.cases import DEFAULT_CASES

        cases = load(DEFAULT_CASES)
        self.assertTrue(cases)
        self.assertTrue(all(c.kind == "silence" for c in cases))


class SeedTests(unittest.TestCase):
    def test_no_gold_token_appears_in_its_own_question(self):
        """Otherwise a plugin scores by echoing the prompt back, and the benchmark
        measures nothing at all."""
        import re

        for fact in facts():
            for pattern in fact["expect"]:
                self.assertIsNone(
                    re.search(pattern, fact["question"], re.I),
                    f"{fact['id']}: gold {pattern!r} is already in its own question")

    def test_emitted_hit_cases_match_the_committed_file(self):
        """The committed corpus is a generated artefact, so a fact edited without
        regenerating is a diff someone has to see rather than drift nobody notices."""
        from benchmarks.plugin_recall.cases import DEFAULT_CASES

        committed = (DEFAULT_CASES.parent / "v1_hits.jsonl").read_text()
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "v1_hits.jsonl"
            emit_cases(out)
            self.assertEqual(out.read_text(), committed)


class EnvTests(unittest.TestCase):
    def test_extra_env_reaches_the_hook(self):
        """This is how a plugin is pointed at a benchmark store instead of a real one. If
        it silently did not arrive, every run would grade the operator's production
        memory -- which happened, and cost a 0% hit rate against a store never read."""
        body = 'printf \'{"hookSpecificOutput":{"additionalContext":"%s"}}\' "$BENCH_MARKER"'
        with TemporaryDirectory() as tmp:
            root = make_plugin(Path(tmp) / "p", body)
            reply = invoke(discover(str(root)), "q", session_id="s", cwd=Path(tmp),
                           extra_env={"BENCH_MARKER": "reached"})
            self.assertEqual(reply.context.strip(), "reached")

    def test_env_values_are_not_copied_into_the_report(self):
        """A report is something people paste into issues, and a value here can be a path
        or a token. Only names are kept."""
        with TemporaryDirectory() as tmp:
            root = make_plugin(Path(tmp) / "p", SPEAKS)
            from benchmarks.plugin_recall.cases import DEFAULT_CASES

            result = run(discover(str(root)), load(DEFAULT_CASES), cwd=Path(tmp),
                         extra_env={"SECRET_TOKEN": "hunter2"})
            self.assertIn("SECRET_TOKEN", summary(result)["env_overrides"])
            self.assertNotIn("hunter2", render(result))
            self.assertNotIn("hunter2", json.dumps(summary(result)))


if __name__ == "__main__":
    unittest.main()
