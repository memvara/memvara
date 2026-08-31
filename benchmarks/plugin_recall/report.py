"""Rendering a result, including the parts that are unflattering to report.

Every figure a run can produce is printed with the count it was computed from. A rate over
three cases and a rate over three hundred are not the same claim, and a table that hides
the denominator invites them to be quoted as though they were.
"""

from __future__ import annotations

import json
from statistics import median

from . import __version__
from .runner import Result


def _percentile(values: list[float], share: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * share))]


def _rate(value: float | None, n: int, unavailable: str) -> str:
    return unavailable if value is None else f"{value * 100:5.1f}%  (n={n})"


def summary(result: Result) -> dict:
    scored = [o for o in result.outcomes]
    tokens = [float(o.reply.tokens) for o in scored]
    latency = [o.reply.elapsed_ms for o in scored]
    return {
        "harness_version": __version__,
        "plugin": result.plugin.label(),
        "plugin_root": str(result.plugin.root),
        "command": " ".join(result.plugin.argv),
        "cwd": str(result.cwd),
        "env_overrides": list(result.env),
        "cases": len(scored),
        "shared_session": result.shared_session,
        "validated": result.validated,
        "hit_rate": result.rate("hit"),
        "hit_cases": len(result.of_kind("hit")),
        "silence_rate": result.rate("silence"),
        "silence_cases": len(result.of_kind("silence")),
        "balanced": result.balanced,
        "session_preamble_tokens": result.preamble.tokens,
        "tokens_mean": round(sum(tokens) / len(tokens), 1) if tokens else 0.0,
        "tokens_median": round(median(tokens), 1) if tokens else 0.0,
        "tokens_p90": round(_percentile(tokens, 0.9), 1),
        "latency_median_ms": round(median(latency), 1) if latency else 0.0,
        "latency_p90_ms": round(_percentile(latency, 0.9), 1),
        "hook_errors": sum(1 for o in scored if o.reply.error),
        "prompt_chars": {o.case.id: len(o.case.prompt) for o in scored},
    }


def render(result: Result, *, verbose: bool = False) -> str:
    s = summary(result)
    lines = [
        f"Plugin Recall Benchmark v{s['harness_version']}",
        f"  plugin    {s['plugin']}",
        f"  root      {s['plugin_root']}",
        f"  command   {s['command']}",
        f"  cwd       {s['cwd']}",
        f"  env       {', '.join(s['env_overrides']) or '(none)'}",
        f"  sessions  {'one shared (marginal-cost mode)' if s['shared_session'] else 'one per case'}",
        "",
        "  hit rate       "
        + _rate(s["hit_rate"], s["hit_cases"],
                "unavailable -- no hit cases loaded; see cases/build_private.py"),
        "  silence rate   " + _rate(
            s["silence_rate"], s["silence_cases"],
            "UNVALIDATED -- the plugin injected nothing on any prompt, including the\n"
            "                 warmup. A broken hook scores 100% on a silence corpus, so\n"
            "                 this is withheld rather than reported as a pass."),
        "  balanced       "
        + _rate(s["balanced"], s["cases"],
                "unavailable -- needs both populations, and is the only combined"
                " figure an always-inject plugin cannot win"),
        "",
        f"  injected tokens   median {s['tokens_median']:.0f}   mean {s['tokens_mean']:.0f}"
        f"   p90 {s['tokens_p90']:.0f}",
        f"  session preamble  {s['session_preamble_tokens']} tok, paid once per session",
        f"  hook latency      median {s['latency_median_ms']:.0f}ms   p90 {s['latency_p90_ms']:.0f}ms",
    ]
    if s["hook_errors"]:
        lines.append(
            f"  hook errors       {s['hook_errors']} of {s['cases']} -- a hook that fails "
            "says nothing, which scores as silence; read the per-case listing")

    families: dict[str, list[bool]] = {}
    for outcome in result.of_kind("silence"):
        families.setdefault(outcome.case.family, []).append(outcome.correct)
    if len(families) > 1 and s["validated"]:
        # Worth its four lines: a plugin can score well on silence without judging
        # relevance at all. supermemory skips any prompt under twelve characters, which
        # passes every bare acknowledgement here and no lexical trap. One blended number
        # hides that; the split says which mechanism earned the score.
        lines += ["", "  silence by family:"]
        for family, marks in sorted(families.items()):
            passed = sum(1 for m in marks if m)
            lines.append(f"    {family:<8} {passed}/{len(marks)}")

    if verbose:
        lines += ["", "  per case:"]
        for outcome in result.outcomes:
            mark = "PASS" if outcome.correct else "FAIL"
            lines.append(f"    {mark}  {outcome.case.kind:<7} {outcome.case.id:<18} "
                         f"{outcome.note}")
            if not outcome.correct and outcome.reply.spoke:
                first = outcome.reply.context.strip().splitlines()[0][:96]
                lines.append(f"          injected: {first}")
            if outcome.reply.system_message:
                # Never scored -- it does not reach the model -- but it is where a plugin
                # says "recall failed", and a reader looking at an unvalidated run needs
                # to see that sentence to know what they are looking at.
                lines.append(f"          status:   {outcome.reply.system_message[:96]}")
    return "\n".join(lines)


def as_json(result: Result) -> str:
    body = summary(result)
    body["outcomes"] = [
        {"id": o.case.id, "kind": o.case.kind, "correct": o.correct,
         "tokens": o.reply.tokens, "ms": round(o.reply.elapsed_ms, 1),
         "note": o.note, "error": o.reply.error}
        for o in result.outcomes
    ]
    return json.dumps(body, indent=2)
