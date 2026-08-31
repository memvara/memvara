"""A `Recorder` that survives the process, for the tokens capture spends.

The library defines where these numbers go: `write.tokens_in` and `write.tokens_out`,
whose catalogue entry says outright that this is the series to bill on, because providers
charge per token and the ratio between tokens and call-count is unbounded. So the names
are imported rather than invented — a hook that emitted `capture.tokens` would produce a
series nothing else in the system knows how to read.

What the library does not provide is somewhere durable to put them. `MemoryRecorder` keeps
everything in a dict and says so: correct for a test, wrong for a process that runs at the
end of every turn forever. This appends one JSON object per emission instead, which is the
smallest thing that is still queryable next month.

Emission here is deliberately *not* fire-and-forget in the library's sense. Telemetry that
raises on a write path would turn a measurement into a failed capture, and capture failing
because its accounting failed is the wrong trade — the whole module swallows `OSError` and
carries on unmeasured.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

#: One JSON object per line, beside the store rather than inside it. The store is the
#: user's memory and answers questions about them; this is operational accounting about
#: the tool, and mixing the two would put "we spent 4,897 tokens" into recall results.
DEFAULT_PATH = Path.home() / ".memvara" / ".hooks" / "usage.jsonl"

#: Truncated rather than rotated, for the same reason capture.log is: a debugging and
#: accounting aid that needs its own maintenance job is worse than a smaller one.
MAX_BYTES = 2 * 1024 * 1024


def _names() -> "tuple[str, str]":
    """The library's own series names, or the literals if it cannot be imported.

    The fallback is not laziness. This module is imported by a hook that must keep working
    when the library is missing, and the strings are the stable public contract described
    in the metric catalogue — pinning them here costs nothing and keeps the hook honest
    about which series it writes.
    """
    try:
        from memvara.telemetry import WRITE_TOKENS_IN, WRITE_TOKENS_OUT

        return WRITE_TOKENS_IN, WRITE_TOKENS_OUT
    except Exception:
        return "write.tokens_in", "write.tokens_out"


TOKENS_IN, TOKENS_OUT = _names()


class JsonlRecorder:
    """Satisfies `memvara.telemetry.Recorder`: counter, gauge, timing.

    Implementing the full protocol rather than a single `record_tokens()` helper is what
    lets this be handed to the library later without rewriting the call sites — the
    library's own emission points already speak exactly these three methods.
    """

    def __init__(self, path: "Path | None" = None) -> None:
        self.path = Path(path) if path else DEFAULT_PATH

    def _write(self, kind: str, name: str, value: float, tags: "dict[str, str]") -> None:
        # Tags nest rather than merge into the envelope. Flattened, a tag named `kind`
        # silently overwrote the counter/gauge/timing field of the same name — which is
        # what the cache-read tag did. The row still parsed as valid JSON; the metric type
        # was just quietly gone. Nested, a caller may use any tag name it likes and none
        # of them can reach an envelope field.
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kind": kind,
            "metric": name,
            "value": value,
            "tags": dict(tags),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.path.exists() and self.path.stat().st_size > MAX_BYTES:
                self.path.write_text("")
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError:
            # Unmeasured is survivable. A failed capture because accounting failed is not.
            pass

    # -- the Recorder protocol -------------------------------------------------

    def counter(self, name: str, value: int = 1, /, **tags: str) -> None:
        self._write("counter", name, int(value), tags)

    def gauge(self, name: str, value: float, /, **tags: str) -> None:
        self._write("gauge", name, float(value), tags)

    def timing(self, name: str, ms: float, /, **tags: str) -> None:
        self._write("timing", name, float(ms), tags)


def record_extraction(usage: "dict", *, model: str, recorder: "JsonlRecorder | None" = None) -> None:
    """Emit one headless extraction's cost.

    Cache reads and cache writes are counted as input, and tagged so they stay separable.
    They are the overwhelming majority of what a headless run spends — a measured
    two-sentence extraction was 10 real input tokens against 16,294 cached and 4,897
    written — so a figure that quietly omitted them would understate the true cost by
    three orders of magnitude and make the batching threshold look unnecessary.
    """
    rec = recorder or JsonlRecorder()

    fresh = int(usage.get("input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_write = int(usage.get("cache_creation_input_tokens") or 0)
    out = int(usage.get("output_tokens") or 0)

    if fresh:
        rec.counter(TOKENS_IN, fresh, source="capture", model=model, kind="fresh")
    if cache_read:
        rec.counter(TOKENS_IN, cache_read, source="capture", model=model, kind="cache_read")
    if cache_write:
        rec.counter(TOKENS_IN, cache_write, source="capture", model=model, kind="cache_write")
    if out:
        rec.counter(TOKENS_OUT, out, source="capture", model=model)


def totals(path: "Path | None" = None) -> "dict[str, int]":
    """Sum the ledger. Cheap enough to call from a status command."""
    target = Path(path) if path else DEFAULT_PATH
    out: "dict[str, int]" = {}
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        metric = str(record.get("metric") or "")
        if metric:
            out[metric] = out.get(metric, 0) + int(record.get("value") or 0)
    return out


if __name__ == "__main__":  # `python3 lib/usage.py` prints the ledger
    for metric, value in sorted(totals().items()):
        print(f"{metric:20s} {value:>12,}")
