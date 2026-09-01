"""Build dataset v2. Run it to regenerate the three files beside this one.

    PYTHONPATH=. python3 benchmarks/agent_memory/datasets/build_v2.py

**v2 is v1 plus new material, and the addition is literal.** This script reads the
committed `v1/*.jsonl` and appends to them; it never regenerates v1 and cannot perturb
it. Every v1 question keeps its id, its wording and its gold answer, so a v1 result and a
v2 result can be compared question by question even though the totals cannot be compared
directly — v2 asks more questions, and `benchmarks/agent_memory/README.md` says which.

## What v2 adds, and why each addition exists

v1 measured two of its own dimensions and found that neither separated anything:

1. **`irrelevance` was a three-way tie at 50%.** Six negative questions, of which three
   named a fact slot outright — every system saw an empty slot and abstained — and three
   were open, where every system answered from the nearest match. Easy and impossible,
   with nothing in between, so the dimension ranked nobody. v2 adds two bands that sit
   between: a slot that exists but held nothing *at the instant asked*, and a slot that
   exists but about which the record knew nothing *as of the belief instant asked*. Both
   are questions a time-aware store can decline and a time-blind one cannot.

2. **`multi_hop` was six questions over a store with almost no graph in it.**
   `Memvara.connectivity()` on the v1 corpus reports 3 joinable claims out of 193 — 1.6%
   — and memvara's own `docs/BENCHMARKS.md` says a walk cannot pay for itself at that
   rate. Six questions over three edges measures the wording of the six questions. v2
   adds the connective layer the entities always implied: teams have leads, leads are
   people with cities and languages and employers, employers have head offices. The
   chains that were two hops long now run to four.

Nothing here is added to make a particular system look better. The connective layer makes
every chained question *harder* to retrieve, because `team_lead` goes from one claim in
the store to seven, and the negatives added are ones every system is free to get right.

## Golds are authored here and checked somewhere else

Every gold below is written out by hand, exactly as in v1, and none is read back from
`timeline.Truth`. The suite then derives them independently and asserts the two agree —
including the chained questions, which carry no probe and so were checked by nothing until
v2 added `CHAINS` to `tests/test_agent_memory_bench.py`, and the open negatives, which
name no slot and are checked through `OPEN_NEGATIVES` in the same file. Two derivations
that must match is the only defence against a self-consistent set of wrong answers that
every system fails identically and nobody questions.

## Everything here is synthetic

Invented people, invented companies, invented services, exactly as in v1. Nothing came
from a real user, a real conversation or a real system.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "v1"
OUT = HERE / "v2"

#: v2 inherits v1's instant. Moving it would silently change the answer to every
#: present-tense question in the inherited half, which is the one thing "v2 is v1 plus"
#: must not do.
EVALUATED_AT = "2026-08-01T00:00:00+00:00"

DESCRIPTION = (
    "Agent Memory Benchmark v2. Every v1 scenario, question and gold answer unchanged, "
    "plus an organisation graph that gives chained questions something to walk, and "
    "negative questions in four bands: an absent slot, a slot empty at the instant "
    "asked, a slot unknown as of the belief instant asked, and an open question about "
    "something the store never held."
)

_events: list[dict[str, Any]] = []
_questions: list[dict[str, Any]] = []


def _iso(day: str) -> str:
    return datetime.fromisoformat(day).replace(tzinfo=timezone.utc).isoformat() \
        if len(day) == 10 else day


def ev(scenario: str, eid: str, subject: str, predicate: str, obj: str, *,
       valid_from: str, recorded_at: str | None = None, source: str,
       text: str, confidence: float = 1.0) -> None:
    _events.append({
        "id": eid, "scenario": scenario,
        "recorded_at": _iso(recorded_at or valid_from), "valid_from": _iso(valid_from),
        "subject": subject, "predicate": predicate, "object": obj,
        "text": text, "source": source, "confidence": confidence,
    })


def q(scenario: str, qid: str, category: str, question: str, gold: dict[str, Any], *,
      probe: tuple[str, str] | None = None, at: str | None = None,
      known_at: str | None = None, about: str | None = None, note: str = "") -> None:
    row: dict[str, Any] = {"id": qid, "scenario": scenario, "category": category,
                           "question": question}
    if probe:
        row["probe"] = list(probe)
    if at:
        row["at"] = _iso(at)
    if known_at:
        row["known_at"] = _iso(known_at)
    if about:
        row["about"] = about
    row["gold"] = gold
    if note:
        row["note"] = note
    _questions.append(row)


def value(v: str, *aliases: str) -> dict[str, Any]:
    row: dict[str, Any] = {"kind": "value", "value": v}
    if aliases:
        row["aliases"] = list(aliases)
    return row


def values(*v: str) -> dict[str, Any]:
    return {"kind": "set", "values": list(v)}


NOTHING: dict[str, Any] = {"kind": "none"}


# ---------------------------------------------------------------------------
# org_chart: the connective layer. Teams get leads, leads get lives.
#
# Every team in v1 owns services and none but team-payments had a lead, so a question
# that walked from a service to a person ran out of graph after one hop. These are the
# edges that were always implied by the entities and never written down.
# ---------------------------------------------------------------------------
S = "org_chart"

#: `team -> (lead, city, employer, editor, two languages)`. team-payments is absent: it
#: has a lead in v1, and one that changes, and overwriting it here would rewrite a v1
#: scenario rather than extend it.
LEADS: dict[str, tuple[str, str, str, str, tuple[str, str]]] = {
    "team-core": ("Nora Vlasic", "Zagreb", "Globex", "Neovim", ("English", "Croatian")),
    "team-growth": ("Tomas Ek", "Gothenburg", "Globex", "VS Code", ("English", "Swedish")),
    "team-infra": ("Ada Nwosu", "Lagos", "Acme Corp", "Emacs", ("English", "Yoruba")),
    "team-insights": ("Ravi Menon", "Bengaluru", "Acme Corp", "IntelliJ", ("English", "Tamil")),
    "team-search": ("Lena Fischer", "Vienna", "Initech", "Zed", ("German", "English")),
    "team-trust": ("Iris Aalto", "Turku", "Initech", "Kate", ("Finnish", "English")),
}

for _i, (_team, (_lead, _city, _employer, _editor, _langs)) in enumerate(LEADS.items(), 1):
    ev(S, f"e-org-{_i:02d}a", _team, "team_lead", _lead, valid_from="2025-02-01",
       source="org_chart", text=f"{_lead} leads {_team}.")
    ev(S, f"e-org-{_i:02d}b", _lead, "lives_in", _city, valid_from="2025-02-01",
       source="hr_directory", text=f"{_lead} is based in {_city}.")
    ev(S, f"e-org-{_i:02d}c", _lead, "works_at", _employer, valid_from="2025-02-01",
       source="hr_directory", text=f"{_lead} is employed by {_employer}.")
    ev(S, f"e-org-{_i:02d}d", _lead, "favourite_editor", _editor, valid_from="2025-03-01",
       source=_lead, text=f"I write everything in {_editor}.")
    for _j, _language in enumerate(_langs, 1):
        ev(S, f"e-org-{_i:02d}l{_j}", _lead, "speaks", _language, valid_from="2025-02-01",
           source="hr_directory", text=f"{_lead} speaks {_language}.")

# Sam Okonkwo and Priya Raman lead team-payments in v1 and were dead ends there: named as
# the object of a claim and the subject of none. Giving them lives is what turns v1's
# `checkout-service -> team-payments -> lead` into a chain that continues.
ev(S, "e-org-sam-1", "Sam Okonkwo", "lives_in", "Accra", valid_from="2026-03-01",
   source="hr_directory", text="Sam Okonkwo is based in Accra.")
ev(S, "e-org-sam-2", "Sam Okonkwo", "works_at", "Globex", valid_from="2026-03-01",
   source="hr_directory", text="Sam Okonkwo is employed by Globex.")
ev(S, "e-org-priya-1", "Priya Raman", "lives_in", "Chennai", valid_from="2025-04-01",
   source="hr_directory", text="Priya Raman is based in Chennai.")
ev(S, "e-org-priya-2", "Priya Raman", "works_at", "Acme Corp", valid_from="2025-04-01",
   source="hr_directory", text="Priya Raman is employed by Acme Corp.")

# Acme Corp is a company in v1 with a plan and no address. Globex and Initech have one.
ev(S, "e-org-acme-hq", "Acme Corp", "hq_city", "Toronto", valid_from="2024-01-01",
   source="crm", text="Acme Corp is headquartered in Toronto.")

# Two more projects, so `works_on` is not a single edge. Kestrel's region moves, which
# makes the two-hop question about it a temporal question as well as a chained one.
ev(S, "e-org-kes-1", "Project Kestrel", "deploy_region", "us-east-1",
   valid_from="2025-05-01", source="platform_team",
   text="Project Kestrel runs in us-east-1.")
ev(S, "e-org-kes-2", "Project Kestrel", "deploy_region", "eu-west-1",
   valid_from="2026-04-01", source="platform_team",
   text="Project Kestrel now runs in eu-west-1.")
ev(S, "e-org-kes-3", "Project Kestrel", "owned_by", "team-core", valid_from="2025-05-01",
   source="service_catalog", text="Project Kestrel is owned by team-core.")
ev(S, "e-org-van-1", "Project Vantage", "deploy_region", "eu-central-1",
   valid_from="2025-08-01", source="platform_team",
   text="Project Vantage runs in eu-central-1.")
ev(S, "e-org-van-2", "Project Vantage", "owned_by", "team-search", valid_from="2025-08-01",
   source="service_catalog", text="Project Vantage is owned by team-search.")
ev(S, "e-org-works-1", "nadia", "works_on", "Project Kestrel", valid_from="2025-06-01",
   source="staffing", text="Nadia has been assigned to Project Kestrel.")
ev(S, "e-org-works-2", "omar", "works_on", "Project Vantage", valid_from="2025-09-01",
   source="staffing", text="Omar has been assigned to Project Vantage.")


# ---------------------------------------------------------------------------
# staffing: everybody works somewhere.
#
# v1's filler people have a birthplace, an editor and two languages, and no employer, so
# no path led out of a person into a company. This is the cheapest edge that raises the
# join rate, and it is the one a real directory would have first.
# ---------------------------------------------------------------------------
S = "staffing"

#: Fixed rather than random. A generator whose output depends on a seed is one more thing
#: a reader has to run to know what is in the file.
EMPLOYERS: dict[str, str] = {
    "anton": "Globex", "bianca": "Acme Corp", "cyrus": "Initech", "dana": "Globex",
    "delia": "Acme Corp", "emil": "Initech", "erin": "Globex", "farida": "Acme Corp",
    "gustav": "Initech", "hana": "Globex", "ines": "Acme Corp", "jonas": "Initech",
    "kira": "Globex", "lukas": "Acme Corp", "marta": "Initech", "nadia": "Globex",
    "nils": "Acme Corp", "olga": "Initech", "omar": "Globex", "pavel": "Acme Corp",
    "petra": "Initech", "quentin": "Globex", "renata": "Acme Corp", "rosa": "Initech",
    "samir": "Globex", "tanya": "Acme Corp", "ulrich": "Initech", "vera": "Globex",
    "wei": "Acme Corp", "xenia": "Initech", "yusuf": "Globex", "zara": "Acme Corp",
}

for _n, (_person, _employer) in enumerate(EMPLOYERS.items(), 1):
    ev(S, f"e-staff-{_n:03d}", _person, "works_at", _employer, valid_from="2025-01-01",
       source="hr_directory", text=f"{_person.title()} works at {_employer}.")


# ---------------------------------------------------------------------------
# chains_v2: what the connective layer makes askable.
#
# All unprobed, all `multi_hop`. A probed chain would measure nothing: told the start and
# the relations, any key-value store follows them. The difficulty is finding the slot
# from the wording, which is what the dimension is for and what its results should be
# read as measuring.
# ---------------------------------------------------------------------------
S = "chains_v2"

q(S, "q2-chain-payments-lead-city", "multi_hop",
  "In which city does the person who leads team-payments live?", value("Accra"),
  note="Two hops: team-payments to its current lead, that lead to a city. The previous "
       "lead lives in Chennai, so the stale answer is also two hops away.")
q(S, "q2-chain-checkout-lead-city", "multi_hop",
  "In which city does the lead of the team that owns the checkout service live?",
  value("Accra"),
  note="Three hops: service, team, lead, city.")
q(S, "q2-chain-pricing-lead", "multi_hop",
  "Who leads the team that owns the pricing service?", value("Lena Fischer"),
  note="Two hops. Seven teams have leads in v2 and one did in v1, so the wrong answers "
       "are now plausible.")
q(S, "q2-chain-search-lead-languages", "multi_hop",
  "Which languages does the lead of team-search speak?", values("German", "English"),
  note="Two hops onto a multi-valued slot.")
q(S, "q2-chain-infra-lead-employer-city", "multi_hop",
  "In which city is the head office of the company employing the lead of team-infra?",
  value("Toronto"),
  note="Three hops: team, lead, employer, head office.")
q(S, "q2-chain-kestrel-region", "multi_hop",
  "Which region is the project Nadia works on deployed to?", value("eu-west-1"),
  note="Two hops where the second moved in April 2026. Getting the chain right and the "
       "clock wrong gives us-east-1.")
q(S, "q2-chain-kestrel-region-then", "multi_hop",
  "Which region was the project Nadia works on deployed to on 2025-12-01?",
  value("us-east-1"), at="2025-12-01",
  note="The same chain with the clock rewound.")
q(S, "q2-chain-vantage-owner", "multi_hop",
  "Which team owns the project Omar works on?", value("team-search"))
q(S, "q2-chain-vantage-owner-lead", "multi_hop",
  "Who leads the team that owns Project Vantage?", value("Lena Fischer"))
q(S, "q2-chain-kestrel-lead-editor", "multi_hop",
  "Which editor does the lead of the team that owns Project Kestrel use?",
  value("Neovim"),
  note="Three hops ending on a fast-moving personal preference.")
q(S, "q2-chain-trust-lead-employer", "multi_hop",
  "Which company employs the lead of team-trust?", value("Initech"))
q(S, "q2-chain-growth-lead-city", "multi_hop",
  "Where does the lead of team-growth live?", value("Gothenburg"))


# ---------------------------------------------------------------------------
# absent_v2: the two middle bands of the negative category.
#
# v1's negatives were an absent slot (every system abstains) or an open question about
# something that was never held (no system abstains). These sit between: the slot is
# real, the store holds values for it, and the answer to *this* question is still
# nothing, because of where the question puts one of the two clocks.
# ---------------------------------------------------------------------------
S = "absent_v2"

# Band 2: the world clock is before anything was ever true.
q(S, "q2-none-alice-before", "negative",
  "Where did Alice live on 2025-06-01?", NOTHING,
  probe=("alice", "lives_in"), at="2025-06-01",
  note="Alice's first residence begins 2026-01-10. Answering Berlin here is answering a "
       "question about a period the record says nothing about.")
q(S, "q2-none-bob-before", "negative",
  "What was Bob's job title on 2024-06-01?", NOTHING,
  probe=("bob", "job_title"), at="2024-06-01",
  note="Bob's first title begins 2025-02-01.")
q(S, "q2-none-atlas-before", "negative",
  "Which region was Project Atlas deployed to on 2025-01-01?", NOTHING,
  probe=("Project Atlas", "deploy_region"), at="2025-01-01",
  note="Atlas's first region begins 2025-09-01.")
q(S, "q2-none-checkout-before", "negative",
  "Which datastore did the checkout service use on 2025-01-01?", NOTHING,
  probe=("checkout-service", "datastore"), at="2025-01-01",
  note="The checkout service's first datastore begins 2025-04-01.")

# Band 3: the world clock is inside a value's life and the belief clock is before we
# heard of it. A single-clock store cannot separate these two questions; it is the same
# collapse that costs it the delayed-knowledge scenarios, asked where the honest answer
# is silence.
q(S, "q2-none-auth-unknown", "negative",
  "What did the record say the auth service used, as of 2024-06-01?", NOTHING,
  probe=("auth-service", "auth_strategy"), at="2024-12-01", known_at="2024-06-01",
  note="API keys were in force from 2024-11-01 and first recorded on that day. Asked what "
       "was believed five months earlier, the answer is nothing — the store had not heard "
       "of the service.")
q(S, "q2-none-atlas-unknown", "negative",
  "What did the record say Project Atlas's region was, as of 2025-06-01?", NOTHING,
  probe=("Project Atlas", "deploy_region"), at="2026-01-01", known_at="2025-06-01",
  note="us-east-1 was in force from 2025-09-01 and recorded that day.")

# Band 4: open, and about something the store never held. Two of these name an entity the
# store knows well, which is what makes them harder than v1's — the nearest match is very
# close and still wrong.
q(S, "q2-none-globex-plan", "negative",
  "Which plan is Globex on?", NOTHING,
  note="Globex is in memory with a head office and thirty employees. `plan` is a real "
       "predicate held by Acme Corp. Neither fact is about Globex's plan.")
q(S, "q2-none-frank-title", "negative",
  "What is Frank's job title?", NOTHING,
  note="Frank has a residence and nothing else. Job titles belong to Bob.")
q(S, "q2-none-meridian-region", "negative",
  "Which region is Project Meridian deployed to?", NOTHING,
  note="Three projects are in memory and this is not one of them.")
q(S, "q2-none-orbit-lead", "negative",
  "Who leads team-orbit?", NOTHING,
  note="Seven teams have leads. team-orbit does not exist.")


# ---------------------------------------------------------------------------

def _interleave(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round-robin the questions across categories.

    `--limit N` takes a prefix, and a prefix of a category-sorted file would be one
    category. Round-robin makes a limited run a spread across all of them, which is what
    a smoke run has to be to mean anything. Same rule as v1, applied to the union: v2's
    order is not v1's order with rows appended, and cannot be.
    """
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(row["category"], []).append(row)
    order = sorted(buckets)
    out: list[dict[str, Any]] = []
    index = 0
    while any(buckets[c][index:] for c in order):
        for category in order:
            bucket = buckets[category]
            if index < len(bucket):
                out.append(bucket[index])
        index += 1
    return out


def _read(name: str) -> list[dict[str, Any]]:
    with (SOURCE / name).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """The three files' contents, as Python objects. v1's rows first, then v2's."""
    inherited_meta = json.loads((SOURCE / "metadata.json").read_text(encoding="utf-8"))
    events = _read("events.jsonl") + _events
    questions = _read("questions.jsonl") + _questions

    seen: set[str] = set()
    for row in events + questions:
        if row["id"] in seen:
            raise ValueError(f"v2 reuses the id {row['id']!r}, which v1 already spends")
        seen.add(row["id"])

    meta = {
        "benchmark": "agent-memory",
        "version": "v2",
        "description": DESCRIPTION,
        "evaluated_at": EVALUATED_AT,
        "predicates": inherited_meta["predicates"],
        "dimensions": inherited_meta["dimensions"],
        "counts": {
            "events": len(events),
            "questions": len(questions),
            "scenarios": len({e["scenario"] for e in events}),
            "entities": len({e["subject"] for e in events}),
        },
    }
    return meta, events, _interleave(questions)


def _write(path: Path, lines: list[str]) -> None:
    """Write `lines`, each terminated by a single LF, on every platform. See
    `build_v1._write` for why this function exists rather than `Path.write_text`."""
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line + "\n")


def main(argv: list[str] | None = None) -> int:
    args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args.add_argument("--out", default=str(OUT), metavar="DIR")
    out = Path(args.parse_args(argv).out)

    meta, events, questions = build()
    out.mkdir(parents=True, exist_ok=True)
    _write(out / "metadata.json", [json.dumps(meta, indent=2, sort_keys=False)])
    _write(out / "events.jsonl", [json.dumps(row, sort_keys=False) for row in events])
    _write(out / "questions.jsonl", [json.dumps(row, sort_keys=False) for row in questions])
    print(f"{len(events)} events, {len(questions)} questions, "
          f"{meta['counts']['scenarios']} scenarios -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
