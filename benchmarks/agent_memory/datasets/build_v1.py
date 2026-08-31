"""Build dataset v1. Run it to regenerate the three files beside this one.

    PYTHONPATH=. python3 benchmarks/agent_memory/datasets/build_v1.py

The generated files are committed, and `tests/test_agent_memory_bench.py` asserts that
rerunning this script reproduces them byte for byte. That is what makes the dataset a
published artefact rather than something that quietly drifts: a change to a scenario is a
diff in `events.jsonl`, visible in review, and a change large enough to move a score is
supposed to be visible.

## Golds are authored here, and checked against the model separately

Every gold answer below is written out by hand. None is read back from
`timeline.Truth`, even though `Truth` could compute most of them. The test suite then
asserts that the two agree. Two independent derivations that must match is the cheapest
protection against the failure a benchmark cannot detect from the inside — a scoring bug
that produces a self-consistent set of wrong answers, which every system then fails
identically and nobody questions, because the numbers look plausible.

## Everything here is synthetic

Invented people, invented companies, invented services. Nothing in this file came from a
real user, a real conversation or a real system, and the whole dataset is safe to publish.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
OUT = HERE / "v1"

#: The instant every unqualified "now" resolves to. A constant rather than the clock,
#: because a benchmark whose gold answers change overnight cannot be reproduced.
EVALUATED_AT = "2026-08-01T00:00:00+00:00"

#: The predicate schema, published with the dataset and handed to every adapter. See
#: `dataset.PredicateDecl` for why this is part of the input rather than something each
#: system is left to infer.
PREDICATES: dict[str, dict[str, str]] = {
    "lives_in": {"cardinality": "one", "volatility": "slow"},
    "job_title": {"cardinality": "one", "volatility": "slow"},
    "works_at": {"cardinality": "one", "volatility": "slow"},
    "works_on": {"cardinality": "one", "volatility": "slow"},
    "plan": {"cardinality": "one", "volatility": "slow"},
    "datastore": {"cardinality": "one", "volatility": "slow"},
    "auth_strategy": {"cardinality": "one", "volatility": "slow"},
    "deploy_region": {"cardinality": "one", "volatility": "slow"},
    "owned_by": {"cardinality": "one", "volatility": "slow"},
    "team_lead": {"cardinality": "one", "volatility": "slow"},
    "hq_city": {"cardinality": "one", "volatility": "static"},
    "born_in": {"cardinality": "one", "volatility": "static"},
    "favourite_editor": {"cardinality": "one", "volatility": "fast"},
    "status": {"cardinality": "one", "volatility": "fast"},
    "speaks": {"cardinality": "many", "volatility": "static"},
}

#: Which categories roll up into each reported dimension. A category belongs to exactly
#: one dimension, so the dimension rows partition the questions and their totals add up
#: to the overall total — a reader can check the arithmetic.
DIMENSIONS: dict[str, list[str]] = {
    "current_state": ["current_state"],
    "temporal": ["historical_state", "change_time", "change_detection"],
    "knowledge_time": ["knowledge_time"],
    "contradiction": ["contradiction"],
    "provenance": ["provenance"],
    "retrieval": ["multi_hop", "distractor"],
    "irrelevance": ["negative"],
}

DESCRIPTION = (
    "Agent Memory Benchmark v1. Fourteen authored scenarios over one shared memory, "
    "covering progressive change, reversion, delayed knowledge, same-instant "
    "contradiction, repeated observation, multi-valued predicates, multi-hop retrieval "
    "and hard negatives, plus unrelated filler that is never asked about."
)

_events: list[dict[str, Any]] = []
_questions: list[dict[str, Any]] = []


def ev(scenario: str, eid: str, subject: str, predicate: str, obj: str, *,
       valid_from: str, recorded_at: str | None = None, source: str,
       text: str, confidence: float = 1.0, valid_to: str | None = None) -> None:
    """Append one observation. `recorded_at` defaults to `valid_from` — told the day it
    became true, which is the ordinary case and not the interesting one."""
    row: dict[str, Any] = {
        "id": eid, "scenario": scenario,
        "recorded_at": _iso(recorded_at or valid_from), "valid_from": _iso(valid_from),
        "subject": subject, "predicate": predicate, "object": obj,
        "text": text, "source": source, "confidence": confidence,
    }
    if valid_to:
        row["valid_to"] = _iso(valid_to)
    _events.append(row)


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


def on(day: str) -> dict[str, Any]:
    return {"kind": "date", "value": day}


NOTHING: dict[str, Any] = {"kind": "none"}


def _iso(day: str) -> str:
    return datetime.fromisoformat(day).replace(tzinfo=timezone.utc).isoformat() \
        if len(day) == 10 else day


# ---------------------------------------------------------------------------
# 1. alice_relocation — the canonical progression: A, then B, then C.
# ---------------------------------------------------------------------------
S = "alice_relocation"
ev(S, "e-alice-1", "alice", "lives_in", "Berlin", valid_from="2026-01-10",
   source="alice", text="I live in Berlin.")
ev(S, "e-alice-2", "alice", "lives_in", "London", valid_from="2026-03-15",
   source="hr_directory", text="Alice relocated to the London office.")
ev(S, "e-alice-3", "alice", "lives_in", "New York", valid_from="2026-06-02",
   source="alice", text="I have moved to New York.")

q(S, "q-alice-current", "current_state", "Where does Alice live now?",
  value("New York", "NYC"), probe=("alice", "lives_in"))
q(S, "q-alice-hist-mar", "historical_state", "Where did Alice live on 2026-03-20?",
  value("London"), probe=("alice", "lives_in"), at="2026-03-20")
q(S, "q-alice-hist-jan", "historical_state", "Where did Alice live on 2026-01-20?",
  value("Berlin"), probe=("alice", "lives_in"), at="2026-01-20")
q(S, "q-alice-hist-jul", "historical_state", "Where did Alice live on 2026-07-01?",
  value("New York", "NYC"), probe=("alice", "lives_in"), at="2026-07-01")
q(S, "q-alice-hist-edge", "historical_state", "Where did Alice live on 2026-03-14?",
  value("Berlin"), probe=("alice", "lives_in"), at="2026-03-14",
  note="The day before the move. An off-by-one in interval handling shows up here.")
q(S, "q-alice-hist-boundary", "historical_state", "Where did Alice live on 2026-03-15?",
  value("London"), probe=("alice", "lives_in"), at="2026-03-15",
  note="The move day itself. Intervals are closed at the start and open at the end.")
q(S, "q-alice-changes", "change_detection",
  "Which cities has Alice lived in, according to everything on record?",
  values("Berlin", "London", "New York"), probe=("alice", "lives_in"))
q(S, "q-alice-ctime", "change_time", "On what date did Alice move to London?",
  on("2026-03-15"), probe=("alice", "lives_in"),
  about="London")
q(S, "q-alice-ktime", "knowledge_time",
  "On what date did the system first record that Alice lived in London?",
  on("2026-03-15"), probe=("alice", "lives_in"),
  about="London")
q(S, "q-alice-prov-london", "provenance",
  "Which source reported that Alice lives in London?",
  value("hr_directory"), probe=("alice", "lives_in"),
  about="London")
q(S, "q-alice-prov-ny", "provenance",
  "Which source reported that Alice lives in New York?",
  value("alice"), probe=("alice", "lives_in"),
  about="New York")

# ---------------------------------------------------------------------------
# 2. bob_promotions — four values on one slot, so "latest wins" is not enough.
# ---------------------------------------------------------------------------
S = "bob_promotions"
ev(S, "e-bob-1", "bob", "job_title", "engineer", valid_from="2025-02-01",
   source="hr_directory", text="Bob joined as an engineer.")
ev(S, "e-bob-2", "bob", "job_title", "senior engineer", valid_from="2025-11-01",
   source="hr_directory", text="Bob was promoted to senior engineer.")
ev(S, "e-bob-3", "bob", "job_title", "staff engineer", valid_from="2026-04-01",
   source="hr_directory", text="Bob was promoted to staff engineer.")
ev(S, "e-bob-4", "bob", "job_title", "engineering manager", valid_from="2026-07-15",
   source="bob", text="I moved into an engineering manager role.")

q(S, "q-bob-current", "current_state", "What is Bob's job title now?",
  value("engineering manager"), probe=("bob", "job_title"))
q(S, "q-bob-hist-1", "historical_state", "What was Bob's job title on 2025-06-01?",
  value("engineer"), probe=("bob", "job_title"), at="2025-06-01")
q(S, "q-bob-hist-2", "historical_state", "What was Bob's job title on 2026-01-10?",
  value("senior engineer"), probe=("bob", "job_title"), at="2026-01-10")
q(S, "q-bob-hist-3", "historical_state", "What was Bob's job title on 2026-05-05?",
  value("staff engineer"), probe=("bob", "job_title"), at="2026-05-05")
q(S, "q-bob-changes", "change_detection",
  "Which job titles has Bob held, according to everything on record?",
  values("engineer", "senior engineer", "staff engineer", "engineering manager"),
  probe=("bob", "job_title"))
q(S, "q-bob-ctime", "change_time", "On what date did Bob become a staff engineer?",
  on("2026-04-01"), probe=("bob", "job_title"),
  about="staff engineer")
q(S, "q-bob-prov", "provenance",
  "Which source reported that Bob is an engineering manager?",
  value("bob"), probe=("bob", "job_title"),
  about="engineering manager")

# ---------------------------------------------------------------------------
# 3. charlie_reversion — A, B, then back to A. The middle period is the question.
# ---------------------------------------------------------------------------
S = "charlie_reversion"
ev(S, "e-charlie-1", "charlie", "lives_in", "Berlin", valid_from="2025-05-01",
   source="charlie", text="I live in Berlin.")
ev(S, "e-charlie-2", "charlie", "lives_in", "London", valid_from="2025-12-01",
   source="charlie", text="I have moved to London.")
ev(S, "e-charlie-3", "charlie", "lives_in", "Berlin", valid_from="2026-05-20",
   source="charlie", text="I have moved back to Berlin.")

q(S, "q-charlie-current", "current_state", "Where does Charlie live now?",
  value("Berlin"), probe=("charlie", "lives_in"))
q(S, "q-charlie-middle", "historical_state", "Where did Charlie live on 2026-01-15?",
  value("London"), probe=("charlie", "lives_in"), at="2026-01-15",
  note="The middle period of a reversion. A store keyed only on the current value "
       "cannot distinguish this instant from today.")
q(S, "q-charlie-first", "historical_state", "Where did Charlie live on 2025-06-01?",
  value("Berlin"), probe=("charlie", "lives_in"), at="2025-06-01")
q(S, "q-charlie-after", "historical_state", "Where did Charlie live on 2026-06-01?",
  value("Berlin"), probe=("charlie", "lives_in"), at="2026-06-01")
q(S, "q-charlie-changes", "change_detection",
  "Which cities has Charlie lived in, according to everything on record?",
  values("Berlin", "London"), probe=("charlie", "lives_in"),
  note="Two distinct cities across three moves. Counting moves rather than values "
       "gives three.")
q(S, "q-charlie-ctime", "change_time", "On what date did Charlie move to London?",
  on("2025-12-01"), probe=("charlie", "lives_in"), about="London",
  note="Berlin would be ambiguous here: Charlie held it twice, so `about` could not "
       "name which interval the question means. See dataset.validate.")

# ---------------------------------------------------------------------------
# 4. atlas_deploy — delayed knowledge. True on the 1st, heard on the 10th.
# ---------------------------------------------------------------------------
S = "atlas_deploy"
ev(S, "e-atlas-1", "Project Atlas", "deploy_region", "us-east-1",
   valid_from="2025-09-01", source="deploy_log",
   text="Project Atlas is deployed to us-east-1.")
ev(S, "e-atlas-2", "Project Atlas", "deploy_region", "eu-west-1",
   valid_from="2026-03-01", recorded_at="2026-03-10", source="deploy_log",
   text="Project Atlas moved to eu-west-1 on 1 March; the log was imported on 10 March.")

q(S, "q-atlas-current", "current_state",
  "Which region is Project Atlas deployed to now?",
  value("eu-west-1"), probe=("Project Atlas", "deploy_region"))
q(S, "q-atlas-hist-feb", "historical_state",
  "Which region was Project Atlas deployed to on 2026-02-01?",
  value("us-east-1"), probe=("Project Atlas", "deploy_region"), at="2026-02-01")
q(S, "q-atlas-hist-gap", "historical_state",
  "Which region was Project Atlas deployed to on 2026-03-05?",
  value("eu-west-1"), probe=("Project Atlas", "deploy_region"), at="2026-03-05",
  note="Inside the window between the move and the import. As we understand it today "
       "the answer is eu-west-1, even though the store did not know it at the time.")
q(S, "q-atlas-asof", "knowledge_time",
  "What would the system have said on 2026-03-05 about Project Atlas's deployment region?",
  value("us-east-1"), probe=("Project Atlas", "deploy_region"),
  at="2026-03-05", known_at="2026-03-05",
  note="Both clocks rewound. This is the audit reading, and it is a different answer "
       "from the previous question about the same instant.")
q(S, "q-atlas-ctime", "change_time",
  "On what date did Project Atlas move to eu-west-1?",
  on("2026-03-01"), probe=("Project Atlas", "deploy_region"),
  about="eu-west-1")
q(S, "q-atlas-ktime", "knowledge_time",
  "On what date did the system learn that Project Atlas had moved to eu-west-1?",
  on("2026-03-10"), probe=("Project Atlas", "deploy_region"),
  note="Nine days after it happened. A store with one clock has to answer this and the "
       "previous question with the same date.",
  about="eu-west-1")
q(S, "q-atlas-prov", "provenance",
  "Which source reported that Project Atlas is deployed to eu-west-1?",
  value("deploy_log"), probe=("Project Atlas", "deploy_region"),
  about="eu-west-1")

# ---------------------------------------------------------------------------
# 5. checkout_datastore — an engineering fact with a documented origin.
# ---------------------------------------------------------------------------
S = "checkout_datastore"
ev(S, "e-checkout-1", "checkout-service", "datastore", "Redis",
   valid_from="2025-04-01", source="adr_014",
   text="ADR 014: checkout-service will use Redis for session state.")
ev(S, "e-checkout-2", "checkout-service", "datastore", "PostgreSQL",
   valid_from="2026-01-20", source="adr_027",
   text="ADR 027: checkout-service migrates from Redis to PostgreSQL.")

q(S, "q-checkout-current", "current_state",
  "Which datastore does the checkout service use now?",
  value("PostgreSQL", "Postgres"), probe=("checkout-service", "datastore"))
q(S, "q-checkout-hist", "historical_state",
  "Which datastore did the checkout service use on 2025-08-01?",
  value("Redis"), probe=("checkout-service", "datastore"), at="2025-08-01")
q(S, "q-checkout-changes", "change_detection",
  "Which datastores has the checkout service used, according to everything on record?",
  values("Redis", "PostgreSQL"), probe=("checkout-service", "datastore"))
q(S, "q-checkout-ctime", "change_time",
  "On what date did the checkout service migrate to PostgreSQL?",
  on("2026-01-20"), probe=("checkout-service", "datastore"),
  about="PostgreSQL")
q(S, "q-checkout-prov-new", "provenance",
  "Which source recorded that the checkout service uses PostgreSQL?",
  value("adr_027"), probe=("checkout-service", "datastore"),
  about="PostgreSQL")
q(S, "q-checkout-prov-old", "provenance",
  "Which source recorded that the checkout service used Redis?",
  value("adr_014"), probe=("checkout-service", "datastore"),
  about="Redis")

# ---------------------------------------------------------------------------
# 6. auth_migration — a second delayed-knowledge case, on an engineering slot.
# ---------------------------------------------------------------------------
S = "auth_migration"
ev(S, "e-auth-1", "auth-service", "auth_strategy", "API keys",
   valid_from="2024-11-01", source="rfc_12",
   text="RFC 12: auth-service authenticates callers with API keys.")
ev(S, "e-auth-2", "auth-service", "auth_strategy", "OAuth 2.0",
   valid_from="2026-02-10", recorded_at="2026-02-24", source="rfc_31",
   text="RFC 31: auth-service cut over to OAuth 2.0 on 10 February; the RFC landed on the 24th.")

q(S, "q-auth-current", "current_state",
  "Which authentication strategy does the auth service use now?",
  value("OAuth 2.0", "OAuth2", "OAuth"), probe=("auth-service", "auth_strategy"))
q(S, "q-auth-hist", "historical_state",
  "Which authentication strategy did the auth service use on 2026-02-15?",
  value("OAuth 2.0", "OAuth2", "OAuth"), probe=("auth-service", "auth_strategy"),
  at="2026-02-15")
q(S, "q-auth-asof", "knowledge_time",
  "What would the system have said on 2026-02-15 about the auth service's authentication strategy?",
  value("API keys"), probe=("auth-service", "auth_strategy"),
  at="2026-02-15", known_at="2026-02-15")
q(S, "q-auth-ctime", "change_time",
  "On what date did the auth service cut over to OAuth 2.0?",
  on("2026-02-10"), probe=("auth-service", "auth_strategy"),
  about="OAuth 2.0")
q(S, "q-auth-ktime", "knowledge_time",
  "On what date did the system learn that the auth service had cut over to OAuth 2.0?",
  on("2026-02-24"), probe=("auth-service", "auth_strategy"),
  about="OAuth 2.0")
q(S, "q-auth-prov", "provenance",
  "Which source recorded that the auth service uses OAuth 2.0?",
  value("rfc_31"), probe=("auth-service", "auth_strategy"),
  about="OAuth 2.0")

# ---------------------------------------------------------------------------
# 7. dana_conflict — two reports about the same instant, one arriving later.
# ---------------------------------------------------------------------------
S = "dana_conflict"
ev(S, "e-dana-1", "dana", "lives_in", "London", valid_from="2026-04-01",
   recorded_at="2026-04-02", source="colleague_hearsay", confidence=0.4,
   text="Someone mentioned that Dana had moved to London.")
ev(S, "e-dana-2", "dana", "lives_in", "Paris", valid_from="2026-04-01",
   recorded_at="2026-04-05", source="dana", confidence=1.0,
   text="I moved to Paris at the start of April, not London.")
ev(S, "e-dana-3", "dana", "lives_in", "Paris", valid_from="2026-04-01",
   recorded_at="2026-04-20", source="hr_directory", confidence=1.0,
   text="Dana's address of record is in Paris, effective 1 April.")

q(S, "q-dana-current", "contradiction", "Where does Dana live now?",
  value("Paris"), probe=("dana", "lives_in"),
  note="Two sources reported different cities for the same instant. The later record "
       "corrects the earlier one.")
q(S, "q-dana-asof", "contradiction",
  "What would the system have said on 2026-04-03 about where Dana lived?",
  value("London"), probe=("dana", "lives_in"), at="2026-04-03", known_at="2026-04-03",
  note="The correction had not arrived. A store that overwrites in place cannot "
       "reconstruct this and cannot show what someone acted on that day.")
q(S, "q-dana-hist", "historical_state", "Where did Dana live on 2026-04-03?",
  value("Paris"), probe=("dana", "lives_in"), at="2026-04-03",
  note="Same instant as the previous question, asked with today's understanding. The "
       "two answers differ, and that difference is the whole point of two clocks.")
q(S, "q-dana-changes", "change_detection",
  "Which cities has Dana lived in, according to everything the system still believes?",
  values("Paris"), probe=("dana", "lives_in"),
  note="London was retracted, not outlived. It is not a city Dana has lived in.")
q(S, "q-dana-prov-paris", "provenance",
  "Which source first reported that Dana lives in Paris?",
  value("dana"), probe=("dana", "lives_in"),
  about="Paris")
q(S, "q-dana-prov-london", "provenance",
  "Which source reported that Dana lived in London?",
  value("colleague_hearsay"), probe=("dana", "lives_in"),
  about="London")

# ---------------------------------------------------------------------------
# 8. erin_ambiguity — a move, a contradicting report a month later, then a move back.
# ---------------------------------------------------------------------------
S = "erin_ambiguity"
ev(S, "e-erin-1", "erin", "lives_in", "London", valid_from="2026-03-01",
   recorded_at="2026-03-05", source="erin", text="I moved to London in March.")
ev(S, "e-erin-2", "erin", "lives_in", "Berlin", valid_from="2026-04-01",
   recorded_at="2026-04-03", source="colleague_hearsay", confidence=0.5,
   text="Erin was seen working out of the Berlin office in April.")
ev(S, "e-erin-3", "erin", "lives_in", "London", valid_from="2026-05-01",
   recorded_at="2026-05-08", source="erin", text="I am back in London as of May.")

q(S, "q-erin-april", "historical_state", "Where did Erin live on 2026-04-15?",
  value("Berlin"), probe=("erin", "lives_in"), at="2026-04-15",
  note="A later report about a later instant is a change, not a correction: it has its "
       "own valid_from, so it does not retract March.")
q(S, "q-erin-march", "historical_state", "Where did Erin live on 2026-03-15?",
  value("London"), probe=("erin", "lives_in"), at="2026-03-15")
q(S, "q-erin-current", "current_state", "Where does Erin live now?",
  value("London"), probe=("erin", "lives_in"))
q(S, "q-erin-changes", "change_detection",
  "Which cities has Erin lived in, according to everything on record?",
  values("London", "Berlin"), probe=("erin", "lives_in"))
q(S, "q-erin-ctime", "change_time",
  "On what date did Erin start living in Berlin?",
  on("2026-04-01"), probe=("erin", "lives_in"), about="Berlin")
q(S, "q-erin-ktime", "knowledge_time",
  "On what date did the system first record that Erin was living in Berlin?",
  on("2026-04-03"), probe=("erin", "lives_in"), about="Berlin",
  note="Two days after the fact. The change_time question about the same value answers "
       "2026-04-01, and a store with one clock has to give the same date to both.")

# ---------------------------------------------------------------------------
# 9. acme_plan — a customer plan that goes up and then comes back down.
# ---------------------------------------------------------------------------
S = "acme_plan"
ev(S, "e-acme-1", "Acme Corp", "plan", "free", valid_from="2024-03-01",
   source="billing_system", text="Acme Corp signed up on the free plan.")
ev(S, "e-acme-2", "Acme Corp", "plan", "pro", valid_from="2025-01-15",
   source="billing_system", text="Acme Corp upgraded to the pro plan.")
ev(S, "e-acme-3", "Acme Corp", "plan", "enterprise", valid_from="2025-09-01",
   source="billing_system", text="Acme Corp upgraded to the enterprise plan.")
ev(S, "e-acme-4", "Acme Corp", "plan", "pro", valid_from="2026-06-01",
   source="billing_system", text="Acme Corp downgraded to the pro plan.")

q(S, "q-acme-current", "current_state", "Which plan is Acme Corp on now?",
  value("pro"), probe=("Acme Corp", "plan"))
q(S, "q-acme-hist-free", "historical_state", "Which plan was Acme Corp on on 2024-06-01?",
  value("free"), probe=("Acme Corp", "plan"), at="2024-06-01")
q(S, "q-acme-hist-pro", "historical_state", "Which plan was Acme Corp on on 2025-03-01?",
  value("pro"), probe=("Acme Corp", "plan"), at="2025-03-01")
q(S, "q-acme-hist-ent", "historical_state", "Which plan was Acme Corp on on 2025-11-01?",
  value("enterprise"), probe=("Acme Corp", "plan"), at="2025-11-01")
q(S, "q-acme-changes", "change_detection",
  "Which plans has Acme Corp been on, according to everything on record?",
  values("free", "pro", "enterprise"), probe=("Acme Corp", "plan"))
q(S, "q-acme-ctime", "change_time",
  "On what date did Acme Corp move onto the enterprise plan?",
  on("2025-09-01"), probe=("Acme Corp", "plan"),
  about="enterprise")

# ---------------------------------------------------------------------------
# 10. billing_repeat — the same fact restated four times, then one real change.
# ---------------------------------------------------------------------------
S = "billing_repeat"
for eid, day, source, text in (
    ("e-billing-1", "2025-01-10", "adr_009", "ADR 009: billing-service stores invoices in MySQL."),
    ("e-billing-2", "2025-04-02", "oncall_notes", "billing-service is on MySQL, as noted during the incident."),
    ("e-billing-3", "2025-08-19", "standup", "Reminder: billing-service uses MySQL."),
    ("e-billing-4", "2025-12-01", "runbook", "The billing-service runbook documents MySQL."),
):
    ev(S, eid, "billing-service", "datastore", "MySQL", valid_from=day, source=source, text=text)
ev(S, "e-billing-5", "billing-service", "datastore", "PostgreSQL", valid_from="2026-05-05",
   source="adr_041", text="ADR 041: billing-service migrates from MySQL to PostgreSQL.")

q(S, "q-billing-changes", "change_detection",
  "Which datastores has the billing service used, according to everything on record?",
  values("MySQL", "PostgreSQL"), probe=("billing-service", "datastore"),
  note="Five observations, two values. A system that counts writes answers five.")
q(S, "q-billing-current", "current_state",
  "Which datastore does the billing service use now?",
  value("PostgreSQL", "Postgres"), probe=("billing-service", "datastore"))
q(S, "q-billing-hist", "historical_state",
  "Which datastore did the billing service use on 2025-09-01?",
  value("MySQL"), probe=("billing-service", "datastore"), at="2025-09-01")
q(S, "q-billing-ctime", "change_time",
  "On what date did the billing service migrate to PostgreSQL?",
  on("2026-05-05"), probe=("billing-service", "datastore"),
  about="PostgreSQL")
q(S, "q-billing-prov", "provenance",
  "Which source first recorded that the billing service used MySQL?",
  value("adr_009"), probe=("billing-service", "datastore"),
  about="MySQL")

# ---------------------------------------------------------------------------
# 11. alice_languages — a multi-valued predicate, which must not resolve.
# ---------------------------------------------------------------------------
S = "alice_languages"
ev(S, "e-lang-1", "alice", "speaks", "English", valid_from="2024-01-01",
   source="alice", text="I speak English.")
ev(S, "e-lang-2", "alice", "speaks", "German", valid_from="2025-03-01",
   source="alice", text="I have learned German.")
ev(S, "e-lang-3", "alice", "speaks", "Portuguese", valid_from="2026-02-01",
   source="alice", text="I have picked up Portuguese.")

q(S, "q-lang-current", "contradiction", "Which languages does Alice speak now?",
  values("English", "German", "Portuguese"), probe=("alice", "speaks"),
  note="`speaks` is declared multi-valued, so a later value joins the earlier ones. A "
       "system that resolves every slot the same way answers Portuguese alone.")
q(S, "q-lang-hist", "historical_state",
  "Which languages did Alice speak on 2025-06-01?",
  values("English", "German"), probe=("alice", "speaks"), at="2025-06-01")
q(S, "q-lang-changes", "change_detection",
  "Which languages has Alice ever spoken, according to everything on record?",
  values("English", "German", "Portuguese"), probe=("alice", "speaks"))

# ---------------------------------------------------------------------------
# 12. london_crowd — five people, four of them in the same city.
# ---------------------------------------------------------------------------
S = "london_crowd"
ev(S, "e-frank-1", "frank", "lives_in", "London", valid_from="2025-01-01",
   source="hr_directory", text="Frank is based in London.")
ev(S, "e-grace-1", "grace", "lives_in", "London", valid_from="2025-02-01",
   source="hr_directory", text="Grace is based in London.")
ev(S, "e-judy-1", "judy", "lives_in", "London", valid_from="2025-07-01",
   source="hr_directory", text="Judy is based in London.")
ev(S, "e-heidi-1", "heidi", "lives_in", "London", valid_from="2024-06-01",
   source="hr_directory", text="Heidi is based in London.")
ev(S, "e-heidi-2", "heidi", "lives_in", "Madrid", valid_from="2026-02-01",
   source="heidi", text="I have relocated to Madrid.")
ev(S, "e-ivan-1", "ivan", "lives_in", "Lisbon", valid_from="2025-03-01",
   source="hr_directory", text="Ivan is based in Lisbon.")

q(S, "q-crowd-ivan", "distractor", "Where does Ivan live?", value("Lisbon"),
  note="Four of the five people in this scenario live in London. Ivan does not.")
q(S, "q-crowd-heidi", "distractor", "Where does Heidi live now?", value("Madrid"),
  note="Heidi lived in London until 2026. The stale value is also the popular one.")
q(S, "q-crowd-judy", "distractor", "Where does Judy live?", value("London"))
q(S, "q-crowd-grace", "distractor", "Where does Grace live?", value("London"))
q(S, "q-crowd-heidi-hist", "historical_state", "Where did Heidi live on 2025-08-01?",
  value("London"), probe=("heidi", "lives_in"), at="2025-08-01")
q(S, "q-crowd-heidi-changes", "change_detection",
  "Which cities has Heidi lived in, according to everything on record?",
  values("London", "Madrid"), probe=("heidi", "lives_in"))

# ---------------------------------------------------------------------------
# 13. ownership_chain — answers that need two facts joined.
# ---------------------------------------------------------------------------
S = "ownership_chain"
ev(S, "e-chain-1", "alice", "works_on", "Project Atlas", valid_from="2025-10-01",
   source="hr_directory", text="Alice was assigned to Project Atlas.")
ev(S, "e-chain-2", "bob", "works_at", "Globex", valid_from="2025-06-01",
   source="hr_directory", text="Bob works at Globex.")
ev(S, "e-chain-3", "Globex", "hq_city", "Munich", valid_from="2024-01-01",
   source="company_registry", text="Globex is headquartered in Munich.")
ev(S, "e-chain-4", "checkout-service", "owned_by", "team-payments",
   valid_from="2025-04-01", source="service_catalog",
   text="checkout-service is owned by team-payments.")
ev(S, "e-chain-5", "team-payments", "team_lead", "Priya Raman", valid_from="2025-04-01",
   source="hr_directory", text="Priya Raman leads team-payments.")
ev(S, "e-chain-6", "team-payments", "team_lead", "Sam Okonkwo", valid_from="2026-03-01",
   source="hr_directory", text="Sam Okonkwo took over team-payments.")

q(S, "q-chain-atlas", "multi_hop",
  "Which region is the project Alice works on deployed to?", value("eu-west-1"),
  note="Two hops: Alice to Project Atlas, Project Atlas to its region. The region "
       "changed in March 2026, so the stale answer is also two hops away.")
q(S, "q-chain-globex", "multi_hop",
  "In which city is Bob's employer headquartered?", value("Munich"))
q(S, "q-chain-lead-now", "multi_hop",
  "Who currently leads the team that owns the checkout service?", value("Sam Okonkwo"))
q(S, "q-chain-lead-then", "multi_hop",
  "Who led the team that owns the checkout service on 2025-09-01?",
  value("Priya Raman"), at="2025-09-01",
  note="Two hops and a rewound clock. Both have to be right.")
q(S, "q-chain-lead-current", "current_state", "Who leads team-payments now?",
  value("Sam Okonkwo"), probe=("team-payments", "team_lead"))
q(S, "q-chain-lead-hist", "historical_state",
  "Who led team-payments on 2025-12-01?", value("Priya Raman"),
  probe=("team-payments", "team_lead"), at="2025-12-01")

# ---------------------------------------------------------------------------
# 14. absent — facts the system was never told, about entities it has heard of.
# ---------------------------------------------------------------------------
S = "absent"
ev(S, "e-absent-1", "mallory", "born_in", "Ljubljana", valid_from="2024-01-01",
   source="hr_directory", text="Mallory was born in Ljubljana.")
ev(S, "e-absent-2", "mallory", "favourite_editor", "Emacs", valid_from="2025-05-01",
   source="mallory", text="I use Emacs.")
ev(S, "e-absent-3", "reporting-service", "owned_by", "team-insights",
   valid_from="2025-02-01", source="service_catalog",
   text="reporting-service is owned by team-insights.")
ev(S, "e-absent-4", "Initech", "hq_city", "Austin", valid_from="2024-01-01",
   source="company_registry", text="Initech is headquartered in Austin.")

q(S, "q-absent-mallory", "negative", "Where does Mallory live?", NOTHING,
  probe=("mallory", "lives_in"),
  note="Mallory is in memory, with a birthplace and an editor. Neither is a residence.")
q(S, "q-absent-reporting", "negative",
  "Which datastore does the reporting service use?", NOTHING,
  probe=("reporting-service", "datastore"))
q(S, "q-absent-initech", "negative", "Which plan is Initech on?", NOTHING,
  probe=("Initech", "plan"))
q(S, "q-absent-open-1", "negative", "Where does Oscar live?", NOTHING,
  note="Oscar appears nowhere in the dataset at all.")
q(S, "q-absent-open-2", "negative",
  "Which authentication strategy does the reporting service use?", NOTHING)
q(S, "q-absent-open-3", "negative", "Which region is Project Chronos deployed to?",
  NOTHING, note="A plausible sibling of Project Atlas that does not exist.")


# ---------------------------------------------------------------------------
# 15. quotes_correction — a second same-instant contradiction, on an engineering slot.
# ---------------------------------------------------------------------------
S = "quotes_correction"
ev(S, "e-quotes-1", "quotes-service", "datastore", "Redis", valid_from="2026-01-01",
   recorded_at="2026-01-05", source="standup", confidence=0.5,
   text="Someone said in standup that quotes-service is backed by Redis.")
ev(S, "e-quotes-2", "quotes-service", "datastore", "DynamoDB", valid_from="2026-01-01",
   recorded_at="2026-01-12", source="adr_038", confidence=1.0,
   text="ADR 038: quotes-service has been backed by DynamoDB since 1 January. The "
        "standup note was wrong.")

q(S, "q-quotes-current", "contradiction",
  "Which datastore does the quotes service use now?", value("DynamoDB"),
  probe=("quotes-service", "datastore"),
  note="The first report was corrected, not outlived: both describe the same instant.")
q(S, "q-quotes-asof", "contradiction",
  "What would the system have said on 2026-01-08 about the quotes service's datastore?",
  value("Redis"), probe=("quotes-service", "datastore"),
  at="2026-01-08", known_at="2026-01-08")
q(S, "q-quotes-changes", "change_detection",
  "Which datastores has the quotes service used, according to everything the system still believes?",
  values("DynamoDB"), probe=("quotes-service", "datastore"),
  note="Redis was retracted. A store that keeps every write answers both.")
q(S, "q-quotes-hist", "historical_state",
  "Which datastore did the quotes service use on 2026-01-08?", value("DynamoDB"),
  probe=("quotes-service", "datastore"), at="2026-01-08",
  note="The same instant as q-quotes-asof, asked with today's understanding.")
q(S, "q-quotes-prov", "provenance",
  "Which source recorded that the quotes service uses DynamoDB?", value("adr_038"),
  probe=("quotes-service", "datastore"), about="DynamoDB")
q(S, "q-quotes-ktime", "knowledge_time",
  "On what date did the system record that the quotes service uses DynamoDB?",
  on("2026-01-12"), probe=("quotes-service", "datastore"), about="DynamoDB")

# ---------------------------------------------------------------------------
# 16. Open questions — the same facts, with no slot named, over the whole memory.
# ---------------------------------------------------------------------------
q("london_crowd", "q-crowd-frank", "distractor", "Where does Frank live?", value("London"))
q("atlas_deploy", "q-open-atlas", "distractor",
  "Which region is Project Atlas deployed to?", value("eu-west-1"),
  note="The probed form of this question is q-atlas-current. Here the system has to "
       "find the slot itself, among 260 memories.")
q("auth_migration", "q-open-auth", "distractor",
  "Which authentication strategy does the auth service use?",
  value("OAuth 2.0", "OAuth2", "OAuth"))
q("acme_plan", "q-open-acme", "distractor", "Which plan is Acme Corp on?", value("pro"))
q("alice_languages", "q-lang-early", "historical_state",
  "Which languages did Alice speak on 2024-06-01?", values("English"),
  probe=("alice", "speaks"), at="2024-06-01")

q("ownership_chain", "q-chain-datastore", "multi_hop",
  "Which datastore does the service owned by team-payments use?",
  value("PostgreSQL", "Postgres"),
  note="A reverse hop: from the team to the service it owns, then to that service's "
       "datastore, which changed in January 2026.")
q("ownership_chain", "q-chain-languages", "multi_hop",
  "Which languages does the person who works on Project Atlas speak?",
  values("English", "German", "Portuguese"),
  note="A reverse hop onto a multi-valued slot.")


# ---------------------------------------------------------------------------
# Filler. Never asked about, and there to make retrieval do some work.
# ---------------------------------------------------------------------------
FILLER_PEOPLE = [
    "nadia", "omar", "petra", "quentin", "rosa", "samir", "tanya", "ulrich", "vera",
    "wei", "xenia", "yusuf", "zara", "anton", "bianca", "cyrus", "delia", "emil",
    "farida", "gustav", "hana", "ines", "jonas", "kira", "lukas", "marta", "nils",
    "olga", "pavel", "renata",
]
FILLER_CITIES = ["Porto", "Tallinn", "Osaka", "Nairobi", "Bogota", "Helsinki",
                 "Krakow", "Seville", "Toronto", "Auckland", "Dakar", "Bergen"]
FILLER_EDITORS = ["Neovim", "Zed", "Sublime Text", "Helix", "VS Code", "IntelliJ", "Kate"]
FILLER_LANGUAGES = ["Italian", "Finnish", "Swahili", "Korean", "Dutch", "Czech", "Tamil"]
FILLER_SERVICES = [
    "search-service", "notify-service", "ingest-service", "media-service",
    "ledger-service", "quota-service", "audit-service", "shipping-service",
    "catalog-service", "pricing-service", "session-service", "webhook-service",
    "export-service", "throttle-service", "geocode-service",
]
FILLER_TEAMS = ["team-core", "team-growth", "team-infra", "team-search", "team-trust"]
FILLER_STATUSES = ["healthy", "degraded", "maintenance", "healthy", "healthy"]


def _filler() -> None:
    """Deterministic filler. The seed is fixed and the ordering is derived from it, so
    regenerating this file produces the same bytes on any machine."""
    rng = random.Random(20260801)
    S = "filler"
    n = 0
    for person in FILLER_PEOPLE:
        n += 1
        city = rng.choice(FILLER_CITIES)
        ev(S, f"e-fill-{n:04d}", person, "born_in", city, valid_from="2024-01-01",
           source="hr_directory", text=f"{person.title()} was born in {city}.")
        n += 1
        editor = rng.choice(FILLER_EDITORS)
        day = f"2025-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
        ev(S, f"e-fill-{n:04d}", person, "favourite_editor", editor, valid_from=day,
           source=person, text=f"I do most of my work in {editor}.")
        for language in rng.sample(FILLER_LANGUAGES, 2):
            n += 1
            ev(S, f"e-fill-{n:04d}", person, "speaks", language, valid_from="2024-06-01",
               source="hr_directory", text=f"{person.title()} speaks {language}.")
    for service in FILLER_SERVICES:
        n += 1
        team = rng.choice(FILLER_TEAMS)
        ev(S, f"e-fill-{n:04d}", service, "owned_by", team, valid_from="2025-01-01",
           source="service_catalog", text=f"{service} is owned by {team}.")
        n += 1
        store = rng.choice(["MySQL", "PostgreSQL", "Redis", "DynamoDB"])
        ev(S, f"e-fill-{n:04d}", service, "datastore", store, valid_from="2025-03-01",
           source="service_catalog", text=f"{service} stores its data in {store}.")
        for week in range(4):
            n += 1
            status = rng.choice(FILLER_STATUSES)
            day = f"2026-0{week + 1}-1{week}"
            ev(S, f"e-fill-{n:04d}", service, "status", status, valid_from=day,
               source="healthcheck", text=f"{service} reported {status}.")


def _interleave(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round-robin the questions across categories.

    `--limit N` takes a prefix, and a prefix of a category-sorted file would be one
    category. Round-robin makes a limited run a spread across all of them, which is what
    a smoke run has to be to mean anything.
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


def build() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """The three files' contents, as Python objects."""
    _filler()
    meta = {
        "benchmark": "agent-memory",
        "version": "v1",
        "description": DESCRIPTION,
        "evaluated_at": EVALUATED_AT,
        "predicates": PREDICATES,
        "dimensions": DIMENSIONS,
        "counts": {
            "events": len(_events),
            "questions": len(_questions),
            "scenarios": len({e["scenario"] for e in _events}),
            "entities": len({e["subject"] for e in _events}),
        },
    }
    return meta, list(_events), _interleave(_questions)


def main(argv: list[str] | None = None) -> int:
    """Write the three files. `--out DIR` sends them somewhere else, which is how the
    test suite regenerates and compares without touching the committed copies."""
    args = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    args.add_argument("--out", default=str(OUT), metavar="DIR")
    out = Path(args.parse_args(argv).out)

    meta, events, questions = build()
    out.mkdir(parents=True, exist_ok=True)
    (out / "metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    for name, rows in (("events.jsonl", events), ("questions.jsonl", questions)):
        with (out / name).open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=False) + "\n")
    print(f"{len(events)} events, {len(questions)} questions, "
          f"{meta['counts']['scenarios']} scenarios -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
