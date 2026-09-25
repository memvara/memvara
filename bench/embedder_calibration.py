"""Where each cosine threshold in `embed/calibration.py` sits in each embedding space.

Run:  PYTHONPATH=. python3 bench/embedder_calibration.py [--model ID ...] [--show]

`--model hashing` measures `HashingEmbedder`, and `--show` prints every merge pair with
its cosine.

Needs sentence-transformers (`pip install 'memvara[local-embed]'`) and the LongMemEval
`s` file (`python3 bench/longmemeval.py --download --dataset s`), which supplies turns
that the invented values below have nothing to do with.

Two thresholds read a cosine, and a cosine is not portable between models:

* **The grounding rescue** (`write/pipeline.py`) keeps a model-proposed claim that shares
  no word with its source when its best 1,200-character chunk cosine reaches
  `grounding_rescue`. It should keep paraphrases and refuse inventions. The pairs: 15
  paraphrases, each of a fact its source turn states in other words, and 15 invented
  values of the kind a small model emits on a turn that states no such fact, each
  against 20 turns it has nothing to do with, 300 pairs.
* **The duplicate merge** (`consolidate/merge.py`) folds two live claims in one slot
  when their cosine reaches `merge`. It should fold restatements and never two different
  values: in a slot that holds many values, both are true, and a merge retires one. The
  pairs: 69 that differ in one value and 26 restatements of one value. The first 14
  different values are sentences written for 0.99 on bge-small. The rest are claims
  rendered as the store renders them, `subject predicate object`: 55 different values
  (dates, versions, quantities, codes one character apart, 8 with no number to tell them
  apart, and 12 that hold the same numbers a letter or a word apart) and 26 restatements
  whose `value_key` differs, so the write path's exact-duplicate check does not already
  fold them and only the merge can. The 8 restatements measured before them could never
  reach the merge: 6 differ only in case, punctuation or a leading article, which
  `value_key` folds, and 2 change the predicate, which puts them in two slots.
* **The near-duplicate check** (`write/pipeline.py`) reads a new turn as a restatement of
  the nearest claim, and extracts nothing from it, when their cosine reaches `merge` and
  the two hold the same numbers. A turn worded like a claim embeds exactly as that claim
  would, so the merge's pairs measure the check for those turns. The pairs added for it
  are turns a person writes, in the first person: 8 that repeat a claim's value and 8
  that state another value, each against the claim.

The report counts the merges at each threshold twice: all of them, and those between two
values holding the same numbers in the same order, which are the only ones left once the
merge refuses two values whose numbers differ.

The original eval behind 0.40, 33 inventions from two 4B-class models over real turns,
is not in this repository. These pairs are a reconstruction written for this script, so
a threshold taken from them should be read as that.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evalkit as ek  # noqa: E402
import longmemeval as lme  # noqa: E402

# The merge's own reading of a value's numbers, so the report counts what it refuses.
from memvara.embed.calibration import numbers  # noqa: E402
from memvara.embed import HashingEmbedder  # noqa: E402
from memvara.types import Claim, Scope  # noqa: E402

CHUNK = 1200
MODELS = ("sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "hashing")
SCOPE = Scope("bench", "user")

# (source turn, a fact it states, in words it does not use)
PARAPHRASES = [
    ("I prefer short answers, please don't write essays.", "keep replies brief"),
    ("My flight to Lisbon leaves on Friday morning.",
     "travelling to Portugal at the end of the week"),
    ("I've been vegetarian for about six years now.", "does not eat meat"),
    ("We adopted a greyhound from the rescue last spring.", "has a retired racing dog"),
    ("I'm allergic to penicillin, found out the hard way.", "cannot take that antibiotic"),
    ("Our team moved the standup to 9:30 because of the new hires in Berlin.",
     "daily sync starts half past nine"),
    ("I usually cycle to the office unless it's raining.", "commutes by bike"),
    ("I just finished my PhD in marine biology.", "holds a doctorate studying ocean life"),
    ("My daughter starts kindergarten in September.", "child begins school this autumn"),
    ("I can't stand cilantro, it tastes like soap to me.", "dislikes coriander"),
    ("We're renting a flat near the canal in Amsterdam.", "lives in a Dutch city by the water"),
    ("I play bass in a jazz trio on weekends.", "performs music as a hobby"),
    ("The deploy failed because the database migration timed out.",
     "release broke on a slow schema change"),
    ("I switched from an iPhone to a Pixel last month.", "now uses an Android phone"),
    ("I get migraines when I skip coffee.", "headaches without caffeine"),
]

# Placeholder values a small model emits on a turn that states no such fact.
INVENTIONS = ["Acme", "Acme Corp", "unknown", "software engineer at Google",
              "a golden retriever named Max", "pollen allergy", "lives in New York",
              "likes pizza", "John Smith", "enjoys hiking and photography",
              "prefers email communication", "works remotely", "has two children",
              "favorite color is blue", "speaks Spanish"]

# One slot, two different values: must never merge.
DIFFERENT = [
    ("server runs on port 8080", "server runs on port 8081"),
    ("app version is 1.2.3", "app version is 1.2.4"),
    ("user's room is 101", "user's room is 102"),
    ("user's phone extension is 4521", "user's phone extension is 4522"),
    ("appointment is on 2023-05-01", "appointment is on 2023-05-02"),
    ("dose is 5 mg", "dose is 10 mg"),
    ("budget is $50", "budget is $60"),
    ("user's locker code is 3481", "user's locker code is 3418"),
    ("meeting is in room B12", "meeting is in room B21"),
    ("user's flight is LH 400", "user's flight is LH 401"),
    ("project deadline is March 3", "project deadline is March 13"),
    ("user lives at 12 Main Street", "user lives at 21 Main Street"),
    ("the API key prefix is sk-ab", "the API key prefix is sk-ba"),
    ("user's employee id is E-10293", "user's employee id is E-10239"),
]

# One slot, two values, as `(predicate, one object, another)`: must never merge.
DIFFERENT_CLAIMS = [
    # dates
    ("has_appointment_on", "2024-03-14", "2024-03-15"),
    ("was_born_on", "1990-07-04", "1990-07-14"),
    ("lease_ends_on", "31 August 2025", "31 August 2026"),
    ("started_job_in", "2019", "2021"),
    ("conference_is_on", "June 5", "June 6"),
    ("flies_out_on", "12/03/2024", "13/03/2024"),
    ("passport_expires_in", "2027", "2028"),
    ("standup_is_at", "9:30", "10:30"),
    # versions
    ("runs_python", "3.11", "3.12"),
    ("uses", "Postgres 15", "Postgres 16"),
    ("app_targets", "iOS 17.2", "iOS 17.4"),
    ("pinned", "numpy 1.26.4", "numpy 1.26.2"),
    ("cluster_runs", "Kubernetes 1.28", "Kubernetes 1.29"),
    ("library_version_is", "0.15.0", "0.15.1"),
    ("uses", "Node 18", "Node 20"),
    # quantities
    ("monthly_rent_is", "1200 euros", "1250 euros"),
    ("runs_each_morning", "5 km", "8 km"),
    ("team_size_is", "12 engineers", "14 engineers"),
    ("salary_is", "85000", "95000"),
    ("sleeps", "7 hours a night", "6 hours a night"),
    ("timeout_is", "30 seconds", "60 seconds"),
    ("daughter_age_is", "7", "9"),
    ("takes", "2 tablets a day", "3 tablets a day"),
    ("batch_size_is", "32", "64"),
    ("weight_goal_is", "70 kg", "75 kg"),
    # codes one character apart
    ("order_number_is", "88213", "88214"),
    ("wifi_password_is", "kiwi2024", "kiwi2025"),
    ("booking_reference_is", "QX7F2L", "QX7F2K"),
    ("postcode_is", "10115", "10117"),
    ("error_code_is", "E1042", "E1043"),
    ("tracks_ticket", "JIRA-4411", "JIRA-4412"),
    ("seat_is", "14C", "14D"),
    ("deployed_commit", "a1b2c3d", "a1b2c3e"),
    ("car_plate_is", "B-MX 4421", "B-MX 4427"),
    ("gate_is", "A23", "A32"),
    # no number to tell them apart
    ("blood_type_is", "A positive", "A negative"),
    ("injured", "left knee", "right knee"),
    ("prefers_meetings_in", "the morning", "the evening"),
    ("gym_day_is", "Monday", "Tuesday"),
    ("is_allergic_to", "cats", "dogs"),
    ("owns", "iPhone 15", "iPhone 15 Pro"),
    ("booking_reference_is", "QXAFBL", "QXAFBK"),
    ("drinks", "green tea", "black tea"),
    # the same numbers, a letter or a word apart
    ("availability_zone_is", "us-east-1a", "us-east-1b"),
    ("instance_type_is", "m5.large", "m5.xlarge"),
    ("seat_is", "22A", "22F"),
    ("booking_reference_is", "KLM7QX", "KLM7QZ"),
    ("office_is_in", "the north wing", "the south wing"),
    ("parks_on", "level B", "level C"),
    ("prefers_seat", "aisle", "window"),
    ("default_branch_is", "main", "master"),
    ("currency_is", "EUR", "USD"),
    ("role_is", "admin", "viewer"),
    ("temperature_unit_is", "Celsius", "Fahrenheit"),
    ("doctor_is", "Dr. Smith", "Dr. Smyth"),
]

# One value, written twice in words the entity fold does not unify: should merge.
RESTATED_CLAIMS = [
    ("lives_in", "Berlin", "Berlin, Germany"),
    ("lives_in", "NYC", "New York City"),
    ("lives_in", "San Francisco", "SF"),
    ("lives_in", "the UK", "United Kingdom"),
    ("was_born_in", "Munich", "München"),
    ("works_at", "Meta", "Meta Platforms"),
    ("uses", "Postgres", "PostgreSQL"),
    ("uses", "JS", "JavaScript"),
    ("uses_editor", "VS Code", "Visual Studio Code"),
    ("uses_os", "macOS", "Mac OS"),
    ("prefers", "dark mode", "dark theme"),
    ("prefers", "short answers", "brief answers"),
    ("prefers", "email", "e-mail"),
    ("is_allergic_to", "peanuts", "peanut"),
    ("likes", "cats", "cat"),
    ("favorite_color_is", "grey", "gray"),
    ("job_title_is", "software engineer", "software developer"),
    ("speaks", "Spanish", "the Spanish language"),
    ("studied", "computer science", "CS"),
    ("commutes_by", "bike", "bicycle"),
    ("timezone_is", "CET", "Central European Time"),
    ("dose_is", "5 mg", "5mg"),
    ("standup_is_at", "9:30", "09:30"),
    ("runs_version", "1.2.3", "v1.2.3"),
    ("appointment_is_on", "2023-05-01", "May 1, 2023"),
    ("salary_is", "85000", "85,000"),
]


# (a turn in a person's own words, the claim it repeats): the near-duplicate check's
# restatements.
FIRST_PERSON_SAME = [
    ("I have an appointment on 2023-05-01.", "user has appointment on 2023-05-01"),
    ("My booking reference is KLM7QX.", "user booking reference is KLM7QX"),
    ("I live in Berlin.", "user lives in Berlin"),
    ("I work at Acme.", "user works at Acme"),
    ("My flight is LH 400.", "user's flight is LH 400"),
    ("I pinned numpy 1.26.4", "user pinned numpy 1.26.4"),
    ("My lease ends on 31 August 2025.", "user lease ends on 31 August 2025"),
    ("I track ticket JIRA-4411.", "user tracks ticket JIRA-4411"),
]

# (a turn in a person's own words, a claim holding another value): must never be read as a
# restatement of it.
FIRST_PERSON_OTHER = [
    ("I have an appointment on 2023-05-02.", "user has appointment on 2023-05-01"),
    ("My appointment is on 2023-05-02", "user has appointment on 2023-05-01"),
    ("I pinned numpy 1.26.2 in the project.", "user pinned numpy 1.26.4"),
    ("I pinned numpy 1.26.2", "user pinned numpy 1.26.4"),
    ("My booking reference is KLM7QZ.", "user booking reference is KLM7QX"),
    ("My flight is LH 401.", "user's flight is LH 400"),
    ("My lease ends on 31 August 2026.", "user lease ends on 31 August 2025"),
    ("I track ticket JIRA-4412 now.", "user tracks ticket JIRA-4411"),
]


def unrelated_turns() -> list[str]:
    """Twenty real turns, over 200 characters, that none of the inventions describe."""
    episodes = json.loads((ROOT / "tests/fixtures/phi4_spike/episodes.json").read_text())
    turns = [e["content"] for e in episodes]
    items = lme.load(ek.require(ek.LME_S), limit=3)
    turns += [t.text for item in items for s in item.sessions[:4] for t in s[:3]]
    return [t for t in turns if len(t) > 200][:20]


def rendered(pairs: list[tuple[str, str, str]]) -> list[tuple[str, str]]:
    """Each `(predicate, object, object)` as the two claim texts the store embeds.

    Refuses a pair whose two claims share a `value_key`: the write path already treats
    those as one value, so the merge never sees them and they would measure nothing.
    """
    out = []
    for predicate, a, b in pairs:
        one, other = (Claim(subject="user", predicate=predicate, object=obj, scope=SCOPE)
                      for obj in (a, b))
        if one.value_key == other.value_key:
            raise ValueError(f"{a!r} and {b!r} are one value to the write path")
        out.append((one.text, other.text))
    return out


def measure(model_id: str, sources: list[str]) -> dict[str, np.ndarray]:
    if model_id == "hashing":
        embedder = HashingEmbedder()

        def unit(texts: list[str]) -> np.ndarray:
            v = np.asarray(embedder.encode(texts), dtype=np.float32)
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            return v / np.where(norms > 0.0, norms, 1.0)
    else:
        from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

        model = SentenceTransformer(model_id, device="cpu")

        def unit(texts: list[str]) -> np.ndarray:
            return np.asarray(model.encode(texts, normalize_embeddings=True),
                              dtype=np.float32)

    def best_chunk(obj: str, source: str) -> float:
        chunks = [source[i:i + CHUNK] for i in range(0, max(len(source), 1), CHUNK)]
        v = unit([obj] + chunks)
        return float(max(v[1:] @ v[0]))

    def pair(a: str, b: str) -> float:
        v = unit([a, b])
        return float(v[0] @ v[1])

    return {
        "paraphrase": np.array([best_chunk(obj, src) for src, obj in PARAPHRASES]),
        "invention": np.array([best_chunk(obj, src) for obj in INVENTIONS for src in sources]),
        "different": np.array([pair(a, b) for a, b in DIFFERENT_PAIRS]),
        "restated": np.array([pair(a, b) for a, b in RESTATED_PAIRS]),
        "first_person_same": np.array([pair(a, b) for a, b in FIRST_PERSON_SAME]),
        "first_person_other": np.array([pair(a, b) for a, b in FIRST_PERSON_OTHER]),
    }


def same_numbers(pairs: list[tuple[str, str]]) -> np.ndarray:
    """Whether each pair's two texts hold the same numbers in the same order: the pairs
    the merge can still fold once it refuses two values whose numbers differ."""
    return np.array([numbers(a) == numbers(b) for a, b in pairs])


DIFFERENT_PAIRS = DIFFERENT + rendered(DIFFERENT_CLAIMS)
RESTATED_PAIRS = rendered(RESTATED_CLAIMS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", action="append", default=[],
                        help="a sentence-transformers model id, or `hashing`; repeatable")
    parser.add_argument("--show", action="store_true",
                        help="print every merge pair with its cosine, highest first")
    args = parser.parse_args()
    sources = unrelated_turns()
    for model_id in args.model or MODELS:
        m = measure(model_id, sources)
        inv, para = m["invention"], m["paraphrase"]
        print(f"\n  {model_id}")
        print(f"    inventions: median {np.median(inv):.3f}, highest {inv.max():.3f}; "
              f"paraphrases: median {np.median(para):.3f}, lowest {para.min():.3f}")
        print(f"    {'rescue at':<12} {'inventions kept':>16} {'paraphrases kept':>17}")
        for t in (0.40, 0.55, 0.60, 0.65):
            print(f"    {t:<12.2f} {np.mean(inv >= t):>15.1%} {np.mean(para >= t):>16.1%}")
        diff, same = m["different"], m["restated"]
        dn, sn = same_numbers(DIFFERENT_PAIRS), same_numbers(RESTATED_PAIRS)
        print(f"    different values: highest {diff.max():.5f}, and with the same "
              f"numbers {diff[dn].max():.5f}")
        print(f"    {'merge at':<10} {'different merged':>17} {'same numbers':>14} "
              f"{'restated merged':>16} {'same numbers':>14}")
        for t in (0.97, 0.98, 0.985, 0.99, 0.995):
            print(f"    {t:<10} {int((diff >= t).sum()):>11} of {len(diff):<3} "
                  f"{int((diff[dn] >= t).sum()):>8} of {int(dn.sum()):<3} "
                  f"{int((same >= t).sum()):>10} of {len(same):<3} "
                  f"{int((same[sn] >= t).sum()):>8} of {int(sn.sum())}")
        fs, fo = m["first_person_same"], m["first_person_other"]
        print(f"    near-duplicate check, first-person turns: {fs.min():.3f}-{fs.max():.3f} "
              f"against the claim they repeat, {fo.min():.3f}-{fo.max():.3f} against a "
              f"claim holding another value")
        if args.show:
            for label, pairs, scores in (("different", DIFFERENT_PAIRS, diff),
                                         ("restated", RESTATED_PAIRS, same)):
                for (a, b), s in sorted(zip(pairs, scores), key=lambda x: -x[1]):
                    print(f"      {label:<9} {s:.5f}  {a!r} / {b!r}")
    print()
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by hand
    sys.exit(main())
