"""Many threads writing through one handle leave a store that is intact and consistent.

The threads draw their operations from a seeded random generator: `remember`, a
retraction, `forget` and `delete`, over two users and three predicates. `erase` is not in
the mix; these tests do not cover erasing under concurrency yet. The order the threads
interleave in is not fixed, so the test checks what must hold after any order rather than
one expected state:
- the file passes `check_store_integrity`;
- no single-valued slot has two live values;
- no row ends before it begins;
- every claim a write acknowledged still exists.

`nightly/test_adv_threads_nightly.py` runs the same check with more threads and more
operations.
"""

from __future__ import annotations

import pathlib
import random
import threading
from collections import Counter
from dataclasses import dataclass, field

from memvara import Memvara

from harness import stores
from harness.clock import INSTANTS
from harness.invariants import check_store_integrity

USERS = ("u1", "u2")
POOLS = {"lives_in": ("Berlin", "Paris", "Rome"), "likes": ("tea", "coffee"),
         "collects": ("stamps", "vinyl")}
#: Past instants and the present only. Scheduled values are the reference model's to
#: exercise; here they would only make the single-valued check below harder to state.
VALID_FROM = (None, None, INSTANTS[1], INSTANTS[3])


@dataclass
class Log:
    """What the threads were told happened."""
    added: set[str] = field(default_factory=set)
    errors: list[BaseException] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


def worker(mem: Memvara, seed: int, operations: int, log: Log) -> None:
    rng = random.Random(seed)
    try:
        for _ in range(operations):
            user = rng.choice(USERS)
            predicate = rng.choice(sorted(POOLS))
            kind = rng.choices(["remember", "retract", "forget", "delete"],
                               weights=[6, 1, 1, 1])[0]
            with log.lock:
                known = sorted(log.added)
            if kind == "delete" and known:
                mem.delete(rng.choice(known), user=user)
            elif kind == "forget":
                mem.forget("user", predicate, user=user)
            else:
                receipt = mem.remember(
                    "user", predicate, rng.choice(POOLS[predicate]), user=user,
                    polarity=-1 if kind == "retract" else 1,
                    valid_from=rng.choice(VALID_FROM))
                with log.lock:
                    log.added.update(c.id for c in receipt.added)
    except BaseException as exc:  # noqa: BLE001 - reported by the test
        log.errors.append(exc)


def run_threads(path: pathlib.Path, threads: int, operations: int, seed: int) -> None:
    log = Log()
    with stores.file(path) as mem:
        workers = [threading.Thread(target=worker, args=(mem, seed + n, operations, log))
                   for n in range(threads)]
        for w in workers:
            w.start()
        for w in workers:
            w.join(timeout=120)
        assert not any(w.is_alive() for w in workers), "a thread did not finish"
        assert not log.errors, f"threads raised: {log.errors!r}"
        rows = list(mem.store.iter_claims(None, True))
        ids = {c.id for c in rows}
        for user in USERS:
            live = Counter(c.predicate for c in mem.get_all(user=user))
            assert live["lives_in"] <= 1, f"{user} has {live['lives_in']} live lives_in"
        for c in rows:
            assert c.valid_to is None or c.valid_to >= c.valid_from, (
                f"{c.id} ends at {c.valid_to}, before it starts at {c.valid_from}")
        assert log.added <= ids, f"acknowledged claims are gone: {sorted(log.added - ids)}"
    assert check_store_integrity(path) == []


def test_four_threads_on_one_handle_leave_a_consistent_store(tmp_path: pathlib.Path) -> None:
    run_threads(tmp_path / "s.db", threads=4, operations=25, seed=20260925)
