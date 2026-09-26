"""A child process for the crash tests: it runs a program against a store and can stop at
a named point inside memvara.

Run it as `python crash_child.py`, with the program as one line of JSON on standard input:

    {"db": "/path/store.db", "user": "u1", "setup": [op, ...],
     "point": "after-claim", "action": op, "hold": false}

`point` and `action` may be null. An op is `[name, arguments]`, and the names are listed
in `OPS`. In `erase`, the argument `{"ref": i}` stands for the first claim id that setup
op `i` produced.

The child prints one line per event, and flushes each at once:
- `ACK {"index": i, "ids": [...]}` after setup op `i` returns;
- `POINT <name>` when the action reaches the point;
- `DONE {"ids": [...]}` when the action returns, or after the setup when there is none.
  At the point `after-commit`, `DONE` comes first, then `POINT`;
- `ERROR <type>: <message>` when an op raises, and the child then exits with status 3.

At the point, the child sleeps until it is killed or, when `hold` is true, reads one more
line from standard input and carries on if that line is `go`.

It imports only memvara and the standard library, so it runs from any directory. The
test process never imports it to run a program, because the pauses patch memvara in the
process that installs them.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Callable

#: The ten points the crash tests kill a child at.
POINTS = ("after-episode", "after-claim", "after-vector", "erase-before-delete",
          "between-migrations", "after-commit", "vecs-growth", "fingerprint-write",
          "encrypt-between-renames", "document-before-claims")
#: The other points a child can stop at: before its first claim write, inside an
#: uncommitted batch after every op in it, and before its action starts.
PAUSES = ("before-claim", "inside-batch", "before-action")

OPS = ("remember", "add", "erase", "add_document", "batch", "open", "encrypt", "fill")
#: `fill` gives up after this many seconds, so a limit that never trips fails the test
#: with a message instead of leaving a child writing until the test's own timeout.
FILL_SECONDS = 60.0


def emit(kind: str, payload: str) -> None:
    print(f"{kind} {payload}", flush=True)


class Program:
    def __init__(self, spec: dict[str, Any]) -> None:
        self.db: str = spec["db"]
        self.user: str = spec["user"]
        self.point: str | None = spec.get("point")
        self.hold: bool = bool(spec.get("hold"))
        self.results: list[list[str]] = []
        self._mem: Any = None
        if self.point is not None and self.point not in POINTS + PAUSES:
            raise ValueError(f"unknown point {self.point!r}; use one of {POINTS + PAUSES}")

    # -- the store, opened only when an op needs it -----------------------------------

    def mem(self) -> Any:
        if self._mem is None:
            from memvara import Memvara, NullLLM
            from memvara.embed import HashingEmbedder
            self._mem = Memvara(self.db, embedder=HashingEmbedder(dim=512), llm=NullLLM())
        return self._mem

    def close(self) -> None:
        if self._mem is not None:
            self._mem.close()
            self._mem = None

    # -- stopping ----------------------------------------------------------------------

    def stop(self) -> None:
        emit("POINT", str(self.point))
        if not self.hold:
            while True:
                time.sleep(3600)
        if sys.stdin.readline().strip() != "go":
            sys.exit(4)

    def install(self) -> None:
        """Patch memvara so the action stops at `self.point`. Every patch fires once."""
        point = self.point
        if point in (None, "after-commit", "inside-batch", "before-action"):
            return      # these stop in `run`, around the action
        import memvara.documents.service as service
        import memvara.embed.fingerprint as fingerprint
        from memvara.store import sqlite

        targets: dict[str, tuple[Any, str, str]] = {
            "after-episode": (sqlite.SQLiteStore, "add_episode", "after"),
            "after-claim": (sqlite.SQLiteStore, "put_claim", "after"),
            "before-claim": (sqlite.SQLiteStore, "put_claim", "before"),
            "after-vector": (sqlite.SQLiteStore, "_set_vector", "after"),
            "erase-before-delete": (sqlite.SQLiteStore, "_erase_row", "before"),
            # Every migration runs in one transaction that commits at the end, so any
            # step between the first and the last interrupts the same way.
            "between-migrations": (sqlite.SQLiteStore, "_migrate_to_v7", "before"),
            "vecs-growth": (sqlite._VecIndex, "_remap", "before"),
            "encrypt-between-renames": (os, "replace", "after"),
            "document-before-claims": (service.DocumentService, "_finish", "before"),
        }
        if point == "fingerprint-write":
            self._tear_fingerprint(fingerprint)
            return
        owner, name, when = targets[str(point)]
        original: Callable[..., Any] = getattr(owner, name)
        fired = [False]

        def patched(*args: Any, **kwargs: Any) -> Any:
            if fired[0] or not self._is_the_target(point, args):
                return original(*args, **kwargs)
            fired[0] = True
            if when == "before":
                self.stop()
                return original(*args, **kwargs)
            result = original(*args, **kwargs)
            self.stop()
            return result

        setattr(owner, name, patched)

    @staticmethod
    def _is_the_target(point: str | None, args: tuple[Any, ...]) -> bool:
        """`os.replace` is global, so the rename point fires only on `encrypt_store`'s
        own renames, which move its `.encrypting-` temporary copy into place."""
        if point != "encrypt-between-renames":
            return True
        return bool(args) and ".encrypting-" in os.fspath(args[0])

    def _tear_fingerprint(self, fingerprint: Any) -> None:
        """Write the first few pieces of the sidecar's JSON, as a crash in the middle of
        `json.dump` would, then stop. The real write goes to a temporary file that is
        renamed over the record afterwards, so the torn pieces land in that file."""
        real_json = fingerprint.json

        class Torn:
            def __getattr__(self, name: str) -> Any:
                return getattr(real_json, name)

            def dump(inner, obj: Any, fh: Any, **kwargs: Any) -> None:
                chunks = list(real_json.JSONEncoder(**kwargs).iterencode(obj))
                for chunk in chunks[:4]:
                    fh.write(chunk)
                fh.flush()
                os.fsync(fh.fileno())
                self.stop()

        fingerprint.json = Torn()

    # -- ops -----------------------------------------------------------------------------

    def ref(self, value: Any) -> Any:
        if isinstance(value, dict) and set(value) == {"ref"}:
            return self.results[int(value["ref"])][0]
        return value

    def run_op(self, op: list[Any]) -> list[str]:
        name, args = op[0], dict(op[1]) if len(op) > 1 else {}
        if name not in OPS:
            raise ValueError(f"unknown op {name!r}; use one of {OPS}")
        if name == "remember":
            receipt = self.mem().remember(args.pop("subject", "user"), args.pop("predicate"),
                                          args.pop("object"), user=self.user, **args)
            return [c.id for c in receipt.added] + [c.id for c in receipt.reinforced]
        if name == "add":
            receipt = self.mem().add(args.pop("text"), user=self.user, **args)
            return list(receipt.episode_ids) + [c.id for c in receipt.added]
        if name == "erase":
            done = self.mem().erase(self.ref(args.pop("id")), user=self.user, **args)
            return ["yes" if done else "no"]
        if name == "add_document":
            doc = self.mem().add_document(args.pop("content"), user=self.user, **args)
            return [doc.id]
        if name == "batch":
            ids: list[str] = []
            with self.mem().store.batch():
                for inner in args["ops"]:
                    ids.extend(self.run_op(inner))
                if self.point == "inside-batch":
                    self.stop()
            return ids
        if name == "open":
            self.mem()
            return []
        if name == "fill":
            return self._fill(int(args["limit"]))
        if name == "encrypt":
            # The store must be closed, and the key comes from MEMVARA_DB_KEY.
            from memvara.store.encryption import encrypt_store
            self.close()
            encrypt_store(self.db)
            return []
        raise AssertionError(f"op {name!r} is in OPS but has no branch here")

    def _fill(self, limit: int) -> list[str]:
        """Lower the largest file this process may write to `limit` bytes, then write
        until a write fails. Each write that succeeds is acknowledged; the failure is
        raised, so the child reports it and exits with status 3. POSIX only."""
        import resource
        import signal
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)   # a failed write, not a dead process
        mem = self.mem()
        resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
        ids: list[str] = []
        deadline = time.monotonic() + FILL_SECONDS
        for i in range(1_000_000):
            if time.monotonic() > deadline:
                raise RuntimeError(f"the disk never filled: {i} writes in {FILL_SECONDS:g} s")
            receipt = mem.remember("user", "visited", f"q{i:06d}x", user=self.user)
            ids.extend(c.id for c in receipt.added)
            emit("ACK", json.dumps({"index": i, "ids": [c.id for c in receipt.added]}))
        return ids

    def run(self, setup: list[list[Any]], action: list[Any] | None) -> None:
        for index, op in enumerate(setup):
            ids = self.run_op(op)
            self.results.append(ids)
            emit("ACK", json.dumps({"index": index, "ids": ids}))
        if self.point == "before-action":
            self.stop()     # lets a test start several children's actions at once
        self.install()
        ids = self.run_op(action) if action is not None else []
        emit("DONE", json.dumps({"ids": ids}))
        if self.point == "after-commit":
            self.stop()     # after DONE, so the test knows the ids of what committed
        self.close()


def main() -> int:
    try:
        spec = json.loads(sys.stdin.readline())
        program = Program(spec)
        program.run(spec.get("setup") or [], spec.get("action"))
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - reported to the parent, which fails the test
        emit("ERROR", f"{type(exc).__name__}: {exc}")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
