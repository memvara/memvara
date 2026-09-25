"""Build the committed stores in tests/fixtures/stores/ by running each release's own code.

Run it from the repository root, in a clone that has the release tags:

    python tests/adversarial/upgrade/build_stores.py            # every store
    python tests/adversarial/upgrade/build_stores.py v0.16.0    # one store

For each tag in `golden.RELEASES` it does four things:

1. It extracts the tag's `memvara` package with `git archive` into a temporary directory.
2. It runs this file again as a child process, with `PYTHONPATH` set to that directory,
   so that `write` runs with the release's own code, the hashing embedder and no model.
   The child reports which memvara it imported, and the build stops if that is not the
   release.
3. It compresses the closed store into `tests/fixtures/stores/<tag>/`.
4. It dumps that committed copy with `golden.dump`, which reads with `sqlite3` alone,
   because opening it with memvara would migrate it, and writes the dump to
   `golden.json` beside it.

Two things make a rebuild reproducible, so that the nightly tier can compare it with the
committed store. Claim, episode and document ids come from `uuid.uuid4`, which the child
replaces with a generator seeded with a constant before memvara is imported; every release
mints its ids there, and nothing in memvara reads meaning from an id. And every instant
the program passes is a whole day in 2024, or 1 January 2100. The instants a release
reads from the clock, such as when a fast-path claim was recorded or when a claim was
erased, differ between builds, and `golden.mask_clock` hides exactly those.

The database and the vector file are committed gzip-compressed. Uncompressed, every
store is over the fixture size limit of 256 KB: at SQLite's default page size each of
today's 53 tables and indexes takes at least one 4 KB page, so an empty database is
already 228 KB, and the vector file reserves room for 256 vectors, 512 KB at width 512.
Compressed, a store is about 20 KB. A smaller page size or embedder width would also
fit, but would make the fixtures unlike any store a real user has.
"""

from __future__ import annotations

import gzip
import inspect
import io
import json
import os
import pathlib
import random
import shutil
import subprocess
import sys
import tarfile
import tempfile
import uuid
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

#: The repository root, and tests/, which holds the harness package.
REPO = pathlib.Path(__file__).resolve().parents[3]
TESTS = REPO / "tests"

if __package__:
    from . import golden
else:  # run as a script, so the package is imported through tests/
    sys.path.insert(0, str(TESTS))
    from adversarial.upgrade import golden
#: Seeds the generator that stands in for `uuid.uuid4`, so a rebuild mints the same ids.
SEED = 20240101
#: How long one release may take to write its store, in seconds.
WRITE_SECONDS = 300


class BuildError(RuntimeError):
    """A store could not be built. The message says why."""


def _git(*args: str, env: Mapping[str, str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          env=dict(env), check=False)


def extract(tag: str, dest: pathlib.Path, *, env: Mapping[str, str]) -> pathlib.Path:
    """Extract release `tag`'s `memvara` package into `dest`, so that `PYTHONPATH=dest`
    imports that release. Returns `dest`."""
    done = _git("archive", "--format=tar", tag, "memvara", env=env)
    if done.returncode != 0:
        raise BuildError(f"`git archive {tag}` failed. Is the tag in this clone? "
                         f"`git fetch --tags` fetches it.\n"
                         f"{done.stderr.decode(errors='replace')}")
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(done.stdout)) as archive:
        if hasattr(tarfile, "data_filter"):
            archive.extractall(dest, filter="data")
        else:  # Python before 3.10.12 has no extraction filters
            archive.extractall(dest)
    return dest


def commit_of(tag: str, *, env: Mapping[str, str]) -> str:
    """The commit `tag` points at."""
    done = _git("rev-parse", f"{tag}^{{commit}}", env=env)
    if done.returncode != 0:
        raise BuildError(f"{tag} is not a tag in this clone")
    return done.stdout.decode().strip()


def write_with_release(tag: str, db: pathlib.Path, *, env: Mapping[str, str],
                       encrypted: bool = False) -> dict[str, Any]:
    """Write the fixture program into a new store at `db` with release `tag`'s own code,
    in a child process, and return what the child reported.

    `env` is the child's environment, from `harness.env.child_env`; its PYTHONPATH is
    replaced with the extracted release. With `encrypted`, the store is encrypted with
    the key in `env`'s MEMVARA_DB_KEY.
    """
    with tempfile.TemporaryDirectory(prefix="memvara-release-") as scratch:
        release = extract(tag, pathlib.Path(scratch) / "release", env=env)
        argv = [sys.executable, str(pathlib.Path(__file__).resolve()), "--write", str(db)]
        if encrypted:
            argv.append("--encrypted")
        done = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                              env={**env, "PYTHONPATH": str(release)}, cwd=scratch,
                              timeout=WRITE_SECONDS, check=False)
        if done.returncode != 0:
            raise BuildError(f"{tag} could not write its store (exit status "
                             f"{done.returncode}):\n{done.stderr[-3000:]}")
        reports = [line for line in done.stdout.splitlines() if line.startswith("WROTE ")]
        if not reports:
            raise BuildError(f"{tag}'s writer printed no WROTE line:\n{done.stdout[-3000:]}")
        info: dict[str, Any] = json.loads(reports[-1][len("WROTE "):])
        imported = pathlib.Path(info["memvara"]).resolve()
        if not imported.is_relative_to(release.resolve()):
            raise BuildError(f"the writer imported memvara from {imported}, not from "
                             f"{tag}; an editable install is shadowing PYTHONPATH")
    return info


def _check_closed(store: pathlib.Path) -> None:
    """The release closed its store: no write-ahead log or shared-memory file is left,
    so the database file holds every write, and both side files exist."""
    known = {*golden.COMPRESSED, golden.RECORD, golden.DB + ".lock"}
    left = sorted(p.name for p in store.iterdir() if p.name not in known)
    if left:
        raise BuildError(f"the release left {left} beside its store, so the database "
                         "file may not hold every write")
    missing = [name for name in (*golden.COMPRESSED, golden.RECORD)
               if not (store / name).exists()]
    if missing:
        raise BuildError(f"the release wrote no {missing}")


def build(tag: str, root: pathlib.Path, *, env: Mapping[str, str]) -> dict[str, Any]:
    """Write release `tag`'s store and its golden record into `root/<tag>/`, replacing
    any that are there, and return the record."""
    if tag not in golden.RELEASES:
        raise BuildError(f"{tag} is not in golden.RELEASES")
    out = root / tag
    with tempfile.TemporaryDirectory(prefix="memvara-store-") as scratch:
        store = pathlib.Path(scratch) / "store"
        store.mkdir()
        info = write_with_release(tag, store / golden.DB, env=env)
        _check_closed(store)
        out.mkdir(parents=True, exist_ok=True)
        for name in golden.COMPRESSED:
            (out / f"{name}.gz").write_bytes(
                gzip.compress((store / name).read_bytes(), compresslevel=9, mtime=0))
        shutil.copyfile(store / golden.RECORD, out / golden.RECORD)
        # Dumped from the committed files, the way every test reads them.
        db = golden.unpack(tag, pathlib.Path(scratch) / "committed", root=root)
        record = {"tag": tag, "commit": commit_of(tag, env=env),
                  "schema_version": golden.schema_version(db), "data": golden.dump(db)}
    if {info["schema"], record["schema_version"]} != {golden.RELEASES[tag]}:
        raise BuildError(f"{tag} wrote schema version {record['schema_version']}, and "
                         f"golden.RELEASES says {golden.RELEASES[tag]}")
    (out / "golden.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n",
                                     encoding="utf-8", newline="\n")
    return record


def write(db: str, *, encrypted: bool = False) -> dict[str, Any]:
    """Write the fixture program into a new store at `db`, with whichever memvara this
    process imports; the builder arranges for that to be one release's own code.

    It runs only in the builder's child process, because it replaces `uuid.uuid4`
    before it imports memvara. A step that needs a feature the release lacks is skipped;
    the release's own signatures say which features it has. Each comment names the
    part of the schema the step gives the migrations to carry.
    """
    generator = random.Random(SEED)
    uuid.uuid4 = lambda: uuid.UUID(int=generator.getrandbits(128), version=4)
    warnings.simplefilter("ignore")  # an old release's warnings say nothing about its store

    import memvara  # noqa: PLC0415 - imported only after uuid4 is replaced
    from memvara import Memvara, NullLLM  # noqa: PLC0415
    from memvara.embed import HashingEmbedder  # noqa: PLC0415
    from memvara.store.sqlite import SCHEMA_VERSION  # noqa: PLC0415

    def takes(method: str, parameter: str) -> bool:
        return parameter in inspect.signature(getattr(Memvara, method)).parameters

    security: dict[str, Any] = {}
    if encrypted:
        security = {"encryption": True,
                    "key_env": {"MEMVARA_DB_KEY": os.environ["MEMVARA_DB_KEY"]}}

    def open_store(**options: Any) -> Any:
        return Memvara(db, embedder=HashingEmbedder(dim=512), llm=NullLLM(), **security,
                       **options)

    mem = open_store()

    def remember(predicate: str, obj: str, day: int, **options: Any) -> str | None:
        """Assert `user <predicate> <obj>`, true and recorded at `day`, for USER unless
        `options` names another scope. The id of the claim stored or reinforced."""
        options.setdefault("user", golden.USER)
        receipt = mem.remember("user", predicate, obj, valid_from=golden.at(day),
                               recorded_at=golden.at(day), **options)
        claims = list(receipt.added) + list(receipt.reinforced)
        return claims[0].id if claims else None

    # A single-valued slot whose first value is superseded.
    remember("lives_in", "Berlin", 0)
    remember("lives_in", "Paris", 30)
    # A multi-valued slot. One value is ended here, and the other is retracted below.
    remember("likes", "green tea", 1)
    coffee = remember("likes", "black coffee", 2)
    mem.delete(coffee, at=golden.at(40), user=golden.USER,
               **({"close": "ended"} if takes("delete", "close") else {}))
    # A retired slot.
    remember("has_pet", "a cat named Tom", 3)
    mem.forget("user", "has_pet", at=golden.at(41), user=golden.USER,
               **({"close": "retired"} if takes("forget", "close") else {}))
    # A claim that cites the turn it came from: the provenance edge version 5 indexes.
    turn = mem.add("We moved the whole team into the new office downtown.",
                   user=golden.USER, ts=golden.at(4))
    remember("works_at", "Acme Corp", 5, sources=[turn.episode_ids[0]])
    # Two turns: the fast path extracts a claim from the first and nothing from the
    # second. The extracted claim's `recorded_at` comes from the clock.
    mem.add("I live in Lisbon", user=golden.USER, ts=golden.at(50))
    mem.add("The quarterly report is due on Friday.", user=golden.USER, ts=golden.at(51))
    # Other scopes: another user, a session, an agent and another tenant.
    remember("lives_in", "Madrid", 7, user="u2")
    remember("prefers", "dark mode", 8, session="s1")
    remember("prefers", "short answers", 9, agent="a1")
    remember("lives_in", "Oslo", 10, tenant="acme")
    # Names the entity fold treats differently from version 16 on, and accented names,
    # all of which versions 6, 12 and 16 re-key.
    for day, language in enumerate(("C++", "C#", "C"), start=11):
        remember("knows_language", language, day)
    for day, place in enumerate(("Zürich", "São Paulo", "Kraków"), start=21):
        remember("visited", place, day)
    remember("likes_band", "the the band", 24)
    # Caller metadata, and a retraction of one value.
    remember("visited", "Rome", 14, note="written by the fixture")
    remember("likes", "green tea", 60, polarity=-1)
    if hasattr(Memvara, "link"):
        # 0.15.0 and later: a link (version 13), a document (14), an expiry (15), and a
        # replacement named by the caller.
        florence = remember("visited", "Florence", 15)
        uffizi = remember("visited", "the Uffizi gallery in Florence", 16)
        mem.link(uffizi, florence, relation="extends", user=golden.USER)
        remember("subscribes_to", "a jazz magazine", 17, expires_at=golden.FAR_FUTURE,
                 expire_reason="the trial ends")
        green = remember("favorite_color", "green", 18)
        remember("favorite_color", "blue", 19, replaces=green, reason="the user said blue")
        mem.add_document("Onboarding notes. The staging database is PostgreSQL 16.",
                         title="Onboarding notes", custom_id="onboarding", extract=False,
                         user=golden.USER)
    mem.close()
    if takes("__init__", "project"):
        # 0.12.0 and later: an undeclared predicate is recorded against the project the
        # store was opened in (version 12).
        mem = open_store(project=golden.PROJECT)
        remember("uses_database", "PostgreSQL", 20)
        mem.close()
    # Last, so that no later write merges the text index. Before version 7 the erased
    # claim's words then stay in the index's shadow table, which is what the version 7
    # migration has to clear.
    mem = open_store()
    secret = remember("codeword", golden.ERASED_WORD, 6)
    if not mem.erase(secret, user=golden.USER):
        raise RuntimeError("the release did not erase the claim it was asked to erase")
    mem.close()
    return {"memvara": memvara.__file__, "schema": SCHEMA_VERSION,
            "version": getattr(memvara, "__version__", None)}


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["--write"]:
        info = write(args[1], encrypted="--encrypted" in args[2:])
        print("WROTE " + json.dumps(info), flush=True)
        return 0
    unknown = [tag for tag in args if tag not in golden.RELEASES]
    if unknown:
        print(f"unknown tags {unknown}; choose from {list(golden.TAGS)}", file=sys.stderr)
        return 2
    from harness.env import child_env  # noqa: PLC0415 - not needed by the writer child

    with tempfile.TemporaryDirectory(prefix="memvara-build-home-") as home:
        env = child_env(pathlib.Path(home))
        for tag in args or golden.TAGS:
            record = build(tag, golden.FIXTURES, env=env)
            data = record["data"]
            print(f"{tag}: schema version {record['schema_version']}, "
                  f"{len(data['claims'])} claims, {len(data['episodes'])} episodes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
