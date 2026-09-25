"""A store's two side files, damaged the way a crash, a full disk or a careless copy can
damage them: the vector file `<db>.vecs` and the embedder record `<db>.embedder.json`.

The database is the authority for vectors, so a damaged vector file must be rebuilt from
it and every search must still find every claim. The embedder record is the only thing
that can tell two embedders of the same width apart, so damage to it must not let an
embedder change go unnoticed.
"""

from __future__ import annotations

import pathlib
import warnings
from collections.abc import Callable

import pytest

from memvara import EmbedderMismatchError, Memvara, NullLLM
from memvara.embed import HashingEmbedder

from harness import known_bugs, stores

USER = "u1"
PLACES = ("Berlin", "Paris", "Rome", "Lisbon", "Oslo", "Vienna")


def written(path: pathlib.Path) -> dict[str, str]:
    """A closed store holding one `visited` claim per place, by id."""
    with stores.file(path) as mem:
        return {mem.remember("user", "visited", place, user=USER).added[0].id: place
                for place in PLACES}


def halve(vecs: pathlib.Path) -> None:
    data = vecs.read_bytes()
    vecs.write_bytes(data[: len(data) // 2])


def scribble_header(vecs: pathlib.Path) -> None:
    data = bytearray(vecs.read_bytes())
    data[:16] = b"\xff" * 16
    vecs.write_bytes(bytes(data))


def delete(vecs: pathlib.Path) -> None:
    vecs.unlink()


@pytest.mark.parametrize("damage", [halve, scribble_header, delete],
                         ids=["truncated", "header-overwritten", "deleted"])
def test_a_damaged_vector_file_is_rebuilt_and_every_claim_is_found(
        damage: Callable[[pathlib.Path], None], tmp_path: pathlib.Path) -> None:
    db = tmp_path / "s.db"
    claims = written(db)
    damage(pathlib.Path(str(db) + ".vecs"))
    with stores.file(db) as mem:
        for claim_id, place in claims.items():
            vector = mem.embedder.encode([f"user visited {place}"])[0]
            scope = mem.store.get_claim(claim_id).scope
            nearest = [i for i, _ in mem.store.vector_search(vector, [scope], 1)]
            assert nearest == [claim_id], f"the vector index lost {place!r}: {nearest}"


def other_embedder() -> HashingEmbedder:
    """The same width as the suite's embedder, a different vector space."""
    return HashingEmbedder(dim=512, ngram=(2, 4))


def test_an_embedder_change_is_noticed_while_the_record_is_intact(
        tmp_path: pathlib.Path) -> None:
    """The control case: with the record intact, opening the store with another
    embedder of the same width warns, as `Memvara._check_embedder` promises."""
    db = tmp_path / "s.db"
    written(db)
    with pytest.warns(Warning, match="unrelated vector spaces"):
        Memvara(str(db), embedder=other_embedder(), llm=NullLLM()).close()


@known_bugs.xfail("B13")
def test_an_embedder_change_is_noticed_after_the_record_is_damaged(
        tmp_path: pathlib.Path) -> None:
    db = tmp_path / "s.db"
    written(db)
    pathlib.Path(str(db) + ".embedder.json").write_text('{"embedder": "hashing:5')
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        try:
            Memvara(str(db), embedder=other_embedder(), llm=NullLLM()).close()
            refused = False
        except EmbedderMismatchError:
            refused = True
    if not refused and not seen:
        raise known_bugs.Reproduced("a store whose embedder record is damaged opened with "
                                    "a different embedder of the same width, and nothing "
                                    "said so")
