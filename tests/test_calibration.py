"""`embed/calibration.py`: the cosine thresholds each embedding space is read with."""

from __future__ import annotations

import pytest

from memvara.embed import CachedEmbedder, HashingEmbedder
from memvara.embed.calibration import BASELINE, Calibration, calibration_of


class _Named:
    """An embedder that is nothing but a fingerprint name, which is all the lookup reads."""

    dim = 4

    def __init__(self, name: str) -> None:
        self.name = name


@pytest.mark.parametrize("embedder", [
    HashingEmbedder(dim=512),
    CachedEmbedder(HashingEmbedder(dim=512)),
    _Named("local:sentence-transformers/all-MiniLM-L6-v2"),
    _Named("local:some-org/unmeasured-model"),
])
def test_a_space_nobody_measured_keeps_the_thresholds_every_release_used(embedder):
    """0.40 and 0.97 were measured under MiniLM and hold under hashing. An embedder that
    was not measured on its own keeps them, so nothing moves for a store that
    bge-small did not write."""
    assert calibration_of(embedder) == BASELINE == Calibration(0.40, 0.97)


def test_bge_small_is_read_with_the_thresholds_measured_in_its_space():
    """bge-small scores the median invented value 0.44 against a source it has nothing to
    do with, where MiniLM scores it 0.02, and scores two values one digit apart up to
    0.985. It gets its own thresholds, and gets them through the cache wrapper too,
    because a cache does not change the space."""
    bge = CachedEmbedder(_Named("local:BAAI/bge-small-en-v1.5"))
    assert calibration_of(bge) == Calibration(grounding_rescue=0.65, merge=0.99)
