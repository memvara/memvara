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
    _Named("local:some-org/unmeasured-model"),
])
def test_a_space_nobody_measured_keeps_the_thresholds_every_release_used(embedder):
    """0.40 and 0.97 hold under hashing, which folds none of the 69 pairs of different
    values at any threshold measured. An embedder that was not measured on its own keeps
    them, so nothing moves for a store that neither MiniLM nor bge-small wrote."""
    assert calibration_of(embedder) == BASELINE == Calibration(0.40, 0.97)


def test_minilm_keeps_its_rescue_and_merges_above_its_closest_different_values():
    """0.40 was measured under MiniLM and stays. Its merge at 0.97 folds 4 of the 24
    pairs of different values that hold the same numbers, which only a threshold can keep
    apart; at 0.985 it folds none of them. Read through the cache wrapper too."""
    minilm = CachedEmbedder(_Named("local:sentence-transformers/all-MiniLM-L6-v2"))
    assert calibration_of(minilm) == Calibration(grounding_rescue=0.40, merge=0.985)


def test_bge_small_is_read_with_the_thresholds_measured_in_its_space():
    """bge-small scores the median invented value 0.44 against a source it has nothing to
    do with, where MiniLM scores it 0.02, and scores two values a letter apart up to
    0.989. It gets its own thresholds, and gets them through the cache wrapper too,
    because a cache does not change the space."""
    bge = CachedEmbedder(_Named("local:BAAI/bge-small-en-v1.5"))
    assert calibration_of(bge) == Calibration(grounding_rescue=0.65, merge=0.99)
