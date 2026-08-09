"""Reciprocal rank fusion, checked against hand-computed values.

The arithmetic is trivial enough that it is tempting to assert only on ordering. That
would miss the failure that actually bites: a weight applied to the wrong list, or an
off-by-one in the rank, still produces a plausible-looking order. So the contributions
are pinned to exact numbers.
"""

from __future__ import annotations

import pytest

from memvara.retrieve import reciprocal_rank_fusion

# Hand-computed contributions at the default k=60, for 0-based rank r: 1 / (60 + r + 1).
R0 = 1.0 / 61.0  # 0.016393442622950821
R1 = 1.0 / 62.0  # 0.016129032258064516
R2 = 1.0 / 63.0  # 0.015873015873015872
R3 = 1.0 / 64.0  # 0.015625


def test_single_ranking_matches_hand_computed_contributions() -> None:
    fused = reciprocal_rank_fusion({"vector": [("a", 0.9), ("b", 0.5), ("c", 0.1)]})

    assert fused["a"] == pytest.approx(R0)
    assert fused["b"] == pytest.approx(R1)
    assert fused["c"] == pytest.approx(R2)


def test_agreement_across_retrievers_sums() -> None:
    fused = reciprocal_rank_fusion(
        {
            "vector": [("a", 0.90), ("b", 0.80)],
            "lexical": [("c", 5.0), ("a", 1.0)],
        }
    )

    # "a" is first for the vector leg and second for the lexical leg.
    assert fused["a"] == pytest.approx(R0 + R1)
    assert fused["b"] == pytest.approx(R1)
    assert fused["c"] == pytest.approx(R0)

    # Agreement beats a single strong hit; and a lone rank-0 hit ("c") still outranks a
    # lone rank-1 hit ("b"), which is the ordering RRF exists to produce.
    assert list(fused) == ["a", "c", "b"]


def test_item_ranked_first_by_one_retriever_and_absent_from_the_other_survives() -> None:
    """The property the whole hybrid design rests on.

    An exact-token match that the embedding never surfaced must not be erased by its
    absence from the vector list. Here "rare" is BM25's top hit and invisible to the
    vector leg, yet it still lands above three claims the vector leg did rank.
    """
    fused = reciprocal_rank_fusion(
        {
            "vector": [("v1", 0.81), ("v2", 0.77), ("v3", 0.74), ("v4", 0.71)],
            "lexical": [("rare", 9.2)],
        }
    )

    assert fused["rare"] == pytest.approx(R0)
    assert list(fused) == ["rare", "v1", "v2", "v3", "v4"]
    # Absence is scored as zero contribution, not as a penalty.
    assert fused["v1"] == pytest.approx(R0)
    assert fused["rare"] == fused["v1"]


def test_weights_scale_each_list_independently() -> None:
    fused = reciprocal_rank_fusion(
        {"vector": [("a", 1.0), ("b", 1.0)], "lexical": [("b", 1.0), ("a", 1.0)]},
        weights={"vector": 2.0, "lexical": 0.5},
    )

    assert fused["a"] == pytest.approx(2.0 * R0 + 0.5 * R1)
    assert fused["b"] == pytest.approx(2.0 * R1 + 0.5 * R0)
    assert list(fused) == ["a", "b"]


def test_missing_weight_defaults_to_one() -> None:
    fused = reciprocal_rank_fusion(
        {"vector": [("a", 1.0)], "lexical": [("b", 1.0)]},
        weights={"vector": 3.0},  # "lexical" unspecified
    )

    assert fused["a"] == pytest.approx(3.0 * R0)
    assert fused["b"] == pytest.approx(R0)


def test_zero_weight_removes_the_retriever_entirely() -> None:
    """A disabled leg must not be able to introduce candidates.

    Adding 0.0 would still create the key, which would let a switched-off retriever
    inject documents no live retriever ever found - they would sort last, but they
    would occupy slots in the final k.
    """
    fused = reciprocal_rank_fusion(
        {"vector": [("only_vector", 1.0)], "lexical": [("a", 1.0)]},
        weights={"vector": 0.0},
    )

    assert "only_vector" not in fused
    assert list(fused) == ["a"]


def test_k_damps_the_head_of_the_curve() -> None:
    small = reciprocal_rank_fusion({"v": [("a", 1.0), ("b", 1.0)]}, k=0)
    large = reciprocal_rank_fusion({"v": [("a", 1.0), ("b", 1.0)]}, k=1000)

    assert small["a"] == pytest.approx(1.0)
    assert small["b"] == pytest.approx(0.5)
    assert large["a"] == pytest.approx(1.0 / 1001.0)
    assert large["b"] == pytest.approx(1.0 / 1002.0)

    # Small k trusts each retriever's own ordering; large k flattens it toward
    # "how many retrievers found this at all".
    assert small["a"] / small["b"] > large["a"] / large["b"]


def test_the_default_k_makes_the_head_of_the_curve_almost_flat() -> None:
    """The measurement behind a design decision elsewhere, pinned here so it cannot
    quietly stop being true.

    At k=60 the whole of first place is worth 1.6% over second place, while the
    post-fusion quality multiplier spans up to 1.66x - so multiplying the two lets
    freshness outrank relevance by about 41 positions. The resolution was not to retune
    k, which would trade that pathology for over-trusting each retriever's own
    ordering, but to stop deriving the caller-facing score from rank at all: `k` still
    sets the candidate union and the agreement signal, where flatness is a virtue.
    """
    fused = reciprocal_rank_fusion({"v": [(f"id{i}", 1.0) for i in range(64)]})
    values = list(fused.values())

    assert values[0] / values[1] == pytest.approx(1.016, abs=0.001)
    # A 1.66x quality multiplier buys exactly this many rank positions: the ratio
    # crosses 1.66 between rank 40 and rank 41.
    assert values[0] / values[40] < 1.66 < values[0] / values[41]


def test_negative_k_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        reciprocal_rank_fusion({"v": [("a", 1.0)]}, k=-1)


def test_ties_break_on_item_id_for_stable_ordering() -> None:
    """Two items each ranked first by one retriever score identically by construction."""
    fused = reciprocal_rank_fusion({"vector": [("zeta", 1.0)], "lexical": [("alpha", 1.0)]})

    assert fused["alpha"] == pytest.approx(fused["zeta"])
    assert list(fused) == ["alpha", "zeta"]

    # Feeding the same information in the opposite order must not change the answer.
    flipped = reciprocal_rank_fusion({"vector": [("alpha", 1.0)], "lexical": [("zeta", 1.0)]})
    assert list(flipped) == ["alpha", "zeta"]


def test_raw_scores_are_ignored() -> None:
    """Only position is read. This is rank fusion, not score fusion."""
    modest = reciprocal_rank_fusion({"v": [("a", 0.01), ("b", 0.009)]})
    huge = reciprocal_rank_fusion({"v": [("a", 9999.0), ("b", -3.0)]})

    assert modest == huge


def test_repeated_id_within_a_list_keeps_its_best_rank() -> None:
    fused = reciprocal_rank_fusion({"v": [("a", 1.0), ("b", 1.0), ("a", 0.5)]})

    assert fused["a"] == pytest.approx(R0)  # not R0 + R2
    assert fused["b"] == pytest.approx(R1)


def test_degenerate_inputs_return_empty() -> None:
    assert reciprocal_rank_fusion({}) == {}
    assert reciprocal_rank_fusion({"vector": [], "lexical": []}) == {}


def test_many_retrievers_compose() -> None:
    """Nothing in the signature is limited to two legs."""
    fused = reciprocal_rank_fusion(
        {"a": [("x", 1.0)], "b": [("x", 1.0)], "c": [("y", 1.0)], "d": [("y", 1.0), ("x", 1.0)]}
    )

    assert fused["x"] == pytest.approx(R0 + R0 + R1)
    assert fused["y"] == pytest.approx(R0 + R0)
    assert list(fused) == ["x", "y"]


def test_long_lists_stay_monotonically_decreasing() -> None:
    ranked = [(f"id{i:03d}", 1.0 - i / 1000.0) for i in range(500)]
    fused = reciprocal_rank_fusion({"v": ranked})

    values = list(fused.values())
    assert values == sorted(values, reverse=True)
    assert list(fused)[:4] == ["id000", "id001", "id002", "id003"]
    assert fused["id003"] == pytest.approx(R3)
