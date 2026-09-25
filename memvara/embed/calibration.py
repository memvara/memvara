"""The cosine thresholds that depend on which embedding space they are read in.

Three places compare a cosine against a fixed number. The grounding rescue in
`write/pipeline.py` keeps a model-proposed claim whose words appear nowhere in its source
when its best chunk cosine against that source reaches `grounding_rescue`: a paraphrase
scores high and an invention does not. The duplicate merge in `consolidate/merge.py` folds
two live claims in one slot into one when their cosine reaches `merge`. The near-duplicate
check in `write/pipeline.py` reads a new turn as a restatement of the nearest claim, and
extracts nothing from it, when their cosine reaches `merge` too.

The merge and the near-duplicate check also share a rule no threshold can replace: two
texts that hold different numbers are never one value, however close they embed.
`numbers` is that rule's reading of a text.

A cosine is not a portable quantity. Both numbers were first measured under
`sentence-transformers/all-MiniLM-L6-v2`, the model `LocalEmbedder()` loaded through
0.15, and both hold under `HashingEmbedder`. More pairs showed the merge's 0.97 is too
low for MiniLM itself: at 0.97, MiniLM folds two dates a day apart into one claim.
`BAAI/bge-small-en-v1.5`, the default since, scores unrelated text far higher: the median
cosine between an invented value and a source it has nothing to do with is 0.44 under it,
against 0.02 under MiniLM. At 0.40, 84% of those inventions would be kept. So each
embedding space gets the thresholds measured in it, and an embedder that is not listed
keeps the ones every release before this used.

`bench/embedder_calibration.py` is the measurement, over pairs written for it: 15
paraphrases of facts stated in real turns, 15 invented values against 20 unrelated turns,
69 pairs of claims that differ in one value, and 26 restatements of one value that only
the merge can fold. The original eval behind 0.40, 33 inventions from two 4B-class
models, is not in this repository, so these values rest on the reconstruction and should
be read with that in mind.

>>> from memvara.embed import HashingEmbedder
>>> calibration_of(HashingEmbedder()) == BASELINE
True
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .base import _name_of

__all__ = ["BASELINE", "Calibration", "calibration_of", "numbers"]


@dataclass(frozen=True, slots=True)
class Calibration:
    """The cosine thresholds for one embedding space."""

    #: The grounding rescue in `write/pipeline.py`: an ungrounded claim is kept when its
    #: best chunk cosine against its source reaches this.
    grounding_rescue: float
    #: The duplicate merge in `consolidate/merge.py`: two live claims in one slot merge
    #: when their cosine reaches this. The near-duplicate check in `write/pipeline.py`
    #: reads a turn as a restatement of the nearest claim at the same value, because a
    #: turn worded like a claim embeds exactly as that claim would: the pairs that measure
    #: the merge measure the check.
    merge: float


#: Every embedder not listed in `_MEASURED`, and the values every release through 0.15
#: used for all of them.
#:
#: The rescue's 0.40 was measured under MiniLM, on the 33 fabricated claims from the eval
#: behind the grounding check plus 8 hand-built paraphrases sharing no vocabulary with
#: their sources. Every wholesale invention -- the "Acme" employer, the fictional
#: pet-and-pollen persona, the `"unknown"` template stubs -- scored 0.33 or below; the
#: paraphrases the rescue exists for scored 0.45 and up; the separating region on that
#: data is [0.34, 0.42] and 0.40 sits inside it with margin on the side that matters.
#: The only two fabrications above it (0.43, 0.46) were typo-variants of text genuinely
#: in the source -- misreadings, not inventions, and the least dangerous thing the
#: filter can miss. On the reconstruction in `bench/embedder_calibration.py`, 0.40 keeps
#: none of 300 inventions and 80% of the paraphrases.
#:
#: Under `HashingEmbedder` the same pairs score 0.0-0.11 -- character n-grams have
#: nothing to say about meaning -- so nothing is ever rescued and `"auto"` degrades to
#: the strict lexical check. That is graceful rather than accidental: the rescue's
#: quality follows the embedder's.
BASELINE = Calibration(grounding_rescue=0.40, merge=0.97)

#: Embedding spaces measured to need other values, keyed by fingerprint name.
#:
#: Both merge values assume the rule in `consolidate/merge.py` that two values holding
#: different numbers never merge, because no threshold keeps those apart in either space:
#: MiniLM scores two appointment dates a day apart at 0.997, and bge-small two numpy
#: versions at 0.995, and at 0.995 neither folds any of the 26 restatements. Of the 69
#: pairs that differ in one value, 24 hold the same numbers, and only a threshold can keep
#: those apart. The closest of them score 0.979 under MiniLM (two booking references a
#: letter apart) and 0.989 under bge-small (two availability zones). Of the 26
#: restatements, 24 hold the same numbers.
#:
#: all-MiniLM-L6-v2: at the baseline's 0.97 the merge folds 22 of the 69 different values,
#: 4 of them with the same numbers, and 4 of the 24 restatements. At 0.985 it folds none
#: of the 24 different values and 1 of the restatements, "grey" and "gray" (0.991). In this
#: space 22 of the 24 restatements score below the closest pair of different values, so a
#: threshold that keeps the values apart leaves the merge little to fold. 0.98 would also
#: fold "1.2.3" and "v1.2.3" (0.982), with 0.001 between it and the closest different
#: pair. The rescue keeps the baseline's 0.40, which was measured in this space.
#:
#: bge-small-en-v1.5: at 0.65 the rescue keeps none of the 300 inventions, whose highest
#: score is 0.614, and 80% of the paraphrases, the same share MiniLM keeps at 0.40. At
#: 0.99 the merge folds none of the 24 different values and 3 of the 24 restatements, with
#: 0.001 between it and the closest different pair. Without the rule on numbers it would
#: also fold 3 of the 69.
_MEASURED: dict[str, Calibration] = {
    "local:sentence-transformers/all-MiniLM-L6-v2": Calibration(
        grounding_rescue=BASELINE.grounding_rescue, merge=0.985),
    "local:BAAI/bge-small-en-v1.5": Calibration(grounding_rescue=0.65, merge=0.99),
}


def calibration_of(embedder: object) -> Calibration:
    """The thresholds measured in this embedder's space, or `BASELINE` if none were.

    Looked up by the embedder's fingerprint name, so a `CachedEmbedder` gets the values of
    the embedder it wraps.
    """
    return _MEASURED.get(_name_of(embedder), BASELINE)


#: A run of digits.
_NUMBER = re.compile(r"\d+")


def numbers(text: str) -> tuple[str, ...]:
    """The runs of digits in `text`, in order, each without its leading zeros.

    Two texts holding different numbers state two values, however close they embed, so
    neither the merge nor the near-duplicate check treats them as one. Compared as digit
    strings rather than as integers, so a pasted value thousands of digits long costs no
    conversion and cannot exceed Python's limit on one.

    >>> numbers("2023-05-01") == numbers("2023-05-02")
    False
    >>> numbers("09:30") == numbers("9:30")
    True
    """
    return tuple(run.lstrip("0") or "0" for run in _NUMBER.findall(text))
