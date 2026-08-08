"""Tier 1 triage: is this turn worth spending an extraction call on?

mem0 sends every turn to an LLM. In real transcripts the overwhelming majority of turns
are acknowledgements, questions, and assistant chatter that carry nothing durable, so
that spend buys almost nothing. This gate is a few string comparisons that remove them
before any model is consulted, and it is the single largest reduction in write-path cost
in the system.

The bias is deliberately asymmetric. A false positive costs one extraction call and is
recovered immediately; a false negative loses a memory permanently and silently. So the
rules only fire on shapes that are unambiguously factless, and everything else passes.
"""

from __future__ import annotations

import re

from ..types import Episode

# `[^\W_]` is "word character but not underscore", which keeps accented and non-Latin
# scripts counting as content instead of being silently dropped as punctuation.
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_SENTENCE = re.compile(r"[^.!?\n]+[.!?\n]?")
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)

# Scripts that do not put spaces between words. Token counting reads an entire Japanese
# or Thai sentence as a single token, so without a second measure the "too short to be a
# fact" rule silently discards every fact those users ever state.
#
# The second measure counts characters *from these scripts only*, rather than characters
# in general: a general character floor low enough for 我住在北京 (5 characters, a
# complete declarative) would also admit every bare Latin noun like "Berlin", which is
# exactly the fragment the rule exists to reject.
_UNSPACED_SCRIPT = re.compile(
    "["
    "぀-ヿ"   # hiragana, katakana
    "㐀-䶿"   # CJK extension A
    "一-鿿"   # CJK unified ideographs
    "豈-﫿"   # CJK compatibility ideographs
    "ｦ-ﾟ"   # halfwidth katakana
    "가-힯"   # hangul syllables
    "฀-๿"   # Thai
    "຀-໿"   # Lao
    "က-႟"   # Myanmar
    "ក-៿"   # Khmer
    "]"
)
# CJK punctuation (U+3000-U+303F, including 。) sits outside every range above, so a bare
# terminator contributes nothing and is still rejected.
_MIN_UNSPACED_CHARS = 4

# Individual words that carry no durable content. Acknowledgements are compositional
# ("sure thing", "no worries", "ok sounds good to me"), so matching whole phrases means
# maintaining a list that never finishes growing and leaks a call every time it misses.
# A vocabulary of filler *words* generalizes: any combination of them is still filler.
_FILLER_TOKENS: frozenset[str] = frozenset(
    {
        "ok", "okay", "k", "kk", "kay", "yes", "yeah", "yup", "yep", "ya", "no",
        "nope", "nah", "sure", "thing", "thanks", "thank", "you", "thx", "ty",
        "worries", "worry", "problem", "probs", "np", "welcome", "re", "please",
        "sounds", "sound", "good", "great", "cool", "nice", "fine", "perfect",
        "awesome", "excellent", "lovely", "brilliant", "right", "correct", "true",
        "got", "it", "gotcha", "understood", "understand", "that", "makes", "make",
        "sense", "will", "do", "done", "roger", "agreed", "agree", "same", "all",
        "alright", "lgtm", "to", "me", "us", "my", "for", "and", "a", "so", "much",
        "lot", "very", "too", "then", "now", "hi", "hello", "hey", "yo", "morning",
        "afternoon", "evening", "night", "bye", "goodbye", "see", "later", "soon",
        "cheers", "ta", "haha", "hah", "lol", "hmm", "hm", "huh", "wow", "oh", "ah",
        "ahh", "nvm", "ditto", "word", "indeed", "absolutely", "definitely", "sweet",
    }
)

# Beyond a handful of words, an all-filler turn is unlikely enough that letting it
# through costs one batched call and keeps the recall bias intact.
_MAX_ACK_TOKENS = 6


def _slug(raw: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", raw.strip().lower()).strip("_") or "unknown"


def _normalize(text: str) -> str:
    """Lowercase, drop punctuation and emoji, collapse whitespace."""
    return " ".join(_NON_WORD.sub(" ", text.lower()).split())


class SalienceGate:
    """Cheap, deterministic "does this plausibly contain a durable fact?" check.

    Returns `(should_extract, reason)`. The reason slug is surfaced in receipts and
    asserted in tests, so a gate decision is never a black box.
    """

    def __init__(self, *, min_tokens: int = 2) -> None:
        # A single bare token has no predicate structure to extract; it is almost always
        # an answer fragment that only means something with the preceding question, which
        # we do not have here.
        self.min_tokens = min_tokens

    def carries_fact(self, ep: Episode) -> tuple[bool, str]:
        content = (ep.content or "").strip()
        if not content:
            return False, "no_content"

        role = (ep.role or "user").strip().lower()
        if role != "user":
            # Assistant text restates or speculates; treating it as evidence is how a
            # memory store ends up believing its own hallucinations.
            return False, f"{_slug(role)}_turn"

        tokens = _normalize(content).split()
        unspaced = len(_UNSPACED_SCRIPT.findall(content))
        # The filler vocabulary is English, so non-Latin text cannot match it. Saying so
        # explicitly keeps that from being an accident of the word list's contents.
        if not unspaced and tokens and len(tokens) <= _MAX_ACK_TOKENS and all(
            t in _FILLER_TOKENS for t in tokens
        ):
            # One token outside the vocabulary is enough to let the turn through:
            # "sounds good, I moved to Berlin" is a fact wearing an acknowledgement.
            return False, "ack_only"

        # Either measure clearing its floor is enough.
        if len(_WORD.findall(content)) < self.min_tokens and unspaced < _MIN_UNSPACED_CHARS:
            return False, "too_short"

        sentences = [s.strip() for s in _SENTENCE.findall(content) if s.strip()]
        if sentences and all(s.endswith("?") for s in sentences):
            # Only a bare interrogative is dropped. "I live in Berlin, what's nearby?"
            # has a declarative sentence and passes.
            return False, "question"

        return True, "has_declarative"
