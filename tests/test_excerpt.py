"""`retrieve/excerpt.py`: which part of a long turn a reader is shown.

The contract has three parts, and each test below holds one of them: the window is never
longer than the limit, it is the part of the turn the question names when there is one,
and when there is none it is exactly the head cut `recall()` always made.
"""

from __future__ import annotations

import random

from memvara import Memvara
from memvara.retrieve.analyze import MIN_TERM_CHARS, tokenize
from memvara.retrieve.excerpt import _WORD, ELLIPSIS, excerpt

WORDS = "alpha beta gamma delta kafka lisbon greyhound moving receipt kayak".split()


def test_a_turn_that_fits_is_returned_whole():
    assert excerpt("I adopted a greyhound.", "greyhound", 280) == "I adopted a greyhound."


def test_the_window_is_never_longer_than_the_limit():
    """The limit is what stops a pasted stack trace from becoming the prompt, so no
    arrangement of sentences, question and limit may produce a longer window. Checked over
    a fixed, seeded sample of shapes, including sentences longer than the limit itself."""
    rng = random.Random(20260924)
    for _ in range(3000):
        sentences = [" ".join(rng.choice(WORDS) for _ in range(rng.randint(1, 40)))
                     + rng.choice(".!?") for _ in range(rng.randint(1, 8))]
        text = " ".join(sentences)
        limit = rng.randint(5, 300)
        window = excerpt(text, " ".join(rng.sample(WORDS, 2)), limit)
        assert len(window) <= limit, (limit, window)


def test_the_sentence_the_question_names_is_shown_with_what_follows_it_first():
    """An answer usually comes after the sentence that names the topic, so the window
    grows forward before it grows back."""
    text = ("First we talked about the weather. Then you asked about kafka. "
            "The answer is to sunset it. Later we discussed lunch.")

    window = excerpt(text, "kafka", 60)

    assert window == "…Then you asked about kafka. The answer is to sunset it.…"


def test_a_question_that_names_nothing_gets_the_head_cut_recall_always_made():
    """Nothing to aim at is not a reason to render a turn differently from before."""
    text = "The opening line of a very long turn. " * 20

    assert excerpt(text, "what is the capital of France", 100) == (
        Memvara._safe_line(text, 100))


def test_a_question_of_stopwords_alone_also_gets_the_head_cut():
    text = "The opening line of a very long turn. " * 20

    assert excerpt(text, "what is it about?", 100) == Memvara._safe_line(text, 100)


def test_words_match_through_their_stem():
    """A question says "adopted" and the turn says "adopt": the same fold the intent
    classifier uses, so they meet."""
    text = "Filler about something else entirely. " * 5 + "We will adopt a dog in May."

    assert excerpt(text, "when was the dog adopted", 40).endswith("We will adopt a dog in May.")


def test_a_tie_goes_to_the_earlier_sentence():
    """Deterministic, and the earlier sentence is where the old cut would have looked."""
    text = "The kafka cluster is old. " + "Filler. " * 30 + "The kafka pipeline is new."

    assert excerpt(text, "kafka", 30).startswith("The kafka cluster is old.")


def test_a_long_sentence_is_cut_around_its_first_match_and_uses_all_its_room():
    """A sentence longer than the window is cut around the matching word, and a match
    near the end of the sentence reaches back rather than leaving the room unused."""
    sentence = "word " * 80 + "kayak."

    window = excerpt(sentence, "kayak", 60)

    assert window.endswith("kayak.")
    assert window.startswith(ELLIPSIS)
    assert len(window) >= 55


def test_a_long_sentence_after_the_first_is_cut_around_its_own_match():
    """The long-sentence cut places its window by offsets into the whole turn. On a
    sentence that does not start the turn, an offset counted from the sentence instead
    puts the window past the word it is meant to show."""
    text = "Opening remarks about nothing. " * 6 + "word " * 40 + "kayak " + "word " * 40 + "end."

    assert "kayak" in excerpt(text, "kayak", 60)


def test_words_are_split_exactly_as_the_question_is():
    """The window finds a turn's words with a regular expression, for speed, and the
    question's words come from `analyze.tokenize`, which walks every character. A word
    split differently on the two sides never matches, so both must split any text the
    same way, including underscores, combining marks and letters that lowercase to two
    characters."""
    rng = random.Random(7)
    pool = "abcXYZ_09 .,!?İıßẞ²³①Ⅻ日本語한국어é\u0301\u0307\t\u00a0-"
    for _ in range(5000):
        text = "".join(rng.choice(pool) for _ in range(rng.randint(0, 40)))
        words = [w for w in _WORD.findall(text.lower()) if len(w) >= MIN_TERM_CHARS]
        assert words == tokenize(text), text


def test_each_side_where_text_was_left_out_is_marked():
    text = "Before. " * 10 + "The kayak receipt is in the drawer. " + "After. " * 10

    window = excerpt(text, "kayak receipt", 50)

    assert window.startswith(ELLIPSIS) and window.endswith(ELLIPSIS)
    assert "The kayak receipt is in the drawer." in window
