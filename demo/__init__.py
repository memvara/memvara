"""The answer-quality demo: an authored support history, arms to answer it, a harness.

`demo/scenario.py` is the corpus and the question set, and `demo/distractors.py` scales
the corpus with generated tickets that move no fact. Those two are re-exported here — both
are pure data with no dependencies, so importing `demo` costs nothing and cannot fail.
`demo.baselines` and the harness are imported by name (`from demo import baselines`)
rather than eagerly, because they pull in numpy, the bench helpers and optionally
sentence-transformers, and a scenario reader should not pay for any of that.

>>> from demo import conversation, questions
>>> len(conversation()), len(questions())
(64, 20)
"""

from __future__ import annotations

from .distractors import scale_conversation, scaled_conversation
from .scenario import Question, Turn, conversation, questions

__all__ = ["Question", "Turn", "conversation", "questions", "scale_conversation",
           "scaled_conversation"]
