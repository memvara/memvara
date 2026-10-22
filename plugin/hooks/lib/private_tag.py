"""Text a person wraps in `<private>...</private>` never reaches memvara through the hooks.

A person sometimes has to show the assistant something that must not be remembered: a
token, a customer's name, a health detail. Capture mines every turn and recall searches
with every prompt, so the tag is the person's way to keep one passage out of both.
`transcript.last_turn_with_context` strips it from the turn capture mines and from the
earlier turns shown as context, and `recall.py` strips it from the prompt before the
prompt becomes a query, a log line or anything else. The model the person is talking to
still sees the text; only the hooks are bound by the tag.

A span is removed whole, tags included, and nothing is put in its place. A placeholder
would be worse than nothing: the fast extractor believes sentence forms, so "my name is
[private]" could be stored as a name.

An opening tag with no closing tag hides everything after it. That reading fails closed:
the close may be missing because a tool argument was cut short before it, and the other
reading would store exactly the text the person marked. Losing the rest of one turn's
capture is the cheaper mistake.
"""

from __future__ import annotations

import re

#: An opening tag, then everything up to the first closing tag, or to the end of the text
#: when there is none. Case does not matter, and neither does space before the `>`.
_SPAN = re.compile(r"<private\s*>.*?(?:</private\s*>|\Z)", re.IGNORECASE | re.DOTALL)


def strip(text: str) -> str:
    """`text` with every `<private>` span removed, tags included."""
    return _SPAN.sub("", text)
