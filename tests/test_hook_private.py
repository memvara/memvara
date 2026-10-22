"""Text a person wraps in `<private>` tags never reaches memvara through the hooks.

A person sometimes has to show the assistant something that must not be remembered: a
token, a customer's name, a health detail. The hooks capture every turn and search with
every prompt, so without a marker the only choices were to keep it out of the
conversation or to let it be stored. Supermemory's coding plugins accept
`<private>...</private>` for this; these tests hold the hooks to the same promise.

Two paths carry a person's text out of the hooks, and both are checked: the turn that
capture mines (and the earlier turns it shows agentic capture as context), and the prompt
that recall sends to the store as its query. The model in front of the person still sees
the text; only the hooks are bound by the tag.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

HOOKS = pathlib.Path(__file__).resolve().parent.parent / "plugin" / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

from lib import private_tag, transcript  # noqa: E402

SECRET = "sk-live-4f9a"


def _jsonl(*entries: dict) -> bytes:
    return "\n".join(json.dumps(e) for e in entries).encode("utf-8")


def _user(text: str) -> dict:
    return {"type": "user", "message": {"role": "user", "content": text}}


def _assistant(text: str) -> dict:
    return {"type": "assistant", "message": {"role": "assistant",
                                             "content": [{"type": "text", "text": text}]}}


@pytest.mark.parametrize("text, kept", [
    (f"the key is <private>{SECRET}</private>, why does auth fail?",
     "the key is , why does auth fail?"),
    (f"<PRIVATE>{SECRET}</Private> then", " then"),
    (f"a <private>\n{SECRET}\nsecond line\n</private> b", "a  b"),
    (f"<private>{SECRET}</private> and <private>other</private> kept", " and  kept"),
    ("no tag here at all", "no tag here at all"),
])
def test_every_private_span_is_removed_with_its_tags(text: str, kept: str) -> None:
    assert private_tag.strip(text) == kept


def test_a_span_with_no_closing_tag_hides_everything_after_it() -> None:
    """Fail closed: a missing close tag, or one cut off by truncation, keeps the rest out.

    The other reading, that an unclosed tag hides nothing, would store exactly the text
    the person marked. Losing the rest of one turn's capture is the cheaper mistake.
    """
    assert private_tag.strip(f"before <private>{SECRET} and more\nlines") == "before "


def test_the_mined_turn_and_its_context_carry_no_private_text() -> None:
    raw = _jsonl(
        _user(f"earlier: my token is <private>{SECRET}</private>"),
        _assistant("Noted."),
        _user(f"deploy with <private>{SECRET}</private> to staging"),
        _assistant(f"Deploying. I used <private>{SECRET}</private> as asked."),
    )
    turn, _, context = transcript.last_turn_with_context(raw, 10_000)
    assert "User: deploy with  to staging" in turn
    assert "Deploying. I used  as asked." in turn
    assert "earlier: my token is" in context
    assert SECRET not in turn
    assert SECRET not in context


def test_the_recall_query_carries_no_private_text(tmp_path) -> None:
    from test_hook_recall_requests import Fake, _run_hook

    fake = Fake()
    _run_hook(fake, tmp_path,
              prompt=f"which database does billing use <private>{SECRET}</private>")
    assert fake.recalls, "the hook sent no recall, so this checked nothing"
    for call in fake.recalls:
        assert SECRET not in json.dumps(call)
        assert "which database does billing use" in call["query"]
