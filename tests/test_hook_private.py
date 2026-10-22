"""Text a person wraps in `<private>` tags never reaches memvara through the hooks.

A person sometimes has to show the assistant something that must not be remembered: a
token, a customer's name, a health detail. The hooks capture every turn and search with
every prompt, so without a marker the only choices were to keep it out of the
conversation or to let it be stored. Supermemory's coding plugins accept
`<private>...</private>` for this; these tests hold the hooks to the same promise.

Two paths carry a person's text out of the hooks, and both are checked: the turn that
capture mines (and the earlier turns it shows agentic capture as context), on every host's
transcript format, and the prompt that recall sends to the store as its query. The model in
front of the person still sees the text, and can repeat it; a repeat inside the capture
window is removed too. A memory tool the model calls itself is not covered by the tag.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
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


def _bash(command: str) -> dict:
    return {"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "tool_use", "name": "Bash", "input": {"command": command}}]}}


# -- what a span is ------------------------------------------------------------------


@pytest.mark.parametrize("text, kept", [
    (f"the key is <private>{SECRET}</private>, why does auth fail?",
     "the key is , why does auth fail?"),
    (f"<PRIVATE>{SECRET}</Private> then", " then"),
    (f"a <private>\n{SECRET}\nsecond line\n</private> b", "a  b"),
    (f"<private>{SECRET}</private> and <private>other</private> kept", " and  kept"),
    (f"a <private reason='token'>{SECRET}</private> b", "a  b"),
    (f"a < private >{SECRET}</ private > b", "a  b"),
    ("a <private/> b", "a  b"),
    ("no tag here at all", "no tag here at all"),
])
def test_every_private_span_is_removed_with_its_tags(text: str, kept: str) -> None:
    assert private_tag.strip(text) == kept


def test_a_nested_span_hides_everything_up_to_its_own_closing_tag() -> None:
    """The first closing tag closes the inner span, not the outer one."""
    text = "a <private>S1 <private>S2</private> S3</private> b"
    assert private_tag.strip(text) == "a  b"


def test_a_span_with_no_closing_tag_hides_everything_after_it() -> None:
    """Fail closed: a missing close tag keeps the rest of that text out.

    The other reading, that an unclosed tag hides nothing, would store exactly the text
    the person marked.
    """
    assert private_tag.strip(f"before <private>{SECRET} and more\nlines") == "before "


def test_a_closing_tag_with_no_opening_tag_is_dropped_and_hides_nothing() -> None:
    assert private_tag.strip("S1</private> mid <private>S2</private> end") == "S1 mid  end"


def test_a_tag_written_in_backticks_is_a_mention_and_not_a_tag() -> None:
    """How a person writes about the tag without hiding the rest of the prompt."""
    text = "what does the `<private>` tag do, and where is `</private>` needed?"
    assert private_tag.strip(text) == text


def test_a_hidden_value_repeated_without_tags_is_removed_too() -> None:
    assert private_tag.strip(f"use <private>{SECRET}</private>; I said {SECRET}") == \
        "use ; I said "


def test_a_hidden_value_shorter_than_three_characters_is_not_hunted_elsewhere() -> None:
    """Removing every "42" from a turn would damage far more than it protects."""
    assert private_tag.strip("pick <private>42</private>, not 42") == "pick , not 42"


# -- capture -------------------------------------------------------------------------


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


def test_the_model_repeating_the_secret_without_tags_is_not_mined() -> None:
    """The person tags it once; the model echoes it in a reply and a command."""
    raw = _jsonl(
        _user(f"deploy with <private>{SECRET}</private>"),
        _bash(f"deploy --key {SECRET}"),
        _assistant(f"Done, I used {SECRET} as the key."),
    )
    turn, _, _ = transcript.last_turn_with_context(raw, 10_000)
    assert "Done, I used  as the key." in turn
    assert SECRET not in turn


def test_a_secret_tagged_in_an_earlier_turn_is_not_mined_when_echoed_later() -> None:
    raw = _jsonl(
        _user(f"remember for this session: <private>{SECRET}</private>"),
        _assistant("Understood."),
        _user("now deploy it"),
        _assistant(f"Deploying with {SECRET}."),
    )
    turn, _, context = transcript.last_turn_with_context(raw, 10_000)
    assert SECRET not in turn
    assert SECRET not in context


def test_a_span_never_reaches_across_from_the_prompt_into_the_reply() -> None:
    """An unclosed tag in the prompt hides the rest of the prompt, not the reply.

    Stripped as one joined string, the prompt's unclosed tag swallowed the `Claude:`
    marker, and the reply's sentence came out filed under `User:`.
    """
    raw = _jsonl(
        _user("what does the <private> tag do exactly?"),
        _assistant(f"<private>{SECRET}</private> is dropped. We always deploy on Fridays."),
    )
    turn, _, _ = transcript.last_turn_with_context(raw, 10_000)
    assert "Claude: is dropped. We always deploy on Fridays." in turn
    assert "Fridays" not in transcript.user_lines(turn)


def test_the_context_is_stripped_before_it_is_cut_to_length() -> None:
    """Cut first, the opening tag could fall outside the kept tail while the secret stayed."""
    raw = _jsonl(
        _user(f"earlier <private>{'x' * 400} {SECRET}</private> done"),
        _assistant("ok"),
        _user("next"),
        _assistant("fine"),
    )
    _, _, context = transcript.last_turn_with_context(raw, 60)
    assert context, "the context window was empty, so this checked nothing"
    assert SECRET not in context


HOST_FORMATS = {
    "claude": """
def u(t): return {"type":"user","message":{"role":"user","content":t}}
def a(t): return {"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":t}]}}
def tool(c): return {"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"Bash","input":{"command":c}}]}}
""",
    "codex": """
def u(t): return {"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":t}]}}
def a(t): return {"type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":t}]}}
def tool(c): return {"type":"response_item","payload":{"type":"custom_tool_call","name":"exec","input":c}}
""",
    "copilot": """
def u(t): return {"type":"user.message","data":{"content":t,"transformedContent":t}}
def a(t): return {"type":"assistant.message","data":{"content":t,"toolRequests":[]}}
def tool(c): return {"type":"assistant.message","data":{"content":"","toolRequests":[{"name":"bash","arguments":{"command":c}}]}}
""",
    "cursor": """
def u(t): return {"role":"user","message":{"content":[{"type":"text","text":"<user_query>\\n"+t+"\\n</user_query>"}]}}
def a(t): return {"role":"assistant","message":{"content":[{"type":"text","text":t}]}}
def tool(c): return {"role":"assistant","message":{"content":[{"type":"tool_use","name":"Shell","input":{"command":c}}]}}
""",
    "opencode": """
def u(t): return {"type":"user","message":{"role":"user","content":t}}
def a(t): return {"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":t}]}}
def tool(c): return {"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"bash","input":{"command":c}}]}}
""",
}


@pytest.mark.parametrize("host", sorted(HOST_FORMATS))
def test_every_host_format_is_stripped(host: str) -> None:
    """Each host is bound before `transcript` is imported, so each runs in its own process."""
    script = f"""
import importlib, json, sys
sys.path.insert(0, {str(HOOKS)!r})
from core import host as H
H.use(importlib.import_module("hosts.{host}").HOST)
from lib import transcript as T
{HOST_FORMATS[host]}
S = {SECRET!r}
raw = "\\n".join(json.dumps(e) for e in (
    u("earlier <private>" + S + "</private> kept-A"), a("ok"),
    u("deploy with <private>" + S + "</private> kept-B"), tool("deploy --key " + S),
    a("Done, I used " + S + "."))).encode()
turn, _, context = T.last_turn_with_context(raw, 10000)
print(json.dumps({{"turn": turn, "context": context}}))
"""
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          timeout=60)
    assert done.returncode == 0, done.stderr
    out = json.loads(done.stdout)
    assert "kept-B" in out["turn"], f"{host}: the turn was not read at all: {out}"
    assert SECRET not in out["turn"]
    assert SECRET not in out["context"]


# -- recall --------------------------------------------------------------------------


def test_the_recall_query_carries_no_private_text(tmp_path) -> None:
    from test_hook_recall_requests import Fake, _run_hook

    fake = Fake()
    _run_hook(fake, tmp_path,
              prompt=f"which database does billing use <private>{SECRET}</private>")
    assert fake.recalls, "the hook sent no recall, so this checked nothing"
    for call in fake.recalls:
        assert SECRET not in json.dumps(call)
        assert "which database does billing use" in call["query"]


def test_a_long_prompt_is_stripped_before_it_is_bounded(tmp_path) -> None:
    """Bounded first, the cut could drop the opening tag and keep the secret behind it."""
    import recall

    # The opening tag sits in the middle the bound drops, and the secret and the closing
    # tag sit in the tail it keeps.
    half = recall.MAX_PROMPT_CHARS // 2
    prompt = (f"which database does billing use {'y ' * half}<private>{'x ' * half}"
              f"{SECRET}</private> thanks")
    from test_hook_recall_requests import Fake, _run_hook

    fake = Fake()
    _run_hook(fake, tmp_path, prompt=prompt)
    assert fake.recalls, "the hook sent no recall, so this checked nothing"
    for call in fake.recalls:
        assert SECRET not in json.dumps(call)
