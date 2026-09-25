"""The crash child and its driver: a child reports what it acknowledged, stops where it is
told, and dies or carries on as the test says. Every failure names what went wrong."""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from harness import crash, stores
from harness.crash import Child, CrashHarnessError


def program(db: pathlib.Path, **fields: Any) -> dict[str, Any]:
    return {"db": str(db), "user": "u1", "setup": [], "point": None, "action": None,
            "hold": False, **fields}


def remember(predicate: str, obj: str, **extra: Any) -> list[Any]:
    return ["remember", {"predicate": predicate, "object": obj, **extra}]



def test_a_child_with_no_point_acknowledges_each_setup_op_and_finishes(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    spec = program(tmp_path / "s.db", setup=[remember("likes", "tea"),
                                             remember("likes", "coffee")])
    with Child(spec, home=home) as child:
        child.wait_for("DONE")
        assert [a["index"] for a in child.acked] == [0, 1]
        assert all(len(a["ids"]) == 1 for a in child.acked)
        assert child.finish() == 0


def test_a_child_killed_at_its_point_leaves_a_store_that_opens(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    db = tmp_path / "s.db"
    spec = program(db, point="after-commit", action=remember("likes", "tea"))
    with Child(spec, home=home) as child:
        child.wait_for("POINT after-commit")
        code = child.kill()
    assert code != 0
    mem = stores.file(db)
    try:
        assert [c.object for c in mem.get_all(user="u1")] == ["tea"]
    finally:
        mem.close()


def test_a_held_child_carries_on_when_it_is_released(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    spec = program(tmp_path / "s.db", point="before-claim", hold=True,
                   action=remember("likes", "tea"))
    with Child(spec, home=home) as child:
        child.wait_for("POINT before-claim")
        child.release()
        child.wait_for("DONE")
        assert child.finish() == 0


def test_a_point_the_action_never_reaches_is_reported_when_the_child_exits(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    spec = program(tmp_path / "s.db", point="erase-before-delete",
                   action=remember("likes", "tea"))
    with Child(spec, home=home) as child:
        with pytest.raises(CrashHarnessError, match="exited with status 0 before printing "
                                                    "'POINT erase-before-delete'"):
            child.wait_for("POINT erase-before-delete")


def test_a_line_that_never_comes_is_reported_after_the_timeout(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    spec = program(tmp_path / "s.db", point="before-claim", hold=True,
                   action=remember("likes", "tea"))
    with Child(spec, home=home, timeout=1.0) as child:
        child.wait_for("POINT before-claim")
        with pytest.raises(CrashHarnessError, match="no line starting with 'DONE' within 1"):
            child.wait_for("DONE")


def test_an_unknown_point_is_refused_by_the_child(
        tmp_path: pathlib.Path, home: pathlib.Path) -> None:
    spec = program(tmp_path / "s.db", point="nowhere", action=remember("likes", "tea"))
    with Child(spec, home=home) as child:
        with pytest.raises(CrashHarnessError, match="unknown point 'nowhere'"):
            child.wait_for("DONE")


def test_a_child_whose_program_cannot_be_sent_is_killed_before_the_error_is_raised(
        tmp_path: pathlib.Path, home: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The child is started before its program is written to it. If that write fails, no
    Child exists for a `with` block to clean up, so the constructor must kill it."""
    started: list[Any] = []
    real_popen = crash.subprocess.Popen

    def popen(*args: Any, **kwargs: Any) -> Any:
        started.append(real_popen(*args, **kwargs))
        return started[-1]

    def refuse(self: Child, line: str) -> None:
        raise BrokenPipeError("the child closed its input")

    monkeypatch.setattr(crash.subprocess, "Popen", popen)
    monkeypatch.setattr(Child, "_send", refuse)
    with pytest.raises(BrokenPipeError):
        Child(program(tmp_path / "s.db"), home=home)
    assert len(started) == 1
    assert started[0].poll() is not None, "the child is still running"
