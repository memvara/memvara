"""The fixture every model-faults test uses to make a scripted model."""

from __future__ import annotations

from typing import Callable, Iterator, Sequence

import pytest

from .scripted import ScriptedModel, check_scripts


@pytest.fixture
def scripted() -> Iterator[Callable[..., ScriptedModel]]:
    """Make scripted models, and fail the test if any of them was called after its script
    ran out. Memvara swallows most exceptions a model raises, so the extra call would
    otherwise pass unnoticed."""
    made: list[ScriptedModel] = []

    def make(**scripts: Sequence[object]) -> ScriptedModel:
        model = ScriptedModel(**scripts)
        made.append(model)
        return model

    yield make
    check_scripts(made)
