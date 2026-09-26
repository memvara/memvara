"""The weekly tier's real servers: every one of the 2,048 combinations
(`switches.every_combination`), one after another. The whole file took 7 minutes 54
seconds on a laptop that was running other tests at the same time."""

from __future__ import annotations

from typing import Callable

import pytest

from .. import switches


@pytest.mark.parametrize("combination", switches.every_combination(),
                         ids=lambda c: c.label)
def test_every_combination_on_a_real_server(
        combination: switches.Combination,
        check_real_server: Callable[[switches.Combination], None]) -> None:
    check_real_server(combination)
