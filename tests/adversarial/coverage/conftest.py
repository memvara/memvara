"""The fixture the switch tests use to build the MCP server's store from environment
variables."""

from __future__ import annotations

from typing import Iterator, Mapping, Protocol

import pytest

from memvara import Memvara
from memvara.server import config as server_config
from memvara.server.config import ServerConfig, build_memvara

from ..model_faults.scripted import ScriptedModel, check_scripts


class Serve(Protocol):
    """What the `served` fixture gives a test: build the server's store from `env`, with
    `model` as its model when one is given."""

    def __call__(self, env: Mapping[str, str],
                 model: ScriptedModel | None = None) -> Memvara: ...


#: The variables every store here starts from: an in-memory store with the hashing
#: embedder, no encryption, and no project read from the working directory's git remote.
BASE = {"MEMVARA_DB": ":memory:", "MEMVARA_EMBEDDER": "hashing", "MEMVARA_USER": "u1",
        "MEMVARA_FEATURE_ENCRYPTION": "0", "MEMVARA_FEATURE_PROJECT_SCOPE": "0"}


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> Iterator[Serve]:
    """Build the MCP server's store the way the server does: read the switches from
    environment variables with `ServerConfig.from_env`, then build the store with
    `build_memvara`. A test passes the variables that set the switch it checks, so
    everything between the environment and the store is the server's own code.

    With `model`, the store's model is that scripted stand-in
    (tests/adversarial/model_faults/scripted.py). It is put where the "openai" backend
    would be, by setting `MEMVARA_LLM=openai` and replacing `_openai` in
    memvara/server/config.py, and it records every call, so a test can see what the
    switch changed. Nothing reaches a network.

    When the test ends, every store made is closed, and every model given is checked with
    `check_scripts` for a call that arrived after its script ran out. Memvara swallows
    most exceptions a model raises, so such a call would otherwise pass unnoticed.
    """
    stores: list[Memvara] = []
    models: list[ScriptedModel] = []

    def build(env: Mapping[str, str], model: ScriptedModel | None = None) -> Memvara:
        base = dict(BASE)
        if model is not None:
            monkeypatch.setattr(server_config, "_openai", lambda *args, **kwargs: model)
            base["MEMVARA_LLM"] = "openai"
            models.append(model)
        mem = build_memvara(ServerConfig.from_env({**base, **env}))
        stores.append(mem)
        return mem

    yield build
    for mem in stores:
        mem.close()
    check_scripts(models)
