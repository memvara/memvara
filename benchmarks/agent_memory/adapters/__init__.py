"""Adapters: the only place in this benchmark that knows about a memory system.

`base` defines the interface. Everything beside it implements it. Adding a system means
adding one module here and one line to `benchmarks/agent_memory/registry.py`, and
touching nothing else — see `benchmarks/agent_memory/CONTRIBUTING.md`.
"""
