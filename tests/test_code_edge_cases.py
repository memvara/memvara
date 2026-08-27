from pathlib import Path

from memvara import CodeIndex, CodeMemory, HashingEmbedder, Memvara, SymbolChange, SymbolKind


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def symbols_of(index: CodeIndex, kind: SymbolKind):
    return [symbol for symbol in index.snapshot.symbols.values() if symbol.kind is kind]


def test_index_extracts_async_symbols_and_multiple_assignments(tmp_path):
    write(tmp_path / "worker.py", """
LIMIT, RETRIES = 10, 3

class Worker:
    timeout = 5

    async def run(self, item):
        return item

async def load(item):
    return item
""")
    index = CodeIndex.from_directory(tmp_path)
    assert len(symbols_of(index, SymbolKind.ASYNC_METHOD)) == 1
    assert len(symbols_of(index, SymbolKind.ASYNC_FUNCTION)) == 1
    assert {s.name for s in symbols_of(index, SymbolKind.VARIABLE)} == {"LIMIT", "RETRIES"}
    assert {s.name for s in symbols_of(index, SymbolKind.CLASS_VARIABLE)} == {"timeout"}


def test_removed_symbol_retires_all_code_claims(tmp_path):
    path = tmp_path / "payments.py"
    write(path, "def refund(payment_id):\n    return payment_id\n")
    first = CodeIndex.from_directory(tmp_path)
    memory = Memvara(":memory:", embedder=HashingEmbedder(dim=32), user="code")
    code = CodeMemory(memory)
    refund = next(s for s in symbols_of(first, SymbolKind.FUNCTION) if s.name == "refund")
    code.remember_context(refund, "Refunds a payment.")
    code.sync(first, previous=None)
    write(path, "def capture(payment_id):\n    return payment_id\n")
    second = CodeIndex.from_directory(tmp_path)
    code.sync(second, previous=first.snapshot)
    assert code.current_context(refund.id) is None
    for predicate in ("code_context", "code_path", "code_signature", "code_kind"):
        assert all(not claim.is_live() for claim in memory.history(refund.id, predicate, user="code"))


def test_move_does_not_call_context_builder(tmp_path):
    old_path = tmp_path / "payments.py"
    write(old_path, "def refund(payment_id):\n    return payment_id\n")
    first = CodeIndex.from_directory(tmp_path)
    memory = Memvara(":memory:", embedder=HashingEmbedder(dim=32), user="code")
    code = CodeMemory(memory)
    refund = next(s for s in symbols_of(first, SymbolKind.FUNCTION) if s.name == "refund")
    code.remember_context(refund, "Refunds a payment.")
    new_path = tmp_path / "billing" / "payments.py"
    new_path.parent.mkdir()
    old_path.rename(new_path)
    second = CodeIndex.from_directory(tmp_path, previous=first.snapshot)
    calls = []
    code.sync(second, previous=first.snapshot, context_builder=lambda symbol, snapshot: calls.append(symbol.id) or "new")
    assert calls == []
    assert code.current_context(refund.id).context == "Refunds a payment."


def test_context_builder_runs_only_for_changed_symbol(tmp_path):
    path = tmp_path / "payments.py"
    write(path, "def refund(payment_id):\n    return payment_id\n\ndef capture(payment_id):\n    return payment_id\n")
    first = CodeIndex.from_directory(tmp_path)
    memory = Memvara(":memory:", embedder=HashingEmbedder(dim=32), user="code")
    code = CodeMemory(memory)
    for symbol in symbols_of(first, SymbolKind.FUNCTION):
        code.remember_context(symbol, f"Original {symbol.name} context.")
    write(path, "def refund(payment_id):\n    return str(payment_id)\n\ndef capture(payment_id):\n    return payment_id\n")
    second = CodeIndex.from_directory(tmp_path)
    calls = []
    code.sync(second, previous=first.snapshot, context_builder=lambda symbol, snapshot: calls.append(symbol.name) or f"Updated {symbol.name} context.")
    assert calls == ["refund"]
    capture = next(s for s in symbols_of(second, SymbolKind.FUNCTION) if s.name == "capture")
    assert code.current_context(capture.id).context == "Original capture context."


def test_invalid_python_remains_file_visible_but_has_no_symbols(tmp_path):
    path = tmp_path / "broken.py"
    write(path, "def broken(:\n    pass\n")
    index = CodeIndex.from_directory(tmp_path)
    assert "broken.py" in index.snapshot.files
    assert index.snapshot.by_path("broken.py") == ()


def test_class_variable_fingerprint_does_not_change_when_unrelated_method_changes(tmp_path):
    path = tmp_path / "service.py"
    write(path, """
class Service:
    timeout = 10

    def run(self):
        return 1
""")
    first = CodeIndex.from_directory(tmp_path)
    timeout_before = next(s for s in symbols_of(first, SymbolKind.CLASS_VARIABLE) if s.name == "timeout")
    write(path, """
class Service:
    timeout = 10

    def run(self):
        return 2
""")
    second = CodeIndex.from_directory(tmp_path)
    timeout_after = next(s for s in symbols_of(second, SymbolKind.CLASS_VARIABLE) if s.name == "timeout")
    assert timeout_before.fingerprint == timeout_after.fingerprint
    changes = second.diff(first.snapshot)
    assert any(change is SymbolChange.CHANGED for change, before, after in changes if before and after and before.name == "run")
    assert any(change is SymbolChange.UNCHANGED for change, before, after in changes if before and after and before.name == "timeout")


def test_duplicate_symbols_are_not_silently_merged_on_move(tmp_path):
    path = tmp_path / "a.py"
    write(path, "def validate(x):\n    return x\n")
    first = CodeIndex.from_directory(tmp_path)
    old = next(s for s in symbols_of(first, SymbolKind.FUNCTION) if s.name == "validate")
    path.unlink()
    write(tmp_path / "b.py", "def validate(x):\n    return x\n\ndef validate(x):\n    return x\n")
    second = CodeIndex.from_directory(tmp_path, previous=first.snapshot)
    changes = second.diff(first.snapshot)
    assert sum(change is SymbolChange.MOVED for change, _, _ in changes) == 0
    assert any(change is SymbolChange.REMOVED and before.id == old.id for change, before, _ in changes)


def test_renamed_symbol_with_unique_implementation_keeps_identity(tmp_path):
    path = tmp_path / "service.py"
    write(path, "def old_name(x):\n    return x + 1\n")
    first = CodeIndex.from_directory(tmp_path)
    old = next(s for s in symbols_of(first, SymbolKind.FUNCTION) if s.name == "old_name")
    write(path, "def new_name(x):\n    return x + 1\n")
    second = CodeIndex.from_directory(tmp_path, previous=first.snapshot)
    new = next(s for s in symbols_of(second, SymbolKind.FUNCTION) if s.name == "new_name")
    assert new.id == old.id
    assert any(change is SymbolChange.RENAMED for change, _, _ in second.diff(first.snapshot))


def test_swap_of_identical_functions_does_not_transfer_identity(tmp_path):
    path = tmp_path / "service.py"
    write(path, "def alpha(x):\n    return x\n\ndef beta(x):\n    return x\n")
    first = CodeIndex.from_directory(tmp_path)
    ids = {s.name: s.id for s in symbols_of(first, SymbolKind.FUNCTION)}
    write(path, "def beta(x):\n    return x\n\ndef alpha(x):\n    return x\n")
    second = CodeIndex.from_directory(tmp_path, previous=first.snapshot)
    current = {s.name: s.id for s in symbols_of(second, SymbolKind.FUNCTION)}
    assert current == ids


def test_signature_change_is_not_treated_as_unchanged(tmp_path):
    path = tmp_path / "service.py"
    write(path, "def validate(x):\n    return x\n")
    first = CodeIndex.from_directory(tmp_path)
    write(path, "def validate(x, strict=False):\n    return x\n")
    second = CodeIndex.from_directory(tmp_path)
    assert any(change is SymbolChange.CHANGED for change, before, after in second.diff(first.snapshot) if before and after)
