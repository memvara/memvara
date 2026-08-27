from pathlib import Path

from memvara import CodeIndex, CodeMemory, HashingEmbedder, Memvara, SymbolChange, SymbolKind


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_python_index_extracts_file_class_methods_and_variables(tmp_path):
    _write(
        tmp_path / "payments.py",
        """
class PaymentService:
    retries = 3

    def refund(self, payment_id: str) -> bool:
        return payment_id in self.pending


def helper(value):
    return value
""",
    )

    index = CodeIndex.from_directory(tmp_path)
    kinds = {symbol.kind for symbol in index.snapshot.symbols.values()}

    assert SymbolKind.MODULE in kinds
    assert SymbolKind.CLASS in kinds
    assert SymbolKind.METHOD in kinds
    assert SymbolKind.CLASS_VARIABLE in kinds
    assert SymbolKind.FUNCTION in kinds


def test_path_move_preserves_symbol_identity(tmp_path):
    old_path = tmp_path / "payments.py"
    _write(old_path, "def refund(payment_id):\n    return payment_id\n")
    first = CodeIndex.from_directory(tmp_path)
    old_function = next(s for s in first.snapshot.symbols.values() if s.name == "refund")

    new_path = tmp_path / "billing" / "payments.py"
    new_path.parent.mkdir()
    old_path.rename(new_path)
    second = CodeIndex.from_directory(tmp_path, previous=first.snapshot)
    new_function = next(s for s in second.snapshot.symbols.values() if s.name == "refund")

    assert new_function.id == old_function.id
    assert new_function.path == "billing/payments.py"
    changes = second.diff(first.snapshot)
    assert any(change is SymbolChange.MOVED for change, _, _ in changes)


def test_implementation_change_is_detected(tmp_path):
    path = tmp_path / "payments.py"
    _write(path, "def refund(payment_id):\n    return payment_id\n")
    first = CodeIndex.from_directory(tmp_path)

    _write(path, "def refund(payment_id):\n    return str(payment_id)\n")
    second = CodeIndex.from_directory(tmp_path)
    changes = second.diff(first.snapshot)

    assert any(change is SymbolChange.CHANGED for change, before, after in changes if before and after)


def test_code_memory_retires_old_context_on_symbol_change(tmp_path):
    path = tmp_path / "payments.py"
    _write(path, "def refund(payment_id):\n    return payment_id\n")
    first = CodeIndex.from_directory(tmp_path)

    memory = Memvara(":memory:", embedder=HashingEmbedder(dim=32), user="code")
    code = CodeMemory(memory)
    refund = next(s for s in first.snapshot.symbols.values() if s.name == "refund")
    code.remember_context(refund, "Refunds a payment by idempotently returning the payment.")

    _write(path, "def refund(payment_id):\n    return str(payment_id)\n")
    second = CodeIndex.from_directory(tmp_path)
    code.sync(
        second,
        previous=first.snapshot,
        contexts={refund.id: "Refunds a payment after normalizing the payment id to a string."},
    )

    history = memory.history(refund.id, "code_context", user="code")
    assert len(history) == 2
    assert history[0].object.startswith("Refunds a payment by idempotently")
    assert history[0].state == "ended"
    assert history[1].state == "live"


def test_unchanged_symbol_does_not_require_new_context(tmp_path):
    path = tmp_path / "payments.py"
    _write(path, "def refund(payment_id):\n    return payment_id\n")
    first = CodeIndex.from_directory(tmp_path)
    second = CodeIndex.from_directory(tmp_path)

    changes = [c for c in second.diff(first.snapshot) if c[0] is not SymbolChange.UNCHANGED]
    assert changes == []
