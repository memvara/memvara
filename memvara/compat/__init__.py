"""Running an existing mem0 integration on memvara, and bringing its data along.

Two pieces, and the second is the one that decides whether a migration happens at all:

* `memvara.compat.mem0.Memory` — mem0 2.x's method surface, mapped onto `Memvara`. An
  existing call site keeps working; the three calls that cannot be honestly translated
  (`update`, `delete`-as-erasure, `Memory.from_config`) say so at the call site instead
  of quietly meaning something else.
* `memvara.compat.mem0_import.import_mem0` — an importer that reads mem0's own mutation
  log, `~/.mem0/history.db`, and replays it. Phase 1 is lossless and costs zero model
  calls; it turns a log mem0 keeps but cannot query into `search(as_of=…)`, `history()`
  and `why()`. Phase 2 is opt-in and pays a model to turn the imported notes into
  structured triples.

Everything here is built on memvara's public API and adds no dependency on mem0 — the
package is not installed and is not needed, either to use the shim or to import a store.

What differs, in the order it will bite:

| mem0                        | here                                                    |
|-----------------------------|---------------------------------------------------------|
| `delete()` erases           | retires — the text stays; `delete_all()`/`reset()` erase |
| a memory id is stable       | it is a *version* id; a supersession mints a new one     |
| `update()` edits in place   | refused; assert the new value instead                    |
| `search(threshold=0.1)`     | no default floor — measure with `calibrate_min_score`    |
| `history()` per memory row  | per fact slot, and synthesized back into mem0's shape    |
| one UPDATE event            | an ADD and a DELETE, because that is what happened       |
"""

from .mem0 import ENTITY_FILTERS, Mem0CompatError, Mem0DeletionWarning, Memory
from .supermemory_import import (SupermemoryError, SupermemoryReceipt,
                                 import_supermemory, read_supermemory_key)
from .mem0_import import (
    ContestedSlot,
    HistoryRow,
    ImportReceipt,
    import_mem0,
    read_history_db,
)
from ._notes import (
    NOTE_PREDICATE,
    SUBJECT_PREFIX,
    ensure_note_predicate,
    note_subject,
)

__all__ = [
    # the shim
    "Memory", "Mem0CompatError", "Mem0DeletionWarning", "ENTITY_FILTERS",
    # the importer
    "import_mem0", "ImportReceipt", "ContestedSlot", "HistoryRow", "read_history_db",
    # moving off supermemory
    "import_supermemory", "SupermemoryReceipt", "SupermemoryError",
    "read_supermemory_key",
    # how an opaque memory string is stored
    # `ensure_note_predicate` is public because it has three consumers now — the
    # shim, the importer, and the CrewAI adapter, which stores opaque sentences for
    # the same reason. A third module importing it from `_notes` meant the leading
    # underscore had stopped being true.
    "NOTE_PREDICATE", "SUBJECT_PREFIX", "note_subject", "ensure_note_predicate",
]
