# Moving off mem0

Replay mem0's own mutation log. No extraction model, no token cost. The
exit stays the same as the entrance: the store is a file you already have.

```python
import os
from memvara import Memvara, NullLLM
from memvara.compat import import_mem0

mem = Memvara("migrated.db", user="alice", llm=NullLLM())
log = os.path.expanduser("~/.mem0/history.db")
receipt = import_mem0(mem, history_db=log)
print(receipt)
```

`expanduser` is on you. The importer opens the path as given, and a bare
`~/...` fails with sqlite3's "unable to open database file".

`ADD` rows become claims. `UPDATE` rows close the old value and write the
new one, on the clock the log recorded, so `get_all(as_of=...)` still
answers. The importer does not guess `ended` vs `retired` beyond what the
log already named.

There is also a method-level shim, `from memvara.compat import Memory`, if
they want mem0's call surface on this store. `update()` is not implemented:
Memvara does not overwrite a claim in place. That is a `Mem0CompatError`,
not a silent no-op.

Guide, with the numbers from a real run: https://memvara.dev/docs/guide#mem0
