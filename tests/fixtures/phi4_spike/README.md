# phi-4-mini extraction spike, 2026-09-03 — the pollution guard's fixture

Eighteen extraction outputs from `phi-4-mini-instruct` (Q8_0, llama.cpp, a 4-core Ice Lake
box), over three real conversational turns, under six configurations: the claim cap at
`12`, `24`, `32` and `none`, and the predicate-list A/B (`A` with the 64-predicate
vocabulary, `B` without). Each `claims/<config>-<episode>.json` is the model's output
verbatim, as a list of claim dicts. `episodes.json` holds the three turns. `keys.json` holds
the hand-written gold: for each turn, the facts it states, each as an object regex, a
subject regex, and the predicates that express it — plus the generic predicates that count
only when the subject names the entity.

The scorer in `tests/test_pollution.py` is the spike's own, moved here so the guard's
measured effect is a test rather than a memory: hit, wrong-predicate, duplicate, unkeyed,
and keyed facts found. The numbers the guard is held to are in that file.

Recorded, not regenerated, with one transformation stated plainly. The spike ran over
three of the operator's own working notes, which do not belong in a public repository. So
the three turns here are paraphrases that keep every fact the gold keys score, and the
identifying values in them — a profiler's name, two ports, a hostname, a vendor, a plan
size, a row limit, a vector width — are substituted, identically, in the turns, in every
claim the model emitted and in the gold regexes. Nothing else in the model's output is
touched: not a predicate, not a subject, not the order, not a confidence. The
substitution keeps every pattern the guard and the scorer use — digits stay digits, the
domain stays `.dev`, the word `port` stays where it was — and
`test_the_fixture_is_what_was_measured` holds the scorer's counts at exactly the values
the raw spike produced, which is the check that the transformation changed nothing the
measurement rests on. The raw spike, scripts included, stays uncommitted in
`memvara-cloud/local/phi4-cpu-spike-2026-09-03/`.
