# A model as the ranker at 720 tokens: 182 of 199, twice

**Letting a model pick the turns, and rendering the ones it picks whole, scores 182 of 199
(91.5%) at a median context of 672 tokens, on two independent runs of the same block.** The
replicate differs from the first run on 4 questions, 2 each way, against a reader noise floor
of about 15. That is +11 against the routed-720 arm on the same
retrieval (13 wins, 2 losses), +8 against the prompt-v2 arm, and +10 against the 4,089-token
control. It is the first configuration to beat the control at any budget, and the first to clear
the offline stop rule the ranking sweep set (mean gold-turn coverage 0.93: it reaches 0.9405).
The prediction written before the run was 176, surprised outside 170 to 182; the pre-registered
adoption bar was +8 or better. Both held.

The companion arm, the same model's spans rendered without the turns around them, scores 165 of
199 at a median of 77 tokens. It wins the same aggregation questions and loses preference and
single-session questions wholesale (preference 4 of 12). So the gain is the ranking, not the
compression: the reader needs whole turns, and what it needed at 720 tokens was the right ten.

## What was run

Every arm here reuses the search phase of memvara-routed720 verbatim: k=200 from the local
memvara-cloud stack with the cross-encoder in the server, core 6c0ff6c. The only change is the
rendered block, supplied to the harness through a new knob, `MEMVARA_CONTEXT_FILE` (fork commit
82cf64a), which replaces the provider's rendering with a pre-rendered block per question and
aborts the run if any question is missing from the file. The recorded `contextTokens` follows
the block.

The blocks come from one gpt-5.4 call per question (`local/compress/extract.py`): the routed
role's top-40 turns in MiniLM-L-6 order, the question and its date, and an instruction to copy
out verbatim the spans that bear on the question and omit every excerpt that has none. 199
calls, $2.25, no parse failures. Against the gold labels the filter keeps 315 of the 352 gold
turns in the top-40 (recall 0.895) and 189 of 7,608 non-gold turns (2.5%). The kept span is a
fifth of its turn's tokens at the median.

| arm | run id | block | correct | % | median tokens |
|---|---|---|---|---|---|
| control (cap 15, both roles) | memvara-cap15-control | provider rendering | 172/199 | 86.4 | 4,089 |
| routed-720 | memvara-routed720 | provider rendering | 171/199 | 85.9 | 672 |
| routed-720 + prompt v2 | memvara-routed720-p2 | provider rendering | 174/199 | 87.4 | 672 |
| B: model as ranker | memvara-llmrank720 | kept turns whole, in rank order, then the rest of the routed list, greedy to 720 | **182/199** | **91.5** | 672 |
| A: spans only | memvara-spans720 | the kept spans, one line per kept turn | 165/199 | 82.9 | 77 |
| C + prompt v2 | memvara-llmrankC720-p2 | B's block with an overflowing kept turn rendered as its span and best-fit fill, plus the v2 answer prompt | 181/199 | 91.0 | 706 |
| E: inclusive filter, adaptive rendering | memvara-llmrankE720 | pass-2 filter over the top-60, whole turns when they fit and spans when they do not | 177/199 | 88.9 | 706 |
| B, replicate | memvara-llmrank720-r2 | identical to arm B | **182/199** | **91.5** | 672 |

Per question type:

| arm | ss-user | ss-assistant | ss-preference | multi-session | temporal | knowledge-update |
|---|---|---|---|---|---|---|
| control | 27/28 | 22/22 | 9/12 | 40/53 | 44/53 | 30/31 |
| routed-720 | 27/28 | 20/22 | 6/12 | 44/53 | 45/53 | 29/31 |
| routed-720 + prompt v2 | 27/28 | 20/22 | 9/12 | 45/53 | 44/53 | 29/31 |
| B: model as ranker | 27/28 | 21/22 | 10/12 | 47/53 | 48/53 | 29/31 |
| B, replicate | 28/28 | 22/22 | 9/12 | 47/53 | 47/53 | 29/31 |
| A: spans only | 26/28 | 18/22 | 4/12 | 43/53 | 45/53 | 29/31 |
| C + prompt v2 | 27/28 | 22/22 | 8/12 | 49/53 | 47/53 | 28/31 |

Arm B's gains are where the sweep said the losses were: multi-session and temporal reasoning,
the questions whose gold turns sat at ranks 14 to 37 in the cross-encoder order. Six of the
fourteen questions the budget had cut are correct now (`9d25d4e0`, `c4f10528`, `gpt4_31ff4165`,
`gpt4_7f6b06db`, `gpt4_a1b77f9c`, `gpt4_e061b84f`); the other seven wins are questions whose gold
turns routed-720 had already rendered.

## Offline screen and predictions

Both arms were screened offline before any judged spend, with the sweep's coverage rule and a
judged-upside score (questions newly fully covered that the routed arm got wrong, minus
questions losing full coverage that it got right):

| arm | mean coverage | fully covered | judged upside | predicted | judged |
|---|---|---|---|---|---|
| routed-720 (baseline) | 0.9032 | 164/192 | — | — | 171 |
| B: model as ranker | 0.9405 | 172/192 | +5 (5 up, 0 down) | 176 (170–182) | **182** |
| A: spans only | 0.8891 | 155/192 | −9 (6 up, 15 down) | 162 (155–169) | 165 |
| C + prompt v2 | 0.9474 | 174/192 | +1 against arm B (1 up, 0 down) | 185 (178–192) | 181 |
| E: inclusive filter, adaptive rendering | 0.9575 | 180/192 | +4 against arm B (4 up, 0 down) | 186 (179–193) | 177 |
| B, replicate | 0.9405 | 172/192 | — | 182 (175–189) | 182 |

The screen ordered the first three arms correctly and they landed inside their stated ranges; the fourth, arm E, landed nine below its point prediction and two below its range, and the section on the filter says why: the screen scores coverage, and arm E lost on questions whose coverage was already complete. Arm B beat
its point prediction by six; the five "up" questions were all won, and seven of its thirteen
wins are questions whose gold turns were already rendered by routed-720. Those seven are
reader-side: the block puts the kept turns first and cuts the tail of near-miss turns, which
may be what helped, but on a single run that is a reading, not a measurement.

The third arm stacked two changes on arm B, the overflow spans (offline +1, no losses) and the
v2 answer prompt (+3 on routed-720 on its own), and came back at 181: 4 wins and 5 losses
against arm B. The single offline "up" question, `gpt4_a56e767c`, was won, along with
`88432d0a` and two reader-side questions; five reader-side questions went the other way. The
prompt's earlier gain does not stack on the model-ranked block, and the pre-registered rule
(candidate final only at 182 or better) leaves the arm at no evidence. What the pair of runs does
establish is the mechanism's level: two blocks built the same way, judged twice, at 181 and 182.

## What arm B still misses

| cause | count | questions |
|---|---|---|
| every gold turn rendered, reader wrong | 9 | `71017277`, `07741c45`, `09d032c9`, `37f165cf`, `51a45a95`, `75f70248`, `a2f3aa27`, `d01c6aa8`, `gpt4_2f8be40d` |
| the filter dropped some gold turns | 5 | `gpt4_f420262d` (kept 0 of 3), `88432d0a` (3 of 4), `gpt4_ab202e7f` (2 of 5), `bf659f65` (2 of 3), `gpt4_7abb270c` (5 of 6) |
| kept, but the whole turns overflowed 720 | 1 | `gpt4_a56e767c` (six gold turns kept; four fit) |
| unreachable with this routing or pool | 2 | `ac031881` (assistant gold, routed user-only), `eac54add` (gold outside the top-40) |

Three levers follow from the table and are the next round: an inclusive second filter pass for
questions that count or list things, over a deeper top-60 (the five dropped-gold misses are all
aggregation questions); spans in place of the whole turn only for kept turns that overflow
(recovers `gpt4_a56e767c` offline with no loss); and the v2 answer prompt, which was +3 on its
own. The reader-side nine are the noise floor's territory: the reader disagrees with itself on
7.8% of identical prompts.

## The filter itself: recall, cost, and a smaller model

The filter is one chat call per question. Four variants were measured against the gold labels
before anything else was judged (`extract.py`, `extract2.py`, `extract_mini.py`,
`extract_nano.py`; caches under `local/compress/`):

| filter | list | gold recall | non-gold keep rate | cost per 199 questions |
|---|---|---|---|---|
| gpt-5.4, precise prompt (arm B's) | top-40 | 0.895 (315/352) | 2.5% | $2.25 |
| gpt-5.4, inclusive prompt for count and list questions | top-60 | 0.949 (335/353) | 2.8% | $3.25 |
| union of the two gpt-5.4 passes | top-60 | 0.958 | 2.9% | $5.50 |
| gpt-5.4-mini, precise prompt | top-40 | 0.912 (321/352) | 6.4% | $0.70 |
| gpt-5.4-nano, precise prompt | top-40 | 0.844 (297/352) | 4.0% | $0.19 |

Two things follow. The inclusive prompt recovers most of what the first pass dropped: on the
five aggregation misses it keeps every gold turn (4 of 4, 5 of 5, 6 of 6). And the mini model is
a usable filter: better recall than gpt-5.4's precise pass at a third of the cost, paid for with
a keep rate two and a half times higher, which under a fixed budget is the expensive side.

## The budget binds once the ranking is fixed

Rendering the higher-recall sets the way arm B does, whole turns first, scores *worse* offline
than arm B: the union −1, mini −4, all three sets together −5 in judged upside. The reason is in
the per-question table: on `88432d0a`, `gpt4_7abb270c`, `gpt4_ab202e7f` and `gpt4_a56e767c`
every gold turn is now kept and the block still stops at 716 to 720 tokens with coverage 0.50
to 0.83. Six whole gold turns plus the non-gold turns kept beside them do not fit. Past arm B the
constraint is the budget, not the ranking, and arm A already showed what pure spans cost.

The rendering that answers both (`blocks_from.py`, `MODE=adaptive`): render the kept turns whole
when they all fit inside 720; when they do not, render every kept turn as its span, then upgrade
spans back to whole turns in rank order while room remains; then fill with the rest of the
routed list. A preference question with two kept turns gets them whole; an aggregation question
with eight kept turns gets eight spans and as many whole turns as fit.

| filter set, adaptive rendering | mean coverage | fully covered | judged upside vs arm B |
|---|---|---|---|
| inclusive pass alone, top-60 | 0.9575 | 180/192 | **+4** (`88432d0a`, `gpt4_7abb270c`, `gpt4_a56e767c`, `gpt4_ab202e7f`), 0 down |
| union of both gpt-5.4 passes, top-60 | 0.9575 | 180/192 | +3, 1 down (`0bc8ad93`) |
| arm B's own pass, top-40 | 0.9487 | 174/192 | +1 (`gpt4_a56e767c`), 0 down |
| gpt-5.4-mini alone, top-40 | 0.9401 | 173/192 | −2 (3 up, 5 down) |

Arm E was the first row: one call per question, the inclusive prompt over the top-60, adaptive
rendering, the v1 answer prompt. Predicted 186, surprised outside 179 to 193, adopted at 184 or
better. **It scored 177 of 199 (88.9%)**, below the range: 4 wins and 9 losses against arm B,
and of the four aggregation questions the offline screen said it would recover, only
`gpt4_a56e767c` came in.

| arm | ss-user | ss-assistant | ss-preference | multi-session | temporal | knowledge-update |
|---|---|---|---|---|---|---|
| B: model as ranker | 27/28 | 21/22 | 10/12 | 47/53 | 48/53 | 29/31 |
| E: inclusive filter, adaptive rendering | 28/28 | 21/22 | 5/12 | 46/53 | 48/53 | 29/31 |

The loss is preference, 10 of 12 down to 5 of 12, and the cause is not the rendering: none of
the twelve preference questions was span-rendered in arm E (58 questions were, and on those E
and B are level, 51 against 50 correct). The inclusive prompt kept 5 to 8 turns on the
preference questions where the precise prompt kept 2 to 4; the gold turn was still rendered,
but the block led with more near-miss turns from the same session and the reader lost the
preference among them. Recall was bought with precision on questions whose coverage was
already complete, and the reader paid for it. The remaining discordance is at the noise floor:
13 discordant pairs against B is what two reads of the same block produce.

So the configuration that stands is arm B's: the precise prompt over the top-40, kept turns
whole and first. The adaptive rendering is not refuted on its own merits (it was level with
whole turns where it applied), but it was carried in by a filter that over-keeps, and the two
have not been separated on a judged run.

## Fitted against general

- **Arm B has two runs; the others one.** At 7.8% self-disagreement about 15 judgements move
  on a re-read, so a single run resolves differences of about 8 or more. Arm B's +11 clears
  that, and its replicate landed on the same 182 with 4 discordant questions. Arm A's −6 does
  not clear it on its own, but its offline −9 and the per-type collapse say the same thing;
  arm E's −5 likewise rests on its preference collapse rather than on the net.
- **The filter saw the question.** It is question-conditioned selection at query time, one
  model call over about 2,700 tokens per question ($0.011 at gpt-5.4 prices). It is the same
  kind of step as the cross-encoder, done by a stronger model; it is not ingest-time
  extraction, and the cost per query is an order of magnitude above the reader's context.
  Whether a small model filters as well is measurable offline against the gold labels and has
  not been measured.
- **Top-40 was chosen from the diagnosis.** Every cut question's gold turns ranked 37 or
  better on this sample; on unseen questions a deeper list will be needed, and it costs
  linearly.
- **Same 199 questions, same seed.** Nothing here has been run on the other 301.

## Cost

| item | spend |
|---|---|
| extraction, 199 calls | $2.25 |
| arm B, answer and judge | $1.10 |
| arm A, answer and judge | $0.83 |
| arm C + v2, answer and judge | $1.29 |
| recall pass, 199 calls over the top-60 | $3.25 |
| gpt-5.4-mini and nano filter measurements | $0.89 |
| arm E, answer and judge | $1.15 |
| arm B replicate, answer and judge | $1.11 |

Gateway balance after the replicate: 14.61 of the 29.90 the key started with; $15.29 of the
user's $20 cap is spent and $4.71 remains under it.

## What this means for memvara

None of this touched memvara core. Retrieval is core 6c0ff6c as shipped, with three read
options set in the local stack (`read_max_episodes=200`, the cross-encoder reranker,
`read_rerank_top_n=200`); everything after the cross-encoder happened in the harness provider
and, for these arms, in a file the provider read. That was deliberate: a judged arm costs a
dollar and ten minutes this way, and only a configuration that has earned it gets ported.

The configuration that has earned it is a pipeline of four steps after the cross-encoder, and
each has a natural home:

1. **Role routing** (user turns unless the question asks what the assistant said): the
   provider today; in core, an opt-in beside `intent.py`, since it is a property of the
   question.
2. **A model as the ranker**: one chat call over the top-N reranked turns that returns the
   turns bearing on the question and the verbatim span in each. In core this is a read option
   beside `read_reranker`, an `LLMFilter` that takes the configured model, so the same
   `recall()` call that reranks can also filter. It is a query-time cost: about 2,700 input
   tokens per question at top-40, 4,000 at top-60, or $0.011 and $0.016 at gpt-5.4 prices,
   $0.0035 with gpt-5.4-mini. The deterministic fast path stays the default; this is what a
   caller turns on when a question is worth a model call.
3. **Adaptive rendering under a token budget**: whole turns when the kept set fits, spans
   when it does not, upgraded back to whole turns while room remains. The span is part of the
   filter's return value, so the caller can render either. In core this is the budgeted
   rendering `recall()` does not yet have; the provider's `fillToBudget` is the reference.
4. **Fill with the rest of the reranked list**: unchanged.

The memvara-cloud `_clone()` gap (read options dropped on the per-request clone, chip filed) has
to close before any of it reaches the hosted service, or the hosted number will be the
cap-15 number.

Not done, and not to be read as done: the other 301 questions have not been run on any of
these arms (the user's decision stands until the 199 number is where they want it); arm B is the only configuration with a replicate; and the filter has only been measured with the question in
hand, which is the query-time design and not an ingest-time one.

## Addendum, 2026-09-04 evening: the intent-routed arm, and where the proxy stops working

One more arm was screened and judged after the replicate. The inclusive filter pass with the
adaptive rendering is applied only on the 98 questions an aggregation rule fires on (how many,
how much, total, all, every, each, list, how long, days, weeks, months, between), and arm B's
precise pass with whole turns everywhere else. Offline it was the best profile of anything
screened: 178 of 192 fully covered and a judged upside of +3 against arm B with no losses.

| arm | run id | correct | % | median tokens | vs arm B |
|---|---|---|---|---|---|
| I: intent-routed filter | memvara-intent720 | 181/199 | 91.0 | 673 | 3 wins, 4 losses, net −1 |

Predicted 185 (178 to 192); landed at 181, inside the range and below the adoption bar of 184.
Of the three questions the screen said it would recover, one came in. Four judged runs of the
mechanism now sit at 181 or 182 (B, its replicate, C + v2, I). The coverage proxy that ordered
the earlier arms correctly no longer predicts judged movement at this level: +3 predicted, −1
delivered. The remaining misses on the 199 are at the reader's noise floor, and further
rendering variants are not worth judged spend. What is left to measure is the reader itself
(several samples with a vote), the competitor through the same harness, and the other 301
questions.

## Parity on the shipped path, 2026-09-05

The mechanism as shipped in core (routing by the question's role, then the selector over the
routed top-40 on the customer's key, gpt-5.4-mini by decision 2), run through the local stack
with the harness rendering the server's order kept-first to 720 tokens, gpt-5.4 reader and judge.

| run | run id | correct | % | median tokens | note |
|---|---|---|---|---|---|
| ranked, shipped path | memvara-ranked-parity2 | 177/199 | 88.9 | 549 | prediction 178 (171–185) held |
| reranked twin, same stack, MEMVARA_RANKED off | memvara-ranked-twin | 135/199 | 67.8 | 567 | both roles at 720: the failure routing removes |
| paired, ranked minus twin | | +42 | | | 45 wins, 3 losses, 151 ties; gate was +8 |

Against the earlier arms the shipped path is +6 over routed-720 (171), +5 over the 4,089-token
control (172), and −5 against arm B (182, gpt-5.4 as the selector on offline blocks). The
offline screens on the shipped path, on the scorer as reviewed: unrouted 0.879 gold recall at a
0.119 keep rate, routed 0.935 at 0.068. One judged run each; reader noise is 7.8% on identical
prompts. Costs: routed screen $0.72, ranked judging $1.08, twin search and judging about $1.20.
