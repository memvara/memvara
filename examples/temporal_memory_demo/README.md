# The 90-second demo

A terminal demo of the one thing Memvara does that a vector store cannot: answer *where
did Alice live in March* after she has moved twice, without having overwritten anything.

```bash
python3 examples/temporal_memory_demo/demo.py          # paced, ~90 seconds
python3 examples/temporal_memory_demo/demo.py --fast   # no pauses
```

It needs no API key, no network and no database server — `pip install memvara` and the
command above is the whole setup. Every value it prints is read back out of a real
store; nothing is typed into a string for the screen.

The rules are box-drawing characters, so the demo sets its own output encoding to UTF-8
rather than inheriting the ANSI code page. Without that it dies on Windows four lines in,
where `sys.stdout` defaults to cp1252 and cp1252 has no `─`.

## The six beats

| | | |
|---|---|---|
| 0–10s | The problem | An agent does not merely forget. It remembers the value you replaced. |
| 10–30s | Three writes | One person, three cities, three dates. No model is called. |
| 30–50s | What is true now | `get_all()` → New York. Every memory layer can do this. |
| 50–65s | What was true then | `get_all(as_of=…)` → London in March, Berlin in January. |
| 65–80s | The record | `history()` — every value, its interval, and whether it *ended* or was *retired* — then `why()` on the live one, which names what it replaced. |
| 80–90s | The close | The name, the claim, and where to go next. |

## What it prints

[`expected-output.txt`](expected-output.txt) is the exact transcript, and it is the
golden file rather than a copy: `tests/test_examples.py` runs `demo.py --fast` and
asserts the output matches it byte for byte. `--fast` removes the pauses and changes
nothing else, so the paced recording shows the same lines at reading speed.

If you change `demo.py`, regenerate it in the same commit:

```bash
python3 examples/temporal_memory_demo/demo.py --fast \
  > examples/temporal_memory_demo/expected-output.txt
```

## The published GIF

The one the README shows lives at
<https://github.com/memvara/memvara/releases/latest/download/demo.gif>, put there by
[`.github/workflows/demo-gif.yml`](../../.github/workflows/demo-gif.yml) when a release
is published. That address always resolves to the newest release that is neither a draft
nor a prerelease, so the README embeds it once and never changes again.

**Nothing is checked in**, and that is the reason for the release asset rather than an
omission. The GIF is 1.1 MB against a 2.3 MiB packed history, so committing it would
make one binary about a third of every clone — and each regeneration would add another
copy, permanently, because git keeps what it has seen. A release asset sits outside the
object database and can be replaced instead of accumulated. `demo.gif` is in
`.gitignore` at every depth, because the command below writes it into whatever directory
you run it from.

**The workflow attaches to a release; it never creates one.** Publishing is a decision —
`release.yml` waits on a human for exactly that reason — so a person publishes the
release and this job only decorates it. To attach the GIF to a release that already
shipped, or to republish after `demo.py` changes, run it from the Actions tab with the
tag as its input.

On a dispatch the tag names **where the asset goes**, not what gets recorded: the GIF
comes from the ref you started the run on. Those are the same thing on a release and
come apart on a backfill, which is the case dispatch exists for — `v0.9.0` predates the
demo, so there is no `record_gif.py` at that tag to run.

It runs `record_gif.py` in deterministic mode, then reads the file back: frame count, how
many frames carry ink, and whether the delays still add up to about ninety seconds. Those
are the three ways this can fail while exiting 0 — a font that loaded but drew nothing, a
run that stopped part-way, and pacing that collapsed.

Two things that look like the job failing and are not. A **prerelease does not move
`latest`**, so publishing `v1.0.0rc1` attaches the GIF to it and leaves the README
showing the last full release's. And **GitHub caches README images through its camo
proxy**, so a replaced asset can take a while to appear on the rendered page — read the
release's own asset when you want to know what was published.

## Recording it yourself

Three routes, in the order to try them. The first two want tooling; the third is what the
workflow uses: only Pillow, and it works in a container with no terminal at all.

### With VHS (reproducible, and the recommended one)

[`demo.tape`](demo.tape) drives [VHS](https://github.com/charmbracelet/vhs), which
records a headless terminal — so the output does not depend on the recording machine's
font, colour theme, window size, or how steadily anybody types, and re-running it
produces the same GIF.

```bash
brew install vhs                                   # or: go install github.com/charmbracelet/vhs@latest
vhs examples/temporal_memory_demo/demo.tape        # writes demo.gif beside the tape
```

The tape pins the font size, the window to 1000×760, and the theme. Change them there
rather than in your terminal, or the next person's recording will not match yours.

### With asciinema (smaller, and text you can select)

```bash
asciinema rec demo.cast --command "python3 examples/temporal_memory_demo/demo.py"
agg demo.cast demo.gif --font-size 16 --theme asciinema     # if you need a GIF
```

An `.cast` file is a few kilobytes and the viewer lets a reader copy the commands out of
it, which a GIF does not. Upload with `asciinema upload demo.cast`.

### With no terminal tooling at all

[`record_gif.py`](record_gif.py) runs the demo and encodes the frames itself. Pillow is
the only thing it needs beyond the standard library:

```bash
pip install Pillow
python3 examples/temporal_memory_demo/record_gif.py demo.gif
```

This is the route for a container or a CI runner, where the two above do not work: VHS
wants a Go toolchain, `ttyd` and `ffmpeg`, and asciinema wants a terminal to attach to.
It writes one frame per output event rather than at a frame rate, because ninety seconds
of still text at 10fps is nine hundred frames that are almost all identical.

**It is deterministic, and that is the point.** The demo runs for real — a real store,
real `memvara` calls — but on a *virtual clock*: `time.sleep` advances a counter instead
of waiting, so the run takes about two seconds and the frame delays come from the
schedule `demo.py` already declares (`BEATS`, and the per-line holds). The same source
therefore always produces the same bytes, which is what lets a check regenerate the GIF
and tell whether it is stale. A measured recording can never answer that: two ninety-
second runs of the same script differ by a few milliseconds a frame, so they differ in
bytes while being equally correct.

`--live` is the measured recording, under a pty, taking the full ninety seconds. Use it
if you want wall-clock evidence rather than a replay; do not use it for anything a check
compares. It is POSIX only, because a pty is — on Windows it refuses and says to drop the
flag. The default runs there like anywhere else. Both modes were confirmed to produce **pixel-identical frames** — the virtual
clock is a statement about time, not about what the program printed.

It sizes the canvas to the content instead of to a fixed window, so the result is
tighter than the tape's: 811×534 and 1.1 MB against the tape's 1000×760, which is
four-fifths empty for the first half of the run. Pass `--cols 98 --rows 37` if you want
the two to match. `--fast` replays the unpaced schedule, which is for checking the
pipeline rather than for publishing.

Nothing wraps, so a `--cols` narrower than the demo's longest line would clip its tail
off the canvas — and a clipped frame is still full of ink, so no check downstream would
notice. It refuses instead, naming the line and the width it needs. The longest line
today is 78 characters against a default of 80.

**Look at the frames before you publish one.** Nothing in the script can tell a correct
frame from a blank one, and a missing font renders as tofu rather than as an error.

### For a screen capture with narration

Run the paced version and read the on-screen lines — they are written to be spoken. Two
things to get right:

1. **Start recording before the command.** The first beat is the problem statement and
   it is the reason anybody watches the rest.
2. **Do not resize mid-run.** The rules are 62 characters wide; a terminal narrower than
   ~68 columns wraps them and the beats stop reading as beats.

A terminal at 100×40 with a 16pt font fills a 1000×760 frame, which is the size the tape
uses and a reasonable one for embedding in a README.

---

Next: [the examples index](../README.md) · [Quickstart](../../docs/getting-started/quickstart.md) · [Bitemporal memory](../../docs/concepts/bitemporal-memory.md)
