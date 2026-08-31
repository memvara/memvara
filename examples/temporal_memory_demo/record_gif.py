"""Record `demo.py` to an animated GIF, with no terminal tooling installed.

    python3 examples/temporal_memory_demo/record_gif.py demo.gif

This is the third of the three recording routes in `README.md`, and the one to reach for
when the first two are not available: VHS wants a Go toolchain, `ttyd` and `ffmpeg`, and
asciinema wants a terminal to attach to. Neither exists in a bare container, which is
where this was written. The only thing it needs beyond the standard library is Pillow.

**The demo runs for real either way.** The transcript is whatever `demo.py` prints today
— a real store, real `memvara` calls, no copy kept in step by hand. What the two modes
differ on is where the *timings* come from.

`--live` measures them: the demo runs for its full ninety seconds under a pty, so it
believes it is a terminal and its `flush=True` pacing behaves exactly as it does on
screen, and every frame boundary is an instant the program actually printed at. That is a
recording in the strict sense, and its output is **not reproducible** — two runs of the
same script differ by a few milliseconds a frame, so the bytes differ. It is also POSIX
only, because a pty is; the default below runs everywhere this library does.

The default instead *replays the schedule the demo declares*: `BEATS` and the per-line
holds, which are already in `demo.py` and already the thing that makes it ninety seconds
rather than however long its pauses happen to add up to. The program executes normally
on a virtual clock — `time.sleep` advances a counter instead of waiting — so the run
takes a second or two and **the same source always produces the same bytes**. That is
what lets CI regenerate the GIF and tell whether it is stale, which a measured recording
can never do.

Neither mode fakes the output. The virtual clock is a statement about time, not about
what the program printed.

## Why one frame per output event rather than a frame rate

The demo is ninety seconds of mostly still text. At 10fps that is nine hundred frames,
almost all identical to their neighbour. GIF stores a delay per frame, so the sixty-odd
instants the program prints at reproduce the pacing exactly, and the file stays around a
megabyte instead of thirty.

## Where it differs from `demo.tape`

The tape fixes a 1000x760 window because that is what a screen recording wants. This
sizes the canvas to the content — the longest line the demo prints, and a viewport that
scrolls the way a terminal does — because a README asset four-fifths empty for its first
half is a worse asset. Same script, same pacing, tighter frame. If you need the two to
match, pass `--cols 98 --rows 37`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: Catppuccin Mocha `base` and `text`, which is the theme `demo.tape` names, so a GIF
#: from either route looks like the same terminal.
BACKGROUND = (30, 30, 46)
FOREGROUND = (205, 214, 244)

#: Monospace faces that carry the box-drawing glyph the demo's rules are made of
#: (U+2500) and the em dash in its closing line. Checked in order; the first that exists
#: wins. A proportional face would render the timeline columns crooked, so this list is
#: deliberately all monospace.
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "C:/Windows/Fonts/consola.ttf",
)

#: Seconds the closing frame holds before the loop restarts. Long enough to read the
#: positioning line, short enough not to look like the GIF has stalled.
TAIL_HOLD = 4.0


#: A codepoint no font has. Private Use Area, so nothing standard assigns it and
#: nothing is going to: whatever a face draws for this is that face's .notdef, which is
#: what it also draws for every other character it lacks.
ABSENT = "\ue000"


def has_glyph(font, character: str) -> bool:
    """Whether `font` really carries `character`, rather than drawing tofu for it.

    Asking Pillow for a mask and checking that it has a width does **not** answer this,
    which is how the first version of this check passed a font with no box-drawing at
    all. A missing glyph is not an empty bitmap — it is the .notdef box, a perfectly
    ordinary 10×14 rectangle of ink. Rendering it into a frame produces a wall of tofu,
    and every downstream check reads tofu as content, because it is.

    So compare the bitmap against the one the font draws for a codepoint it certainly
    lacks. Equal bitmaps mean both came out of .notdef.
    """
    return bytes(font.getmask(character)) != bytes(font.getmask(ABSENT))


def load_font(size: int):
    """Pillow, and a face that can draw a box-drawing rule — or a refusal that says how.

    Both failures are worth naming rather than letting them surface as an ImportError or
    a frame full of tofu: this script is the fallback route, so the person running it has
    already found that the other two do not work here.
    """
    try:
        from PIL import ImageFont
    except ModuleNotFoundError:
        raise SystemExit(
            "This recorder needs Pillow to draw frames: pip install Pillow\n"
            "It is the only dependency beyond the standard library, and it is not a "
            "dependency of memvara itself — nothing in the library needs it.")

    for path in FONT_CANDIDATES:
        if Path(path).exists():
            font = ImageFont.truetype(path, size)
            # The rules are made of U+2500 and the closing line uses an em dash. A face
            # missing either renders that character as tofu, silently.
            if all(has_glyph(font, character) for character in "\u2500\u2014"):
                return font
    raise SystemExit(
        "No monospace font carrying U+2500 and U+2014 was found. Install one (on Debian: "
        "apt-get install fonts-dejavu-core) or add its path to FONT_CANDIDATES.")


class VirtualClock:
    """`time.monotonic` and `time.sleep`, with no wall clock behind either.

    `demo.py` does `import time` and calls `time.sleep` and `time.monotonic` through that
    module object, so replacing the module's `time` attribute is enough to put the whole
    Pacer — the per-line holds *and* `until()`'s beat boundaries, which are computed as a
    difference of two `monotonic()` readings — onto this clock. The demo needs no
    knowledge that it is being replayed, and nothing here has to restate its schedule.
    """

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, seconds)


class TimestampedWriter:
    """A `sys.stdout` that records when, on the virtual clock, each write happened."""

    def __init__(self, clock: VirtualClock) -> None:
        self.clock = clock
        self.events: list[tuple[float, str]] = []

    def write(self, text: str) -> int:
        if text:
            self.events.append((self.clock.now, text))
        return len(text)

    def flush(self) -> None:
        pass

    def reconfigure(self, **_kwargs: object) -> None:
        """`demo.main` sets its own encoding; there is no encoding to set on a list."""


def simulate(fast: bool) -> tuple[list[tuple[float, str]], float]:
    """Run `demo.py` in this process on a virtual clock, returning the same events.

    Deterministic because nothing consults a real clock: every instant in the result is a
    number the demo's own schedule produced. Runs in about a second, which is what makes
    it usable as a CI check rather than a ninety-second job.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_memvara_demo", HERE / "demo.py")
    if spec is None or spec.loader is None:                      # pragma: no cover
        raise SystemExit(f"could not import {HERE / 'demo.py'}")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)

    clock = VirtualClock()
    writer = TimestampedWriter(clock)
    real_time, real_stdout = demo.time, sys.stdout
    demo.time, sys.stdout = clock, writer
    try:
        exit_code = demo.main(["--fast"] if fast else [])
    finally:
        demo.time, sys.stdout = real_time, real_stdout
    if exit_code != 0:                                           # pragma: no cover
        raise SystemExit(f"demo.py returned {exit_code}; nothing was rendered.")
    return writer.events, clock.now


def record(command: list[str], cols: int, rows: int) -> tuple[list[tuple[float, str]], float]:
    """Run `command` under a pty, returning `(elapsed, text)` events and the duration.

    A pty rather than a pipe, and not only so the child prints in colour it does not use:
    a program writing to a pipe is block-buffered by default, which would collapse ninety
    seconds of pacing into one burst at exit and make every frame boundary meaningless.

    `pty`, `fcntl` and `termios` are imported here rather than at the top of the file
    because none of them exists on Windows, and importing them there made the *default*
    mode — which uses none of them — an ImportError on a platform where it works
    perfectly. CI found that: `record_gif.py` would not import at all under
    `py3.13 on windows-latest`.
    """
    try:
        import fcntl
        import pty
        import select
        import struct
        import termios
    except ModuleNotFoundError as exc:                           # pragma: no cover
        raise SystemExit(
            f"--live needs a pty, and this platform has no {exc.name!r}: it is POSIX "
            "only. Drop --live — the default replays the schedule demo.py declares and "
            "produces the same frames, in about two seconds instead of ninety.")

    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
    env = dict(os.environ, TERM="xterm-256color", COLUMNS=str(cols), LINES=str(rows))

    proc = subprocess.Popen(command, stdin=slave, stdout=slave, stderr=slave,
                            env=env, close_fds=True)
    os.close(slave)

    events: list[tuple[float, str]] = []
    start = time.monotonic()
    try:
        while True:
            ready, _, _ = select.select([master], [], [], 0.25)
            if ready:
                try:
                    chunk = os.read(master, 65536)
                except OSError:      # the child closed its end; normal at exit
                    break
                if not chunk:
                    break
                events.append((time.monotonic() - start,
                               chunk.decode("utf-8", "replace")))
            elif proc.poll() is not None:
                break
    finally:
        os.close(master)
    if proc.wait() != 0:
        raise SystemExit(f"{command[1]} exited {proc.returncode}; nothing was recorded.")
    return events, time.monotonic() - start


def screens(events, rows: int, duration: float):
    """Replay the byte stream into one screen per instant the program printed at.

    The demo only ever appends lines — no cursor movement, no clearing — so a full
    terminal emulator would be several hundred lines of code to reach the same answer.
    Anything that stops being true of `demo.py` shows up as a wrong frame rather than
    silently, because the frames are checked by eye before the GIF is published.
    """
    lines: list[str] = []
    pending = ""
    out = []
    for at, chunk in events:
        pending += chunk.replace("\r\n", "\n").replace("\r", "")
        *complete, pending = pending.split("\n")
        lines.extend(complete)
        out.append((at, lines[-rows:]))
    if pending:
        lines.append(pending)
        out.append((duration, lines[-rows:]))

    # Collapse instants that did not change the screen: their time belongs to the frame
    # already on it, and a GIF frame identical to its predecessor is bytes for nothing.
    frames = []
    for at, screen in out:
        if frames and frames[-1][1] == screen:
            continue
        frames.append((at, screen))
    return frames


def refuse_clipped_lines(frames, cols: int) -> None:
    """Stop if the demo prints a line wider than the viewport.

    The canvas is `cols` characters wide and nothing wraps, so a longer line simply runs
    off the right edge and its tail is gone from the published GIF. Every check around
    this one stays green while that happens: the event-stream tests compare what the demo
    *printed*, and the frames are full of ink either way. The longest line today is 78
    characters against a default of 80, so this has two characters of headroom and no
    alarm on it.
    """
    longest = max(((len(line), line) for _at, screen in frames for line in screen),
                  default=(0, ""))
    if longest[0] > cols:
        raise SystemExit(
            f"a line is {longest[0]} characters and the viewport is {cols}, so its tail "
            f"would be clipped off the canvas with nothing to show for it:\n"
            f"    {longest[1]}\n"
            f"Pass --cols {longest[0]} or more, or shorten the line in demo.py — and if "
            "you shorten it, regenerate expected-output.txt in the same commit.")


def render(frames, out: Path, font, cols: int, rows: int) -> None:
    """Draw one GIF frame per instant, sized to the font rather than to fixed numbers.

    `line_h` and `pad` are derived from the face because `--font-size` is a real option:
    they used to be the literals 19 and 20, which are correct at the default size of 16
    and silently wrong above it. At 28 the glyphs are 33 pixels tall on a 19-pixel pitch,
    so every row overlaps the one below and the GIF is unreadable — with an exit code of
    0 and a file of the expected size.
    """
    from PIL import Image, ImageDraw

    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    pad = round(font.size * 1.25)
    width = int(cols * font.getlength("M")) + 2 * pad
    height = rows * line_h + 2 * pad

    images, delays = [], []
    for index, (at, screen) in enumerate(frames):
        nxt = frames[index + 1][0] if index + 1 < len(frames) else at + TAIL_HOLD
        image = Image.new("RGB", (width, height), BACKGROUND)
        draw = ImageDraw.Draw(image)
        for row, line in enumerate(screen):
            draw.text((pad, pad + row * line_h), line, font=font, fill=FOREGROUND)
        # Two colours plus antialiasing greys: a small palette compresses hard, and the
        # demo uses no colour of its own.
        images.append(image.quantize(colors=32, method=Image.Quantize.MEDIANCUT))
        delays.append(max(20, round((nxt - at) * 1000)))

    images[0].save(out, save_all=True, append_images=images[1:], duration=delays,
                   loop=0, optimize=True, disposal=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", nargs="?", default="demo.gif", type=Path,
                        help="where to write the GIF (default: demo.gif)")
    parser.add_argument("--cols", type=int, default=80,
                        help="viewport width; 98 matches demo.tape")
    parser.add_argument("--rows", type=int, default=26,
                        help="viewport height, scrolling like a terminal; 37 matches "
                             "demo.tape")
    parser.add_argument("--font-size", type=int, default=16)
    parser.add_argument("--fast", action="store_true",
                        help="use the unpaced schedule — for checking the pipeline "
                             "rather than for publishing")
    parser.add_argument("--live", action="store_true",
                        help="measure the timings from a real ninety-second run under a "
                             "pty instead of replaying the declared schedule. A "
                             "recording in the strict sense, and not reproducible: two "
                             "runs differ by milliseconds and so differ in bytes. POSIX "
                             "only — a pty is not a thing Windows has.")
    args = parser.parse_args(argv)

    font = load_font(args.font_size)

    if args.live:
        command = [sys.executable, str(HERE / "demo.py")]
        if args.fast:
            command.append("--fast")
        print(f"recording {'--fast' if args.fast else 'the paced run (~90s)'}…",
              flush=True)
        events, duration = record(command, args.cols, args.rows)
    else:
        print("replaying the declared schedule…", flush=True)
        events, duration = simulate(args.fast)

    frames = screens(events, args.rows, duration)
    refuse_clipped_lines(frames, args.cols)

    render(frames, args.out, font, args.cols, args.rows)

    # Read the counts back off the file rather than reporting the ones going in. They
    # differ, and the difference is not a rounding error: Pillow merges frames that
    # encode identically, and TAIL_HOLD adds a delay the input never had. This line used
    # to say "63 frames, 90.3s" about a file holding 56 frames and 94.0 seconds, which
    # is the same class of mistake as documentation that describes an older version of
    # the code — it reads as fact and is checkable by nobody.
    from PIL import Image
    with Image.open(args.out) as gif:
        count, encoded = gif.n_frames, 0.0
        for index in range(gif.n_frames):
            gif.seek(index)
            encoded += gif.info["duration"] / 1000
        size = gif.size
    size_mb = args.out.stat().st_size / 1_000_000
    how = "measured" if args.live else "deterministic"
    print(f"{args.out}: {count} frames, {encoded:.1f}s, {size[0]}x{size[1]}, "
          f"{size_mb:.1f} MB ({how})")
    print("Look at it before publishing it — this script cannot tell a correct frame "
          "from a blank one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
