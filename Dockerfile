# Engram's MCP server, containerised. See docs/DEPLOY.md for how a client launches it.
#
# The shape of this file is decided by one fact: the server speaks JSON-RPC 2.0 over
# **stdio**, not HTTP. So there is no EXPOSE, no port, no HEALTHCHECK and no supervisor,
# and each of those absences is a decision rather than an omission:
#
# * **No EXPOSE.** There is no socket. The client writes requests on the container's
#   stdin and reads responses from its stdout, so `docker run -i` *is* the transport.
# * **No HEALTHCHECK.** A healthcheck runs a second process in the container, and a
#   second process cannot observe the one holding stdio — it could only prove that
#   `python -c "import engram"` works, which was already true when the image was built.
#   Liveness here is the pipe: if the server dies, the client's next read returns EOF
#   immediately, which is a better signal than a 30-second probe interval.
# * **No CMD.** `engram-mcp` takes no arguments and exits 2 on any (see server/cli.py),
#   so an argument list appended by `docker run` is a startup failure, not configuration.
#   Everything is environment, because an environment block is what an MCP client's
#   settings file can actually set.
# * **`-t` is wrong, not merely unnecessary.** A TTY turns on line discipline: echo, and
#   `\n` → `\r\n` on output. Both corrupt a newline-framed JSON stream. `docker run -i`,
#   never `-it`.
#
# Nothing sets PYTHONUNBUFFERED. It is the usual container reflex and it would be cargo
# cult here: `serve_stdio` flushes stdout after every single message, precisely because a
# buffered reply to a blocked client is a hung session, and stderr has been line-buffered
# unconditionally since Python 3.9. There is nothing left for the variable to fix.

ARG PYTHON_VERSION=3.13


# -- build ---------------------------------------------------------------------------
# Separate stage so that pip, hatchling, the wheel cache and the source tree stay out of
# the shipped image. What crosses the line is one directory: the finished venv.
FROM python:${PYTHON_VERSION}-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

# A venv rather than the system site-packages, so the runtime stage copies one
# self-contained tree and inherits nothing from Debian's Python packaging.
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY engram ./engram

# `.` and not `-e .`: an editable install would leave the runtime image pointing at
# /src, which is not copied forward, and the failure would be an ImportError on first
# launch rather than at build time.
RUN pip install .

# The venv's own pip and setuptools are build tooling that a running server never calls.
# Deleting them *here*, before the runtime stage copies the venv, is what makes it a
# saving: 17 MB unpacked and 3.7 MB compressed, measured against the same build with this
# line commented out. The same delete performed in the runtime stage would save nothing —
# see the note beside the one that is there anyway.
RUN rm -rf /opt/venv/lib/python*/site-packages/pip \
           /opt/venv/lib/python*/site-packages/pip-*.dist-info \
           /opt/venv/lib/python*/site-packages/setuptools \
           /opt/venv/lib/python*/site-packages/setuptools-*.dist-info \
           /opt/venv/lib/python*/site-packages/pkg_resources \
           /opt/venv/bin/pip*


# -- runtime -------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim

# `image.source` is omitted rather than guessed: this repository has no remote configured,
# and a label pointing at a URL nobody owns is worse than a missing one — it is what a
# scanner, a registry listing and `docker inspect` will all cite as the provenance. Add it
# when the repository has somewhere to point at.
LABEL org.opencontainers.image.title="engram-mcp" \
      org.opencontainers.image.description="Bitemporal memory for AI agents, as an MCP stdio server." \
      org.opencontainers.image.licenses="Apache-2.0"

COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# The base image's own pip, and the wheels `ensurepip` would rebuild it from. This one
# saves **no image size at all** and is kept anyway, which is worth being explicit about
# because it looks like the line above: a delete in a later layer is a whiteout over a
# layer that still ships the bytes, so the pull is unchanged and only the running
# container's filesystem shrinks (215 → 206 MiB). What it buys is that there is no
# installer on PATH in a process whose entire pitch is the dependency it does not have.
# A speed bump, not a boundary — anyone with a shell in here can still fetch a wheel.
RUN rm -rf /usr/local/lib/python*/site-packages/pip \
           /usr/local/lib/python*/site-packages/pip-*.dist-info \
           /usr/local/lib/python*/ensurepip \
           /usr/local/bin/pip*

# Unprivileged, and with a real home: SQLite writes its journal beside the database, and
# a process that cannot write the directory gets "attempt to write a readonly database"
# on the first commit rather than at open.
RUN useradd --create-home --uid 10001 engram \
 && mkdir -p /data \
 && chown engram:engram /data

# The mount point, pre-created and owned, but deliberately **not** a VOLUME instruction:
# an implicit anonymous volume would let a forgotten `-v` look like it worked, and the
# memory would then live somewhere the user cannot name. See docs/DEPLOY.md.
#
# `/data` is a directory, not a file, because the store is more than one file — the
# SQLite database, its `-wal` and `-shm` siblings, the `<db>.vecs` mmapped vector matrix
# and the `<db>.embedder.json` fingerprint that says which model wrote it. Bind-mounting
# just `memory.db` would persist the rows and silently discard every embedding.
USER engram
WORKDIR /data

# ENGRAM_DB is deliberately unset. The server refuses to start without it and prints the
# client configuration block, which is the behaviour that stops a misconfigured client
# from remembering into a store that dies with the container. Defaulting it here would
# convert that loud failure into a silent one, which is the whole thing config.py exists
# to prevent.

ENTRYPOINT ["python", "-m", "engram.server"]
