#!/bin/zsh
# Start the local Python SteerLab server — the one-click path the app's
# connection-dot menu runs, and equally usable from a terminal.
#
#   usage: start-local-server.sh [--root <workspace-dir>] [--port N]
#
# What it does, in order:
#   1. Cleans a stale pidfile (a server long gone) silently.
#   2. Refuses plainly if something already listens on the port (exit 3).
#   3. Creates Server/.venv.nosync via `python3 -m venv` if absent, then
#      installs the FULL workbench when anything is missing:
#      `pip install -e "Server[lora,gemmascope]"` — the server plus
#      peft+pypdf (LoRA adapters, PDF stimulus ingestion) and sae-lens
#      (Gemma Scope SAE analysis). First install pulls torch/transformers —
#      this can take MANY minutes; progress streams below so a silent wait
#      never looks like a hang.
#   4. Writes the pidfile `<root>/.steerlab-local-server.pid` ("<pid> <port>",
#      one line) so a relaunched app can adopt this server: `exec` below
#      keeps our PID, so $$ IS the server's pid after the swap.
#      BINDING CONTRACT: the exec line passes `--port "$PORT" --root "$ROOT"`
#      explicitly on the argv. The app's adoption check
#      (LocalServerPidfile.commandLineMatchesSteerLabServer) verifies the
#      recorded pid's command line carries THIS pidfile's port before it
#      enables Stop — so a stale/reused pid can never point Stop at a
#      different SteerLab server. Do not move the port into env or a config
#      file; a server started without `--port` on its argv degrades to the
#      safe not-adoptable state.
#   5. Serves 127.0.0.1:<port> (loopback only — CLAUDE.md security posture)
#      with the artifact root pinned EXPLICITLY via `serve --root`, and with
#      `--dev-open-loopback`: this is the SINGLE-USER local instrument on a
#      personal Mac, where the app and the browser workbench talk to the
#      server with no token. The shipped bare CLI defaults to TOKEN mode
#      (WP-S); this script opts out of it explicitly, on the argv, for the
#      one deployment where "any local user" means "you". The flag REFUSES
#      to start on a non-loopback bind or with a Slurm executor declared, so
#      it cannot be copied onto a shared node by accident.
#      Want the token posture here too? Drop `--dev-open-loopback` from the
#      exec line below; `serve` then prints the token-file path to paste into
#      the app's connection sheet.
#
# The artifact root is STEERLAB_ROOT-or-cwd on the server side, which is why
# `cd Server && python -m …` silently points prompts/experiments/runs at the
# wrong tree. This script never relies on cwd for the root: it always passes
# `--root` — defaulting to $STEERLAB_ROOT if set, else the project root.
#
# Exit codes: 0 server exited cleanly · 3 port already in use ·
#             4 python3 missing · anything else: setup or server failure.
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PROJECT_ROOT="${SCRIPT_DIR:h}"

ROOT="${STEERLAB_ROOT:-$PROJECT_ROOT}"
PORT=8080
while (( $# > 0 )); do
  case "$1" in
    --root)
      [[ $# -ge 2 ]] || { echo "--root requires a directory path"; exit 2 }
      ROOT="$2"; shift 2 ;;
    --port)
      [[ $# -ge 2 ]] || { echo "--port requires a number"; exit 2 }
      PORT="$2"; shift 2 ;;
    *)
      echo "unknown argument: $1"
      echo "usage: start-local-server.sh [--root <workspace-dir>] [--port N]"
      exit 2 ;;
  esac
done

if [[ ! -d "$ROOT" ]]; then
  echo "The workspace root does not exist: $ROOT"
  echo "Pass --root <existing workspace folder> (the folder holding prompts/, experiments/, runs/)."
  exit 2
fi

# ── Pidfile: adoption record for the app across relaunches ────────────────
# One per workspace root. A stale file (its process is gone) is cleaned
# silently; a LIVE one is the "already running" case the port guard reports.
PIDFILE="$ROOT/.steerlab-local-server.pid"
if [[ -f "$PIDFILE" ]]; then
  old_pid=""
  old_port=""
  read -r old_pid old_port < "$PIDFILE" 2>/dev/null || old_pid=""
  if [[ -z "${old_pid:-}" ]] || ! kill -0 "$old_pid" 2>/dev/null; then
    rm -f "$PIDFILE"
  fi
fi

# ── Port guard: fail with a sentence, not a uvicorn traceback ─────────────
listener="$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | tail -n +2 | head -n 1 || true)"
if [[ -n "$listener" ]]; then
  echo "Port $PORT is already in use by: $listener"
  echo "If that is a SteerLab server already running, connect to http://127.0.0.1:$PORT instead of starting another."
  echo "Otherwise stop that process, or rerun with --port <another port>."
  exit 3
fi

# ── Python environment (created on first run, completed when partial) ─────
VENV="$PROJECT_ROOT/Server/.venv.nosync"
PY="$VENV/bin/python"

if [[ ! -x "$PY" ]]; then
  if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 was not found on this Mac."
    echo "Install the Xcode Command Line Tools (xcode-select --install) or Python from python.org, then rerun."
    exit 4
  fi
  echo "First run: creating the Python environment at Server/.venv.nosync…"
  python3 -m venv "$VENV"
  "$PY" -m pip install --upgrade pip
fi

# Full-workbench check (cheap — find_spec imports nothing): the server itself
# plus its optional pieces. Covers a fresh venv, an interrupted install, and
# an older venv created before the extras were part of the one-click path.
if ! "$PY" -c 'import importlib.util, sys; sys.exit(0 if all(importlib.util.find_spec(m) for m in ("steerlab_server", "peft", "pypdf", "sae_lens")) else 1)' >/dev/null 2>&1; then
  echo "Installing the full Python workbench: the server plus peft+pypdf (LoRA adapters, PDF stimulus ingestion) and sae-lens (Gemma Scope SAE analysis)."
  echo "First installs pull torch/transformers and friends — this can take MANY minutes. Progress streams below."
  "$PY" -m pip install -e "$PROJECT_ROOT/Server[lora,gemmascope]"
  echo "Python environment ready."
fi

# ── Serve ─────────────────────────────────────────────────────────────────
echo "Serving artifact root: $ROOT"
echo "Binding 127.0.0.1:$PORT (loopback only — reach a remote box over an SSH tunnel, never a public bind)."
echo "Auth: dev-open loopback (single-user Mac instrument). Drop --dev-open-loopback from this script for token mode."
cd "$PROJECT_ROOT"
# `exec` keeps this shell's PID for the Python server, so the pidfile written
# here names the server itself. Cleaned silently once stale (above, and by
# the app's controller).
echo "$$ $PORT" > "$PIDFILE"
exec env PYTHONUNBUFFERED=1 "$PY" -m steerlab_server.cli serve --port "$PORT" --root "$ROOT" --dev-open-loopback
