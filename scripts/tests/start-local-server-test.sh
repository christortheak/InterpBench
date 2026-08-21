#!/bin/zsh
# Self-test for scripts/start-local-server.sh — the testable, pre-serve logic
# only: argument/usage exits, the pidfile stale-cleanup rule, and the busy-port
# guard (exit-code contract with LocalServerController.exitSummary). Never
# creates a venv or starts a server: every case exits before the Python
# environment section. Requires python3 (used as a throwaway port listener).
set -eu

SCRIPT_DIR="${0:A:h}"
START="$SCRIPT_DIR/../start-local-server.sh"
WORK="$(mktemp -d)"
LISTENER_PID=""
SLEEPER_PID=""
cleanup() {
  [[ -n "$LISTENER_PID" ]] && kill "$LISTENER_PID" 2>/dev/null || true
  [[ -n "$SLEEPER_PID" ]] && kill "$SLEEPER_PID" 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT

fail() { echo "FAIL: $1"; exit 1 }

# Runs the script, capturing its exit code in $STATUS without tripping set -e.
run() {
  set +e
  zsh "$START" "$@" >/dev/null 2>&1
  STATUS=$?
  set -e
}

command -v python3 >/dev/null 2>&1 || fail "python3 is required for this self-test"

ROOT="$WORK/workspace"
mkdir -p "$ROOT"
PIDFILE="$ROOT/.steerlab-local-server.pid"

# Case 1: unknown argument → exit 2
run --frobnicate
[[ $STATUS -eq 2 ]] || fail "unknown argument should exit 2, got $STATUS"

# Case 2: --root without a value → exit 2
run --root
[[ $STATUS -eq 2 ]] || fail "--root without value should exit 2, got $STATUS"

# Case 3: nonexistent workspace root → exit 2
run --root "$WORK/does-not-exist"
[[ $STATUS -eq 2 ]] || fail "missing root should exit 2, got $STATUS"

# A throwaway loopback listener on a free port makes the port guard fire
# (exit 3) BEFORE the venv/serve sections — the busy-port cases below piggyback
# on it to exercise pidfile handling without ever starting a real server.
PORT="$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()')"
python3 -c "import socket, time
s = socket.socket()
s.bind((\"127.0.0.1\", $PORT))
s.listen(1)
time.sleep(60)" &
LISTENER_PID=$!
for _ in {1..50}; do
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 && break
  sleep 0.1
done
lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 || fail "test listener never came up on port $PORT"

# Case 4: busy port → exit 3
run --root "$ROOT" --port "$PORT"
[[ $STATUS -eq 3 ]] || fail "busy port should exit 3, got $STATUS"

# Case 5: a STALE pidfile (dead pid) is cleaned silently on the way in
DEAD_PID="$(python3 -c 'import sys; print(2**22)')"  # beyond macOS pid range
echo "$DEAD_PID 8080" > "$PIDFILE"
run --root "$ROOT" --port "$PORT"
[[ $STATUS -eq 3 ]] || fail "busy port with stale pidfile should exit 3, got $STATUS"
[[ ! -f "$PIDFILE" ]] || fail "stale pidfile (dead pid) should have been removed"

# Case 6: a garbled pidfile counts as stale and is cleaned
echo "not-a-pid" > "$PIDFILE"
run --root "$ROOT" --port "$PORT"
[[ $STATUS -eq 3 ]] || fail "busy port with garbled pidfile should exit 3, got $STATUS"
[[ ! -f "$PIDFILE" ]] || fail "garbled pidfile should have been removed"

# Case 7: a LIVE pidfile is preserved — it is the adoption record, not debris
sleep 60 &
SLEEPER_PID=$!
echo "$SLEEPER_PID $PORT" > "$PIDFILE"
run --root "$ROOT" --port "$PORT"
[[ $STATUS -eq 3 ]] || fail "busy port with live pidfile should exit 3, got $STATUS"
[[ -f "$PIDFILE" ]] || fail "live pidfile must not be removed"

# Case 8 (text-level): the serve exec line passes --port and --root
# EXPLICITLY on the argv. The app's adoption check
# (LocalServerPidfile.commandLineMatchesSteerLabServer) binds pid→port
# through this argv before enabling Stop; a port moved into env or a config
# file would silently break that binding and every server would degrade to
# not-adoptable.
grep -Eq 'exec .*steerlab_server\.cli serve --port "\$PORT" --root "\$ROOT"' "$START" \
  || fail "serve exec line must pass --port \"\$PORT\" --root \"\$ROOT\" explicitly on the argv"

echo "PASS: start-local-server.sh — all 8 cases"
