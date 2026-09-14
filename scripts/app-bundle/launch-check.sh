#!/usr/bin/env bash
# The post-assembly launch check `scripts/build-app.sh` runs on a freshly
# built SteerLab.app — OFFLINE.
#
#   usage: launch-check.sh <app-executable> <log-file> [--seconds N]
#                          [--workspace DIR]
#
# Launches the app the way a user will (no DYLD_* environment, cwd /), waits
# `--seconds` (default 10) for initialisation to clear, kills it, and reads
# what it wrote. The app is told it is a launch check through
# `STEERLAB_LAUNCH_CHECK=1`; under that switch it performs NO network activity
# — no site connection, no SSH tunnel, no evidence auto-import, no remote
# polling, no update check — and prints one verdict line when its observation
# window closes. `--workspace DIR` pins `STEERLAB_WORKSPACE` there so the
# check never opens (or writes the upkeep line into) the researcher's own
# workspace; the caller decides what to put in it.
#
# Why this exists (a live controller on 2026-09-13): the check used to be a
# full launch against the researcher's saved sites and Keychain. Through a
# live tunnel the app connected to a controller, its evidence auto-import
# began fetching every succeeded job's bundle the local ledger had never
# seen, the script killed the app eight seconds in, and the controller
# stopped answering for half an hour.
#
# The check PASSES only when all of these hold:
#   1. the process is still alive when the window closes;
#   2. the log carries the armed line   `launch-check: offline mode armed`;
#   3. the log carries the verdict      `launch-check: offline mode, no network activity`;
#   4. the log carries NO violation     `launch-check: offline mode VIOLATED`.
#
# A violation names the refused attempts, so a future code path that reaches
# for the network at launch fails the build here instead of wedging a
# controller. The log is printed either way — the resource self-check line
# (`CodeResources: … no problems`) still lives there.
#
# Exit codes: 0 passed · 2 usage · 6 the check failed (the reason is printed)
set -u

ARMED_LINE="launch-check: offline mode armed"
OFFLINE_LINE="launch-check: offline mode, no network activity"
VIOLATION_PREFIX="launch-check: offline mode VIOLATED"

usage() { sed -n '/^#   usage:/,/^# Exit codes/p' "$0" | sed 's/^# \{0,3\}//'; }
fail() { echo "launch-check: $1" >&2; exit 6; }

[ $# -ge 2 ] || { usage >&2; exit 2; }
EXECUTABLE="$1"; LOG="$2"; shift 2
SECONDS_TO_WAIT=10
WORKSPACE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --seconds)   SECONDS_TO_WAIT="$2"; shift 2 ;;
    --workspace) WORKSPACE="$2"; shift 2 ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "launch-check.sh: unknown flag: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[ -x "$EXECUTABLE" ] || fail "not an executable: $EXECUTABLE"
case "$SECONDS_TO_WAIT" in ''|*[!0-9]*) echo "launch-check.sh: --seconds takes an integer" >&2; exit 2 ;; esac

ENV_ARGS=(-u DYLD_FRAMEWORK_PATH -u DYLD_LIBRARY_PATH -u DYLD_INSERT_LIBRARIES STEERLAB_LAUNCH_CHECK=1)
if [ -n "$WORKSPACE" ]; then
  mkdir -p "$WORKSPACE" || fail "could not create the scratch workspace $WORKSPACE"
  ENV_ARGS+=(STEERLAB_WORKSPACE="$WORKSPACE")
fi

echo "  launching with no DYLD_* environment, offline (STEERLAB_LAUNCH_CHECK=1)…"
( cd / && env "${ENV_ARGS[@]}" "$EXECUTABLE" >"$LOG" 2>&1 ) &
LAUNCH_PID=$!
sleep "$SECONDS_TO_WAIT"
if kill -0 "$LAUNCH_PID" 2>/dev/null; then
  echo "    -> alive past initialization ✓"
  kill "$LAUNCH_PID" 2>/dev/null
  wait "$LAUNCH_PID" 2>/dev/null
else
  wait "$LAUNCH_PID" 2>/dev/null
  echo "    -> EXITED EARLY. Output:" >&2
  sed 's/^/      /' "$LOG" >&2
  fail "the assembled app did not stay up"
fi

if [ -s "$LOG" ]; then
  echo "    launch output ($LOG):"
  sed 's/^/      /' "$LOG"
else
  echo "    no output on stdout/stderr"
fi

if grep -qF "$VIOLATION_PREFIX" "$LOG"; then
  grep -F "$VIOLATION_PREFIX" "$LOG" | sed 's/^/    /' >&2
  fail "the app reached for the network during the launch check — every refused attempt is named above; the build is not shippable until that launch path is gone"
fi
grep -qF "$ARMED_LINE" "$LOG" \
  || fail "the app did not honour STEERLAB_LAUNCH_CHECK=1 (no '$ARMED_LINE' line) — an app that ignores the switch would launch for real against the researcher's sites"
grep -qF "$OFFLINE_LINE" "$LOG" \
  || fail "no offline verdict ('$OFFLINE_LINE') within ${SECONDS_TO_WAIT}s — the app never closed its observation window; raise --seconds only if initialization is genuinely that slow"
echo "    -> offline: no network activity ✓"
exit 0
