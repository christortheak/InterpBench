#!/usr/bin/env bash
# The post-assembly launch check `scripts/build-app.sh` runs on a freshly
# built SteerLab.app — OFFLINE.
#
#   usage: launch-check.sh <app-executable> <log-file> [--seconds N]
#                          [--grace N] [--workspace DIR]
#
# Launches the app the way a user will (no DYLD_* environment, cwd /), waits
# `--seconds` (default 10) for initialisation to clear, stops it (SIGTERM,
# then SIGKILL after `--grace` seconds, default 5), and reads what it wrote.
# The app is told it is a launch check through
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
# Why the stop is verified (a build on 2026-10-12): the app used to run in a
# subshell, and the script killed the subshell's PID. The app was reparented
# to launchd and kept running, offline and against a scratch workspace the
# build had already deleted, until someone killed it by hand; a guard against
# installing over a running app found it and refused the install. The
# subshell now execs the app, so the PID the script holds is the app's, and
# the check confirms that nothing from the checked bundle outlives it.
#
# The check PASSES only when all of these hold:
#   1. the process is still alive when the window closes;
#   2. once stopped, nothing is left running from the checked bundle (the
#      `.app` the executable sits in, or the executable itself outside one);
#   3. the log carries the armed line   `launch-check: offline mode armed`;
#   4. the log carries the verdict      `launch-check: offline mode, no network activity`;
#   5. the log carries NO violation     `launch-check: offline mode VIOLATED`.
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
GRACE_SECONDS=5
WORKSPACE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --seconds)   SECONDS_TO_WAIT="$2"; shift 2 ;;
    --grace)     GRACE_SECONDS="$2"; shift 2 ;;
    --workspace) WORKSPACE="$2"; shift 2 ;;
    -h|--help)   usage; exit 0 ;;
    *) echo "launch-check.sh: unknown flag: $1" >&2; usage >&2; exit 2 ;;
  esac
done
[ -x "$EXECUTABLE" ] || fail "not an executable: $EXECUTABLE"
case "$SECONDS_TO_WAIT" in ''|*[!0-9]*) echo "launch-check.sh: --seconds takes an integer" >&2; exit 2 ;; esac
case "$GRACE_SECONDS" in ''|*[!0-9]*) echo "launch-check.sh: --grace takes an integer" >&2; exit 2 ;; esac

# What "from the checked bundle" means once the app is stopped: anything
# running out of the `.app` the executable sits in (…/X.app/Contents/MacOS/X —
# the main executable and every helper), or the executable itself when it is
# not inside one.
case "$EXECUTABLE" in
  */Contents/MacOS/*) BUNDLE_MARK="${EXECUTABLE%/Contents/MacOS/*}/" ;;
  *)                  BUNDLE_MARK="$EXECUTABLE" ;;
esac

# True while PID $1 is running. An exited process nobody has reaped yet (a
# zombie) still answers `kill -0`, so ask ps for its state instead.
alive() {
  local state
  state="$(ps -o stat= -p "$1" 2>/dev/null)" || return 1
  case "$state" in ''|*Z*) return 1 ;; esac
}

# True once PID $1 has exited, waiting at most $2 seconds.
exits_within() {
  local ticks=$(( $2 * 10 ))
  while alive "$1"; do
    [ "$ticks" -gt 0 ] || return 1
    sleep 0.1
    ticks=$(( ticks - 1 ))
  done
}

# Lists (on stderr) every process still running from the checked bundle;
# true when there is one. This script and its own forks are left out: a fork
# carries this script's command line, which names the executable, until it
# execs. Run it directly, never inside $(…), so every fork is a child of $$.
bundle_survivors() {
  ps -axo pid=,ppid=,command= \
    | LAUNCH_CHECK_MARK="$BUNDLE_MARK" awk -v self="$$" '
        $1 != self && $2 != self && index($0, ENVIRON["LAUNCH_CHECK_MARK"]) {
          if (!found) print "    -> still running from the checked bundle:"
          command = $0; sub(/^ *[0-9]+ +[0-9]+ +/, "", command)
          print "      pid " $1 ": " command; found = 1
        }
        END { exit found ? 0 : 1 }' >&2
}

ENV_ARGS=(-u DYLD_FRAMEWORK_PATH -u DYLD_LIBRARY_PATH -u DYLD_INSERT_LIBRARIES STEERLAB_LAUNCH_CHECK=1)
if [ -n "$WORKSPACE" ]; then
  mkdir -p "$WORKSPACE" || fail "could not create the scratch workspace $WORKSPACE"
  ENV_ARGS+=(STEERLAB_WORKSPACE="$WORKSPACE")
fi

echo "  launching with no DYLD_* environment, offline (STEERLAB_LAUNCH_CHECK=1)…"
# `exec`, so the background job IS the app and $! is its PID. Without it $!
# names the subshell, and killing that leaves the app running.
( cd / && exec env "${ENV_ARGS[@]}" "$EXECUTABLE" >"$LOG" 2>&1 ) &
LAUNCH_PID=$!
sleep "$SECONDS_TO_WAIT"
if alive "$LAUNCH_PID"; then
  echo "    -> alive past initialization ✓"
  kill "$LAUNCH_PID" 2>/dev/null
  if ! exits_within "$LAUNCH_PID" "$GRACE_SECONDS"; then
    echo "    -> still running ${GRACE_SECONDS}s after SIGTERM; sending SIGKILL" >&2
    kill -9 "$LAUNCH_PID" 2>/dev/null
    exits_within "$LAUNCH_PID" 5 \
      || fail "the app (pid $LAUNCH_PID) is still running after SIGKILL — stop it before installing this build"
  fi
  wait "$LAUNCH_PID" 2>/dev/null
  if bundle_survivors; then
    fail "the launch check left processes running from $BUNDLE_MARK (listed above) — stop them (kill <pid>) before installing this build"
  fi
  echo "    -> stopped; nothing left running from the checked bundle ✓"
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
