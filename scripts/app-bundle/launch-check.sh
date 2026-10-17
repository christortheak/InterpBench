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
# Why an interrupted check still stops the app: a background job of a
# non-interactive script ignores SIGINT, so Ctrl-C during a build reached the
# script and not the app, and the script exited with the app still running.
# On exit, SIGINT, or SIGTERM the script now stops the one process it
# launched (SIGTERM, a bounded wait, then SIGKILL), by its PID and never by
# name, and an interrupted check ends by the same signal, so the build that
# called it stops too. Whether that process is running, has exited, or cannot
# be seen is three answers, not two: a check that cannot see the app (ps gave
# no answer) says so and never takes it for exited, and every wait is bounded.
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

# What PID $1 is doing: prints `running`, `exited`, or `unknown`.
#
# `kill -0`, a shell builtin that always runs, says whether the PID exists. One
# that does not is the launched process, exited and already reaped by this
# shell. One that does is asked about through ps, because an exited process
# not yet reaped (a zombie) still answers `kill -0`. A PID whose parent is no
# longer this script is not the process it launched (an exited one's PID,
# handed to someone else), so it counts as exited and is never signalled.
# When ps gives no answer the state is `unknown`, which never counts as
# exited: a check that cannot see the app must not conclude that it stopped.
process_state() {
  local fields ppid stat
  kill -0 "$1" 2>/dev/null || { echo exited; return; }
  fields="$(ps -o ppid=,stat= -p "$1" 2>/dev/null)" || { echo unknown; return; }
  read -r ppid stat <<<"$fields"
  case "${ppid:-}" in ''|*[!0-9]*) echo unknown; return ;; esac
  [ "$ppid" = "$$" ] || { echo exited; return; }
  case "${stat:-}" in
    '') echo unknown ;;
    *Z*) echo exited ;;
    *) echo running ;;
  esac
}

# True once PID $1 has exited, checking for at most $2 seconds. A process
# whose state is unknown has not exited.
exits_within() {
  local ticks=$(( $2 * 10 ))
  while [ "$(process_state "$1")" != exited ]; do
    [ "$ticks" -gt 0 ] || return 1
    sleep 0.1
    ticks=$(( ticks - 1 ))
  done
}

# Lists (on stderr) every process still running from the checked bundle.
# Returns 0 when there is one, 1 when there is none, and 2 when ps could not
# list processes, so that nothing can be said either way. This script and its
# own forks are left out: a fork carries this script's command line, which
# names the executable, until it execs. Every fork made here is a child of
# $$: the one that runs ps, and those of the pipeline that reads its table.
bundle_survivors() {
  local table
  table="$(ps -axo pid=,ppid=,command= 2>/dev/null)" || return 2
  [ -n "$table" ] || return 2
  printf '%s\n' "$table" \
    | LAUNCH_CHECK_MARK="$BUNDLE_MARK" awk -v self="$$" '
        $1 != self && $2 != self && index($0, ENVIRON["LAUNCH_CHECK_MARK"]) {
          if (!found) print "    -> still running from the checked bundle:"
          command = $0; sub(/^ *[0-9]+ +[0-9]+ +/, "", command)
          print "      pid " $1 ": " command; found = 1
        }
        END { exit found ? 0 : 1 }' >&2
}

# The process this invocation launched (the app itself: see `exec` below),
# and the timer of its observation window. Set once each is started.
LAUNCH_PID=""
TIMER_PID=""
LAUNCH_STARTED=""

# Stops the launched app, and nothing else: SIGTERM, up to `--grace` seconds
# to exit, then SIGKILL and up to 5 more. Returns 0 once it has exited (at
# once for one that already had, or for no launch at all) and 1 when that
# cannot be confirmed. Only ever signals the PID it launched, never a name.
stop_launched() {
  if [ -n "$TIMER_PID" ]; then
    kill "$TIMER_PID" 2>/dev/null
    TIMER_PID=""
  fi
  if [ -z "$LAUNCH_PID" ] && [ -n "$LAUNCH_STARTED" ]; then
    # Interrupted between the launch and the line that records its PID.
    set +u; LAUNCH_PID="$!"; set -u
  fi
  [ -n "$LAUNCH_PID" ] || return 0
  if [ "$(process_state "$LAUNCH_PID")" != exited ]; then
    kill -TERM "$LAUNCH_PID" 2>/dev/null
    if ! exits_within "$LAUNCH_PID" "$GRACE_SECONDS"; then
      echo "    -> still running ${GRACE_SECONDS}s after SIGTERM; sending SIGKILL" >&2
      kill -KILL "$LAUNCH_PID" 2>/dev/null
      exits_within "$LAUNCH_PID" 5 || return 1
    fi
  fi
  # It has exited, so this collects its status at once and never blocks.
  wait "$LAUNCH_PID" 2>/dev/null
  LAUNCH_PID=""
}

# What to say when the launched app cannot be confirmed stopped.
not_stopped() {
  echo "launch-check: could not confirm that the app (pid $1) stopped. Check with 'ps -p $1' and stop it ('kill -9 $1') before installing this build." >&2
}

# On any exit: the app must not outlive the check. A check that would
# otherwise pass but cannot confirm the stop fails.
on_exit() {
  local code=$? pid="$LAUNCH_PID"
  if ! stop_launched; then
    not_stopped "$pid"
    [ "$code" -ne 0 ] || code=6
  fi
  exit "$code"
}

# On SIGINT or SIGTERM: stop the app, then end by the same signal, so the
# caller (a build, a terminal) sees an interrupted check and stops too. The
# exit status is the fallback for a shell that started with the signal
# ignored and so cannot take it now.
on_signal() {  # signal name, exit status
  local pid="$LAUNCH_PID"
  trap - EXIT
  echo "launch-check: interrupted by SIG$1; stopping the app this check launched" >&2
  stop_launched || not_stopped "${pid:-unknown}"
  trap - "$1"
  kill -s "$1" "$$"
  exit "$2"
}
trap on_exit EXIT
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM

ENV_ARGS=(-u DYLD_FRAMEWORK_PATH -u DYLD_LIBRARY_PATH -u DYLD_INSERT_LIBRARIES STEERLAB_LAUNCH_CHECK=1)
if [ -n "$WORKSPACE" ]; then
  mkdir -p "$WORKSPACE" || fail "could not create the scratch workspace $WORKSPACE"
  ENV_ARGS+=(STEERLAB_WORKSPACE="$WORKSPACE")
fi

echo "  launching with no DYLD_* environment, offline (STEERLAB_LAUNCH_CHECK=1)…"
# `exec`, so the background job IS the app and $! is its PID. Without it $!
# names the subshell, and killing that leaves the app running.
LAUNCH_STARTED=1
( cd / && exec env "${ENV_ARGS[@]}" "$EXECUTABLE" >"$LOG" 2>&1 ) &
LAUNCH_PID=$!
# The window is waited for with `wait`, which returns as soon as a signal
# arrives, rather than a foreground sleep, which would hold off the trap
# until it finished.
sleep "$SECONDS_TO_WAIT" &
TIMER_PID=$!
wait "$TIMER_PID" 2>/dev/null
TIMER_PID=""
STATE="$(process_state "$LAUNCH_PID")"
case "$STATE" in
  running)
    echo "    -> alive past initialization ✓"
    if ! stop_launched; then
      not_stopped "$LAUNCH_PID"
      LAUNCH_PID=""   # SIGTERM and SIGKILL were both sent; there is nothing left to try
      exit 6
    fi
    bundle_survivors
    case $? in
      0) fail "the launch check left processes running from $BUNDLE_MARK (listed above) — stop them (kill <pid>) before installing this build" ;;
      2) fail "could not list running processes (ps gave no answer), so the check cannot confirm that nothing from $BUNDLE_MARK is still running — look with 'ps -ax' and stop anything from it before installing this build" ;;
    esac
    echo "    -> stopped; nothing left running from the checked bundle ✓"
    ;;
  exited)
    stop_launched
    echo "    -> EXITED EARLY. Output:" >&2
    sed 's/^/      /' "$LOG" >&2
    fail "the assembled app did not stay up"
    ;;
  *)
    fail "could not tell whether the app (pid $LAUNCH_PID) stayed up, because ps gave no answer about it; the check stops it now. Make sure /bin/ps runs for this account, then build again"
    ;;
esac

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
