#!/bin/zsh
# Self-test for scripts/app-bundle/launch-check.sh — the offline launch check
# `build-app.sh` runs on a freshly assembled SteerLab.app (a live controller
# on 2026-09-13: the old check launched the app for real and its evidence
# auto-import began pulling from a controller behind a live tunnel).
#
# Never launches a GUI app. Every case runs the checker against a FAKE app
# executable (a shell script) that behaves one way, and asserts the checker's
# verdict and exit code:
#
#   1. PASS: the fake honours the switch (armed line + offline verdict, stays
#      up) → exit 0; and the environment the fake saw carried
#      STEERLAB_LAUNCH_CHECK=1, STEERLAB_WORKSPACE=<the scratch dir>, and no
#      DYLD_* variable.
#   2. FAIL on a violation verdict (the app names refused network attempts)
#      → exit 6, the violation line echoed.
#   3. FAIL when the app ignores the switch (no armed line) → exit 6.
#   4. FAIL when the app arms but never delivers a verdict → exit 6.
#   5. FAIL when the app exits early → exit 6.
#   6. Usage: missing arguments → exit 2.
#
# Exit codes: 0 all cases passed · 1 a case failed (the failing sentence is
#             printed)
set -eu

SCRIPT_DIR="${0:A:h}"
CHECK="$SCRIPT_DIR/../app-bundle/launch-check.sh"
WORK="$(mktemp -d)"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

fail() { echo "FAIL: $1"; exit 1 }
pass() { echo "ok: $1" }

ARMED="launch-check: offline mode armed (STEERLAB_LAUNCH_CHECK) — everything disabled"
OFFLINE="launch-check: offline mode, no network activity"
VIOLATED="launch-check: offline mode VIOLATED: 2 network attempt(s) refused — cluster connect; GET http://127.0.0.1:8718/api/jobs"

# Write a fake app executable that prints the given lines (to stderr, like
# the app) and then sleeps unless told to exit. It also dumps its environment
# so the PASS case can assert what the checker handed it.
make_fake() {  # name, behaviour
  # Not `path`: that spelling is zsh's array view of $PATH.
  local fake_path="$WORK/$1"
  {
    echo '#!/bin/zsh'
    echo "env > '$WORK/$1.env'"
    case "$2" in
      pass)      echo "echo '$ARMED' >&2; echo 'CodeResources: release mode — 6/6 resource families resolved; no problems' >&2; echo '$OFFLINE' >&2; sleep 30" ;;
      violation) echo "echo '$ARMED' >&2; echo '$VIOLATED' >&2; sleep 30" ;;
      ignores)   echo "echo 'CodeResources: release mode — 6/6 resource families resolved; no problems' >&2; sleep 30" ;;
      noverdict) echo "echo '$ARMED' >&2; sleep 30" ;;
      exits)     echo "echo 'dyld: Library not loaded' >&2; exit 1" ;;
    esac
  } > "$fake_path"
  chmod +x "$fake_path"
  echo "$fake_path"
}

run() {  # fake, extra args...
  local fake="$1"; shift
  set +e
  OUTPUT="$(bash "$CHECK" "$fake" "$WORK/log.txt" --seconds 1 "$@" 2>&1)"
  STATUS=$?
  set -e
}

# 1. PASS ---------------------------------------------------------------------
FAKE="$(make_fake honest pass)"
export DYLD_FRAMEWORK_PATH=/nonexistent/frameworks
run "$FAKE" --workspace "$WORK/scratch-workspace"
unset DYLD_FRAMEWORK_PATH
[[ $STATUS -eq 0 ]] || fail "an honest app must pass (exit $STATUS): $OUTPUT"
[[ "$OUTPUT" == *"offline: no network activity ✓"* ]] || fail "pass verdict not printed: $OUTPUT"
[[ "$OUTPUT" == *"no problems"* ]] || fail "the app log (resource self-check) must be echoed: $OUTPUT"
grep -q '^STEERLAB_LAUNCH_CHECK=1$' "$WORK/honest.env" || fail "the app was not told it is a launch check"
grep -q "^STEERLAB_WORKSPACE=$WORK/scratch-workspace\$" "$WORK/honest.env" || fail "STEERLAB_WORKSPACE was not pinned to the scratch workspace"
[[ -d "$WORK/scratch-workspace" ]] || fail "the scratch workspace directory was not created"
if grep -q '^DYLD_' "$WORK/honest.env"; then fail "a DYLD_* variable leaked into the launch"; fi
pass "honest app passes; env carries the switch and the scratch workspace, no DYLD_*"

# 2. Violation ------------------------------------------------------------------
run "$(make_fake leaky violation)"
[[ $STATUS -eq 6 ]] || fail "a violation verdict must fail with 6 (got $STATUS)"
[[ "$OUTPUT" == *"cluster connect; GET http://127.0.0.1:8718/api/jobs"* ]] || fail "the refused attempts must be echoed: $OUTPUT"
[[ "$OUTPUT" == *"reached for the network"* ]] || fail "the failure sentence must name the cause: $OUTPUT"
pass "violation verdict fails the check and names the attempts"

# 3. Ignores the switch --------------------------------------------------------
run "$(make_fake deaf ignores)"
[[ $STATUS -eq 6 ]] || fail "an app that ignores the switch must fail with 6 (got $STATUS)"
[[ "$OUTPUT" == *"did not honour STEERLAB_LAUNCH_CHECK=1"* ]] || fail "missing-armed-line sentence expected: $OUTPUT"
pass "an app that ignores the switch fails"

# 4. No verdict ----------------------------------------------------------------
run "$(make_fake mute noverdict)"
[[ $STATUS -eq 6 ]] || fail "no verdict must fail with 6 (got $STATUS)"
[[ "$OUTPUT" == *"no offline verdict"* ]] || fail "missing-verdict sentence expected: $OUTPUT"
pass "armed but no verdict fails"

# 5. Exits early ---------------------------------------------------------------
run "$(make_fake crash exits)"
[[ $STATUS -eq 6 ]] || fail "an early exit must fail with 6 (got $STATUS)"
[[ "$OUTPUT" == *"EXITED EARLY"* && "$OUTPUT" == *"dyld: Library not loaded"* ]] || fail "early-exit output expected: $OUTPUT"
pass "early exit fails with the app's output"

# 6. Usage ---------------------------------------------------------------------
set +e
bash "$CHECK" >/dev/null 2>&1; STATUS=$?
set -e
[[ $STATUS -eq 2 ]] || fail "missing arguments must exit 2 (got $STATUS)"
pass "usage exits 2"

echo "launch-check-test: all cases passed"
