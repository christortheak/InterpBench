#!/bin/zsh
# Self-test for scripts/check-binary-fresh.sh (the launcher stale-binary
# detector). Pure filesystem — no build tools required. Exits nonzero with a
# plain sentence on the first failing case.
set -eu

SCRIPT_DIR="${0:A:h}"
CHECK="$SCRIPT_DIR/../check-binary-fresh.sh"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fail() { echo "FAIL: $1"; exit 1 }

# Runs the checker, capturing its exit code in $STATUS without tripping set -e.
check() {
  set +e
  zsh "$CHECK" "$@" >/dev/null 2>&1
  STATUS=$?
  set -e
}

mkdir -p "$WORK/Sources/App"

# Case 1: missing binary → exit 2
check "$WORK/bin" "$WORK/Sources"
[[ $STATUS -eq 2 ]] || fail "missing binary should exit 2, got $STATUS"

# Case 2: usage error (no sources) → exit 2
check "$WORK/bin"
[[ $STATUS -eq 2 ]] || fail "missing source args should exit 2, got $STATUS"

# Case 3: fresh — binary newer than every source → exit 0
echo 'let x = 1' > "$WORK/Sources/App/Main.swift"
echo 'manifest' > "$WORK/Package.swift"
touch -t 202601010000 "$WORK/Sources/App/Main.swift" "$WORK/Package.swift"
echo 'binary' > "$WORK/bin"
touch -t 202606010000 "$WORK/bin"
check "$WORK/bin" "$WORK/Sources" "$WORK/Package.swift"
[[ $STATUS -eq 0 ]] || fail "up-to-date binary should exit 0, got $STATUS"

# Case 4: stale — a source file inside a directory is newer → exit 1
touch -t 202612310000 "$WORK/Sources/App/Main.swift"
check "$WORK/bin" "$WORK/Sources" "$WORK/Package.swift"
[[ $STATUS -eq 1 ]] || fail "newer source in directory should exit 1, got $STATUS"

# Case 5: stale — a directly named source FILE is newer → exit 1
touch -t 202601010000 "$WORK/Sources/App/Main.swift"
touch -t 202612310000 "$WORK/Package.swift"
check "$WORK/bin" "$WORK/Sources" "$WORK/Package.swift"
[[ $STATUS -eq 1 ]] || fail "newer named file should exit 1, got $STATUS"

# Case 6: non-source files (e.g. .log) never mark the build stale
touch -t 202601010000 "$WORK/Package.swift"
echo 'log' > "$WORK/Sources/App/debug.log"
touch -t 202612310000 "$WORK/Sources/App/debug.log"
check "$WORK/bin" "$WORK/Sources" "$WORK/Package.swift"
[[ $STATUS -eq 0 ]] || fail "non-source file should not mark the binary stale, got $STATUS"

# Case 7: nonexistent source paths are skipped, not errors
check "$WORK/bin" "$WORK/DoesNotExist" "$WORK/Sources"
[[ $STATUS -eq 0 ]] || fail "nonexistent source path should be skipped, got $STATUS"

echo "PASS: check-binary-fresh.sh — all 7 cases"
