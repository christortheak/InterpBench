#!/bin/zsh
# The named gate for WP0-AGENT-SURFACE-AUDIT §7 step 12: does the installed
# CLI work from a clean shell?
#
#   usage: install-cli-test.sh [--from-products <dir>] [--keep]
#
# Everything below runs the INSTALLED binary from an unrelated working
# directory with a SCRUBBED environment — `env -i`, so no
# `DYLD_FRAMEWORK_PATH`, no `STEERLAB_*`, nothing inherited from the checkout's
# developer setup. That is the condition the whole step exists to prove, and a
# test that inherited the developer environment would prove nothing.
#
# The three assertions §7 row 12 names, plus the two the install turns on:
#
#   1. `cluster sites list --json` parses as JSON and exits 0.
#   2. `data check` on a scratch workspace exits 65 (the readiness refusal).
#   3. A GPU verb completes — the §6.1 claim that a COLOCATED `mlx.metallib`
#      suffices with no `DYLD_FRAMEWORK_PATH`. Two tiers: `install verify
#      --gpu` dispatches one Metal kernel and needs no weights, so it always
#      runs; the full smoke-test config runs too when its 4B model is already
#      in the Hugging Face cache. NOTHING here downloads a model.
#   4. No absolute build-machine path survives as an `LC_RPATH`.
#   5. `--json` puts exactly one document on stdout and nothing else.
#
# Every case runs under a SCRATCH HOME except the smoke test, which needs the
# real Hugging Face cache — see the note at `SCRATCH_HOME` for why that is
# about the macOS keychain and not about tidiness.
#
# Installs into a TEMPORARY prefix, so running the gate never disturbs a real
# `~/.local` install. Needs a built products directory — it does not build.
#
# Exit codes: 0 all cases passed · 1 a case failed (the failing sentence is
#             printed) · 2 usage, or no build products to install from
set -eu

SCRIPT_DIR="${0:A:h}"
PROJECT_ROOT="${SCRIPT_DIR:h:h}"
INSTALLER="$SCRIPT_DIR/../install-cli.sh"
PRODUCTS="$PROJECT_ROOT/.deriveddata.nosync/Build/Products/Debug"
KEEP=0

while (( $# > 0 )); do
  case "$1" in
    --from-products)
      [[ $# -ge 2 ]] || { echo "--from-products requires a directory path"; exit 2 }
      PRODUCTS="${2:A}"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    *) echo "usage: install-cli-test.sh [--from-products <dir>] [--keep]"; exit 2 ;;
  esac
done

if [[ ! -x "$PRODUCTS/steerlab-cli" ]]; then
  echo "no built steerlab-cli at $PRODUCTS/steerlab-cli."
  echo "Build it first: xcodebuild build -skipMacroValidation -scheme steerlab-cli \\"
  echo "  -destination 'platform=macOS' CLANG_COVERAGE_MAPPING=NO \\"
  echo "  -derivedDataPath .deriveddata.nosync"
  exit 2
fi

WORK="$(mktemp -d)"
cleanup() { (( KEEP )) || rm -rf "$WORK" }
trap cleanup EXIT

PREFIX="$WORK/prefix"
WORKSPACE="$WORK/workspace"
# An unrelated working directory: not the checkout, not the install, not the
# workspace. Writable, because a binary built before CLANG_COVERAGE_MAPPING=NO
# landed (reachable via --from-products) carries coverage instrumentation that
# writes `default.profraw` into cwd, and a read-only cwd turns that into noise
# that reads like a failure.
UNRELATED="$WORK/elsewhere"
# A scratch HOME for every case that does not need the model cache. Two
# reasons, and the second one is a step-12 finding:
#
#   * hermetic — the gate must not read (or depend on) whatever cluster sites
#     and workspaces this machine happens to have saved;
#   * KEYCHAIN ACLs ARE PER-BINARY-IDENTITY. `cluster sites list` reads each
#     saved site's token out of the login keychain, and the installed binary at
#     <prefix>/libexec/steerlab is a DIFFERENT identity from the build product
#     the ACL was granted to — so the first keychain-touching invocation after
#     an install puts up a macOS password prompt and waits for a human. That is
#     correct macOS behaviour, not a defect, and it must not be suppressed or
#     worked around programmatically. An automated gate simply must not be the
#     thing that trips it: with no login keychain under HOME there is nothing
#     to prompt for, and the verb answers `tokenAvailable: false`.
SCRATCH_HOME="$WORK/home"
mkdir -p "$UNRELATED" "$SCRATCH_HOME"

CASES=0
fail() { echo "FAIL: $1"; exit 1 }
pass() { CASES=$((CASES + 1)); echo "  ok — $1" }

# The scrubbed invocation. `env -i` clears EVERYTHING — no
# DYLD_FRAMEWORK_PATH, no STEERLAB_*, nothing from the developer shell. Only
# PATH and TMPDIR are put back (neither is developer state), plus a HOME the
# caller chooses.
steerlab() {
  local home="$SCRATCH_HOME"
  if [[ "$1" == "--real-home" ]]; then home="$HOME"; shift; fi
  (cd "$UNRELATED" && env -i \
    HOME="$home" PATH="/usr/bin:/bin:/usr/sbin:/sbin" \
    TMPDIR="${TMPDIR:-/tmp}" "$PREFIX/bin/steerlab-cli" "$@")
}

# Runs the shim, capturing stdout and the exit code without tripping `set -e`.
run() {
  set +e
  OUT="$(steerlab "$@" 2>"$WORK/stderr")"
  STATUS=$?
  set -e
}

echo "install-cli-test.sh — installing into $PREFIX"
zsh "$INSTALLER" --prefix "$PREFIX" --from-products "$PRODUCTS" >"$WORK/install.log" 2>&1 \
  || { cat "$WORK/install.log"; fail "the installer exited nonzero" }

# ── 0. The layout is what §6.2 specifies ──────────────────────────────────
[[ -x "$PREFIX/bin/steerlab-cli" ]] || fail "no shim at $PREFIX/bin/steerlab-cli"
[[ -x "$PREFIX/libexec/steerlab/steerlab-cli" ]] \
  || fail "no binary at $PREFIX/libexec/steerlab/steerlab-cli"
[[ -f "$PREFIX/libexec/steerlab/mlx.metallib" ]] \
  || fail "no colocated mlx.metallib — GPU verbs would need DYLD_FRAMEWORK_PATH"
[[ -f "$PREFIX/libexec/steerlab/resource-manifest.json" ]] \
  || fail "the installed tree was never stamped"
# A shim, not a symlink: MLX resolves the shader library against the REAL
# binary's directory, so a symlink in bin/ would look in bin/.
[[ ! -L "$PREFIX/bin/steerlab-cli" ]] || fail "the shim is a symlink, not a shim"
# The short name `steerlab` belongs to the Python client; whether the installer
# writes it as an alias depends on what THIS machine has (venv console script,
# PATH), so the gate asserts only its shape: if present, it must be the
# installer's own shim, never something foreign it clobbered into place.
if [[ -e "$PREFIX/bin/steerlab" ]]; then
  grep -qs "Generated by scripts/install-cli.sh" "$PREFIX/bin/steerlab" \
    || fail "$PREFIX/bin/steerlab exists but is not this installer's shim"
fi
pass "the installed layout matches audit §6.2"

# ── 4. No absolute build-machine rpath survived ───────────────────────────
# Checked before anything else runs: this is the leak the release scanner
# cannot see, because it matches `/Users/<name>` in TEXT and never inside a
# Mach-O.
survivors="$(otool -l "$PREFIX/libexec/steerlab/steerlab-cli" | awk '
  /LC_RPATH/ { in_rpath = 1; next }
  in_rpath && /^ *path / {
    sub(/^ *path /, ""); sub(/ \(offset [0-9]+\)$/, "")
    if (substr($0, 1, 1) == "/") print
    in_rpath = 0
  }')"
[[ -z "$survivors" ]] || fail "an absolute LC_RPATH survived the install: $survivors"
# Positive control: the relative rpaths the linker needs are still there, so
# the assertion above cannot pass by having deleted everything.
otool -l "$PREFIX/libexec/steerlab/steerlab-cli" | grep -q "@executable_path" \
  || fail "the relative rpaths were deleted too — the strip was too wide"
pass "no absolute build-machine LC_RPATH survives; the relative ones do"

# ── The version report names the install, not the checkout ────────────────
run --version
(( STATUS == 0 )) || fail "steerlab-cli --version exited $STATUS"
[[ "$OUT" == *"$PREFIX/libexec/steerlab"* ]] \
  || fail "--version does not report the install root: $OUT"
[[ "$OUT" == *"mlx.metallib colocated"* ]] \
  || fail "--version does not report the colocated shader library: $OUT"
pass "steerlab-cli --version reports the installed layout"

# ── 5. --json puts exactly one document on stdout ─────────────────────────
run --version --json
(( STATUS == 0 )) || fail "steerlab-cli --version --json exited $STATUS"
echo "$OUT" | /usr/bin/python3 -c 'import json,sys; json.load(sys.stdin)' \
  || fail "--version --json did not produce one parseable document"
pass "--json produces exactly one JSON document on stdout"

# ── 1. `cluster sites list --json` parses ─────────────────────────────────
run cluster sites list --json
(( STATUS == 0 )) || fail "cluster sites list --json exited $STATUS (expected 0)"
echo "$OUT" | /usr/bin/python3 -c 'import json,sys; json.load(sys.stdin)' \
  || fail "cluster sites list --json did not produce parseable JSON"
pass "cluster sites list --json parses and exits 0"

# ── 2. `data check` on a scratch workspace exits 65 ───────────────────────
run workspace init "$WORKSPACE"
(( STATUS == 0 )) || { cat "$WORK/stderr"; fail "workspace init exited $STATUS" }

run --workspace "$WORKSPACE" experiment create install-gate \
  --model mlx-community/gemma-3-4b-it-4bit
(( STATUS == 0 )) || { cat "$WORK/stderr"; fail "experiment create exited $STATUS" }

run --workspace "$WORKSPACE" data check install-gate
# 65 is the readiness refusal — the audit's one deliberate human-mode exit-code
# migration (step 7, 2 → 65). A fresh manifest has blockers by construction.
(( STATUS == 65 )) || fail "data check exited $STATUS (expected 65)"
pass "data check on a scratch workspace refuses with 65"

# ── 3. A GPU verb completes with no DYLD_FRAMEWORK_PATH ───────────────────
# Tier 1 — always runs, no weights: one Metal kernel dispatched from the
# installed layout. This is the §6.1 claim itself.
run install verify --gpu
(( STATUS == 0 )) || { cat "$WORK/stderr"; fail "install verify --gpu exited $STATUS" }
[[ "$OUT" == *"Metal shaders loaded"* ]] \
  || fail "install verify --gpu did not report loaded shaders: $OUT"
pass "a Metal kernel runs from the install with no DYLD_FRAMEWORK_PATH"

# Tier 2 — the full smoke test, ONLY when its model is already cached. This
# test never downloads weights.
SMOKE_CONFIG="$PROJECT_ROOT/prompts/configs/smoke-test-gemma-only.json"
CACHED_MODEL="$HOME/.cache/huggingface/hub/models--mlx-community--gemma-3-4b-it-4bit"
if [[ -d "$CACHED_MODEL" && -f "$SMOKE_CONFIG" ]]; then
  # The one case that needs the real HOME: the weights live in its
  # Hugging Face cache. Still `env -i`, still an unrelated cwd.
  run --real-home --config "$SMOKE_CONFIG"
  (( STATUS == 0 )) || { cat "$WORK/stderr"; fail "the smoke test exited $STATUS" }
  [[ "$OUT" == *"SMOKE TEST PASSED"* ]] || fail "the smoke test did not pass: $OUT"
  pass "the GPU smoke test completes from the install (hooks fire, steered ≠ baseline)"
else
  echo "  SKIP — the 4B smoke-test model is not in the Hugging Face cache, so"
  echo "         only the weightless Metal-kernel tier of case 3 was proven."
  echo "         Cache mlx-community/gemma-3-4b-it-4bit to run the full tier."
fi

# ── Re-runnable: a second install upgrades in place ───────────────────────
zsh "$INSTALLER" --prefix "$PREFIX" --from-products "$PRODUCTS" \
  >"$WORK/reinstall.log" 2>&1 \
  || { cat "$WORK/reinstall.log"; fail "the installer is not re-runnable" }
run install verify
(( STATUS == 0 )) || fail "the reinstalled tree does not verify (exit $STATUS)"
pass "the installer is re-runnable and the tree still verifies"

# ── A tampered file is caught ─────────────────────────────────────────────
# The positive control for the manifest: without this, `install verify` passing
# would only mean it read a file.
printf 'tampered' >> "$PREFIX/libexec/steerlab/mlx.metallib"
# Asserted in `--json`, where the envelope's state vocabulary is authoritative
# and a refusal is 65. Human mode still exits 1 for this refusal: the audit's
# step 7 migrated exactly ONE human exit code (`data check`), and widening that
# would break `set -e` wrappers on a change nobody reviewed.
run install verify --json
(( STATUS == 65 )) || fail "a tampered install verified anyway (exit $STATUS)"
echo "$OUT" | /usr/bin/python3 -c '
import json, sys
doc = json.load(sys.stdin)
assert doc["state"] == "refused", doc["state"]
assert doc["error"]["code"] == "installIntegrity", doc["error"]["code"]
assert any(p["kind"] == "mismatch" for p in doc["result"]["problems"]), doc["result"]
' || fail "the tamper refusal envelope is not the documented shape"
pass "install verify refuses a tampered tree — refused/65/installIntegrity"

echo "PASS: install-cli-test.sh — $CASES cases"
