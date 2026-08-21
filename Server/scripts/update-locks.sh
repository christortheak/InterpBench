#!/usr/bin/env bash
# Regenerate SteerLab's per-platform Python dependency locks (WP6 R1).
#
# `Server/pyproject.toml` declares FLOORS ("torch>=2.2"). Floors are the right
# thing for a library and the wrong thing for an instrument: two sites can
# satisfy them with different torch/transformers and produce different numbers
# on the substrate where the reproducibility claims actually live. These locks
# are the exact-version half of that contract; each run's `config.json` then
# stamps what was ACTUALLY installed (`pythonEnvironment`), so a reader never
# has to trust that the lock was honored.
#
# Two explicit platform locks, because the resolution genuinely differs:
#   * requirements-macos-arm64.lock   — Mac parity/dev work (MPS)
#   * requirements-linux-x86_64.lock  — the CUDA science substrate
#
# Usage:  Server/scripts/update-locks.sh [--generate-hashes]
# Requires `uv` (a dev-only tool; `Server[dev]` installs it, or
# `pip install uv` into the server venv). Needs PyPI egress.
#
# Review the diff before committing: a lock bump is a substrate change, and on
# a frozen study it is exactly the kind of drift the run stamp exists to make
# visible.

set -euo pipefail

SERVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SERVER_DIR"

UV="${UV:-}"
if [ -z "$UV" ]; then
  if [ -x ".venv.nosync/bin/uv" ]; then UV=".venv.nosync/bin/uv"
  elif command -v uv >/dev/null 2>&1; then UV="uv"
  else
    echo "update-locks.sh: uv not found. Install it into the server venv:" >&2
    echo "  Server/.venv.nosync/bin/pip install uv     (or: pip install -e 'Server[dev]')" >&2
    exit 1
  fi
fi

EXTRA_FLAGS=()
if [ "${1:-}" = "--generate-hashes" ]; then
  EXTRA_FLAGS+=(--generate-hashes)
  shift
fi

# `--extra all` (lora + gemmascope + test), matching what `bootstrap.sh`
# installs on a cluster node. `jlens` is deliberately NOT here, for the same
# two reasons pyproject.toml keeps it out of `all`: it floors
# transformers>=5.5, and it installs from a git URL, so a lock-based install
# would require github.com egress on every node. (Verified 2026-08-18: the two
# groups DO co-resolve — adding `--extra jlens` changes no version in either
# lock, it only appends the git pin line — so the exclusion is a policy choice,
# not a conflict. jlens's own identity is already a commit sha in pyproject.)
EXTRAS=(--extra all)

# Python 3.12 is the version bootstrap.sh creates (`--python 3.12`) and the
# version the dev venv runs.
PYVER=3.12

compile_one() {
  local platform="$1" outfile="$2" label="$3" note="$4" envprefix="$5"
  local tmp
  tmp="$(mktemp)"
  # shellcheck disable=SC2086
  env $envprefix "$UV" pip compile pyproject.toml \
    "${EXTRAS[@]}" \
    --python-version "$PYVER" \
    --python-platform "$platform" \
    "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}" \
    --custom-compile-command "Server/scripts/update-locks.sh" \
    --output-file "$tmp"
  {
    echo "# SteerLab Python dependency lock — $label"
    echo "#"
    echo "# REGENERATE (do not hand-edit):"
    echo "#     Server/scripts/update-locks.sh"
    echo "# which runs, from Server/:"
    echo "#     ${envprefix:+$envprefix }uv pip compile pyproject.toml --extra all \\"
    echo "#         --python-version $PYVER --python-platform $platform \\"
    echo "#         --output-file $(basename "$outfile")"
    echo "# Add --generate-hashes to that script call for a hash-pinned variant"
    echo "# (not the committed default — see Server/README.md 'Dependency locks')."
    echo "#"
    echo "$note"
    echo "#"
    cat "$tmp"
  } > "$outfile"
  rm -f "$tmp"
  echo "update-locks.sh: wrote $outfile ($(grep -cE '^[A-Za-z0-9]' "$outfile") pinned packages)"
}

compile_one aarch64-apple-darwin requirements-macos-arm64.lock \
  "macOS arm64 (Apple silicon), CPython $PYVER" \
'# MACOSX_DEPLOYMENT_TARGET=14.0 is load-bearing, not cosmetic: torch >= 2.12
# publishes its Apple-silicon wheels as macosx_14_0_arm64, and uv resolving
# against its default (older) deployment target silently pins torch back to
# 2.11.0. This project targets macOS 26.x, so 14.0 is the honest floor.
#
# The Mac is a TESTING substrate (CLAUDE.md "Models"): this lock keeps parity
# work reproducible; it never produces study evidence.' \
  "MACOSX_DEPLOYMENT_TARGET=14.0"

compile_one x86_64-unknown-linux-gnu requirements-linux-x86_64.lock \
  "Linux x86_64 + CUDA — THE SCIENCE SUBSTRATE, CPython $PYVER" \
'# torch here resolves to the default PyPI linux wheel, which carries the
# nvidia-* CUDA runtime wheels as real dependencies — correct, and the reason
# this lock is ~50 lines longer than the macOS one.
#
# BUT: which CUDA a site actually wants is a SITE fact, not ours.
# bootstrap.sh installs torch from `--torch-index` (default cu128) BEFORE it
# applies this lock, and then applies the lock with torch + its nvidia-*/triton
# runtime EXCLUDED, so a site keeps its own build. Everything else —
# transformers, accelerate, peft, numpy, scipy, safetensors, huggingface_hub —
# is pinned exactly by this file. The torch line stays here as the recorded
# reference resolution; each run stamps the torch that actually ran.' \
  ""

echo "update-locks.sh: done. Review the diff — a lock bump is a substrate change."
