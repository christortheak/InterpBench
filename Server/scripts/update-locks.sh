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
# Usage:  Server/scripts/update-locks.sh [--upgrade] [--generate-hashes]
#
# Existing pins are kept unless the resolution must move them (2026-10-22):
# each lock is resolved with its committed predecessor as uv's preferences,
# so a constraint change moves only what it has to. --upgrade resolves
# everything fresh against today's index instead.
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
UPGRADE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --generate-hashes) EXTRA_FLAGS+=(--generate-hashes); shift ;;
    --upgrade)         UPGRADE=1; shift ;;
    *) echo "update-locks.sh: unknown flag $1 (use --upgrade, --generate-hashes)" >&2; exit 64 ;;
  esac
done

# The newest torch the science substrate's site index carries. bootstrap.sh
# asks the site's --torch-index (default download.pytorch.org/whl/cu128) for
# the lock's torch version, and that index stops at 2.11.0 (2026-10-22). A lock
# resolved against PyPI's newer torch pinned what a cu128 build cannot run with
# (setuptools 84 against its setuptools<82, the CUDA 13 runtime), and the
# cluster bootstrap replaced the site's torch to satisfy them. Raise this only
# once the site index carries the newer version:
#     pip index versions torch --index-url https://download.pytorch.org/whl/cu128
LINUX_TORCH="2.11.0"

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
  local platform="$1" outfile="$2" label="$3" note="$4" envprefix="$5" torch_pin="${6:-}"
  local tmp constraint_args=()
  tmp="$(mktemp)"
  # uv prefers the versions already in its output file: seeding it with the
  # committed lock keeps every pin the resolution does not have to move.
  if [ "$UPGRADE" -eq 0 ] && [ -f "$outfile" ]; then cp "$outfile" "$tmp"; fi
  if [ -n "$torch_pin" ]; then
    printf 'torch==%s\n' "$torch_pin" > "$tmp.constraints"
    constraint_args=(--constraint "$tmp.constraints")
  fi
  # shellcheck disable=SC2086
  env $envprefix "$UV" pip compile pyproject.toml \
    "${EXTRAS[@]}" \
    --python-version "$PYVER" \
    --python-platform "$platform" \
    "${constraint_args[@]+"${constraint_args[@]}"}" \
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
    if [ -n "$torch_pin" ]; then
      echo "#         --constraint <(echo torch==$torch_pin) \\"
    fi
    echo "#         --output-file $(basename "$outfile")"
    echo "# Add --generate-hashes to that script call for a hash-pinned variant"
    echo "# (not the committed default — see Server/README.md 'Dependency locks')."
    echo "#"
    echo "$note"
    echo "#"
    # uv names the constraint file it was given in its `# via` notes: the
    # temporary path would put this machine's temp folder in the lock.
    sed "s#-c $tmp.constraints#-c update-locks.sh LINUX_TORCH#" "$tmp"
  } > "$outfile"
  rm -f "$tmp" "$tmp.constraints"
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
'# torch is constrained to the newest version the site index carries
# (update-locks.sh LINUX_TORCH, 2.11.0 against download.pytorch.org/whl/cu128),
# so every other pin here is one that site torch can run with. It resolves to
# the default PyPI linux wheel of that version, which carries the CUDA runtime
# wheels (nvidia-*, cuda-*) as real dependencies.
#
# BUT: which CUDA a site actually wants is a SITE fact, not ours.
# bootstrap.sh asks `--torch-index` (default cu128) for the torch VERSION on
# the line below BEFORE it applies this lock, so a site gets its own build of
# that version, and then applies the lock with torch + its nvidia-*/triton
# runtime EXCLUDED and the site torch held. Everything else — transformers,
# accelerate, peft, numpy, scipy, safetensors, huggingface_hub — is pinned
# exactly by this file. Each run stamps the torch that actually ran.' \
  "" "$LINUX_TORCH"

echo "update-locks.sh: done. Review the diff — a lock bump is a substrate change."
