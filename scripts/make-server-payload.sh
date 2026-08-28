#!/usr/bin/env bash
# Build the immutable cluster deployment payload from a SteerLab checkout
# (MAC-DISTRIBUTION-AND-MANAGED-SERVER-PROPOSAL §10, Phase A wave 3).
#
# The payload is EXACTLY what `ClusterProvisioner` pushes from a dev checkout
# today — the filtered `Server/` tree (bootstrap + Slurm templates live
# inside `Server/scripts/`) plus the `prompts/fixtures/` parity fixtures —
# staged with the same rsync filter rules as
# `ClusterProvisioner.pushFilterArguments` (KEEP THE TWO LISTS IN SYNC),
# plus one addition: `deployment-manifest.json`, listing every payload
# file's SHA-256 in the `ResourceManifest` schema (schemaVersion 1) so the
# app can verify the payload before rsync and bootstrap.sh's manifestCheck
# step can verify it again on the cluster.
#
# Deterministic: the manifest walk is sorted and skips dot-prefixed entries,
# matching `ResourceManifest.generate`; two runs over the same tree produce
# byte-identical manifests (given the same versions/revision).
#
#   usage: make-server-payload.sh [flags]
#     --source DIR            SteerLab checkout          (default: this script's repo)
#     --output DIR            payload directory to create
#                             (default: <source>/.build.nosync/server-payload)
#     --app-version V         (default: parsed from SteerLabVersion.swift)
#     --server-version V      (default: parsed from Server/pyproject.toml)
#     --protocol-version N    (default: 1)
#     --force                 replace an existing output directory
#
# Exit codes: 0 = payload built and manifest written; 1 = failure.
set -u

usage() { sed -n '/^#   usage:/,/^# Exit codes/p' "$0" | sed 's/^# \{0,3\}//'; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE="$(dirname "$SCRIPT_DIR")"
OUTPUT=""
APP_VERSION=""
SERVER_VERSION=""
PROTOCOL_VERSION="1"
FORCE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --source)           SOURCE="$2"; shift 2 ;;
    --output)           OUTPUT="$2"; shift 2 ;;
    --app-version)      APP_VERSION="$2"; shift 2 ;;
    --server-version)   SERVER_VERSION="$2"; shift 2 ;;
    --protocol-version) PROTOCOL_VERSION="$2"; shift 2 ;;
    --force)            FORCE=1; shift ;;
    -h|--help)          usage; exit 0 ;;
    *) echo "make-server-payload.sh: unknown flag: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [ ! -d "$SOURCE/Server" ]; then
  echo "make-server-payload.sh: $SOURCE has no Server/ — not a SteerLab checkout" >&2
  exit 1
fi
[ -n "$OUTPUT" ] || OUTPUT="$SOURCE/.build.nosync/server-payload"

command -v python3 >/dev/null 2>&1 || {
  echo "make-server-payload.sh: python3 is required for manifest generation" >&2
  exit 1
}

# Version defaults come from the sources of truth in the checkout.
if [ -z "$APP_VERSION" ]; then
  APP_VERSION="$(sed -n 's/.*static let version = "\(.*\)"/\1/p' \
    "$SOURCE/Sources/ExperimentKit/SteerLabVersion.swift" 2>/dev/null | head -n1)"
fi
if [ -z "$SERVER_VERSION" ]; then
  SERVER_VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' \
    "$SOURCE/Server/pyproject.toml" 2>/dev/null | head -n1)"
fi
if [ -z "$APP_VERSION" ] || [ -z "$SERVER_VERSION" ]; then
  echo "make-server-payload.sh: could not resolve versions (app='$APP_VERSION'" >&2
  echo "  server='$SERVER_VERSION') — pass --app-version/--server-version" >&2
  exit 1
fi

# sourceRevision: only when SOURCE is itself the top of a git repo (never a
# parent repo's SHA for an arbitrary directory). Short=8 matches
# SteerLabVersion's convention. Omitted when unavailable.
SOURCE_REVISION=""
if command -v git >/dev/null 2>&1; then
  toplevel="$(git -C "$SOURCE" rev-parse --show-toplevel 2>/dev/null || true)"
  if [ -n "$toplevel" ] \
    && [ "$(cd "$toplevel" && pwd -P)" = "$(cd "$SOURCE" && pwd -P)" ]; then
    SOURCE_REVISION="$(git -C "$SOURCE" rev-parse --short=8 HEAD 2>/dev/null || true)"
  fi
fi

if [ -e "$OUTPUT" ]; then
  if [ "$FORCE" -eq 1 ]; then
    rm -rf "$OUTPUT"
  else
    echo "make-server-payload.sh: $OUTPUT already exists — payloads are" >&2
    echo "  immutable; pass --force to replace it" >&2
    exit 1
  fi
fi
mkdir -p "$OUTPUT" || exit 1

# ------------------------------------------------------------------ staging --
# The EXACT filter rules of ClusterProvisioner.pushFilterArguments: local
# environments, run outputs, generated caches, and wheel-build debris
# (Server/build/, Server/dist/ — left behind by `python -m build`) never enter
# the payload; Server/ (including Server/tests/ — shipped today) and
# prompts/fixtures/ do.
rsync -a --prune-empty-dirs \
  --exclude ".venv*" \
  --exclude "runs" \
  --exclude "__pycache__" \
  --exclude ".pytest_cache" \
  --exclude ".mypy_cache" \
  --exclude ".ruff_cache" \
  --exclude "*.egg-info" \
  --exclude "*.pyc" \
  --exclude ".coverage*" \
  --exclude ".DS_Store" \
  --exclude "/Server/build/" \
  --exclude "/Server/dist/" \
  --include "/Server/" \
  --include "/Server/***" \
  --include "/prompts/" \
  --include "/prompts/fixtures/" \
  --include "/prompts/fixtures/***" \
  --exclude "*" \
  "$SOURCE/" "$OUTPUT/" || {
  echo "make-server-payload.sh: rsync staging failed" >&2
  exit 1
}

# ----------------------------------------------------------------- manifest --
# Same schema and walk conventions as ResourceManifest.generate: every
# regular file as a sorted "/"-relative path -> lowercase-hex SHA-256;
# dot-prefixed entries skipped entirely; sourceRevision omitted when absent.
MANIFEST_JSON_PATH="$OUTPUT/deployment-manifest.json" \
MANIFEST_APP_VERSION="$APP_VERSION" \
MANIFEST_SERVER_VERSION="$SERVER_VERSION" \
MANIFEST_PROTOCOL_VERSION="$PROTOCOL_VERSION" \
MANIFEST_SOURCE_REVISION="$SOURCE_REVISION" \
MANIFEST_ROOT="$OUTPUT" \
python3 - <<'PYEOF' || exit 1
import hashlib, json, os, sys

root = os.environ["MANIFEST_ROOT"]
manifest_path = os.environ["MANIFEST_JSON_PATH"]
files = {}
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
    for name in sorted(filenames):
        if name.startswith("."):
            continue
        full = os.path.join(dirpath, name)
        relative = os.path.relpath(full, root).replace(os.sep, "/")
        if relative == "deployment-manifest.json":
            continue
        digest = hashlib.sha256()
        with open(full, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        files[relative] = digest.hexdigest()

manifest = {
    "schemaVersion": 1,
    "appVersion": os.environ["MANIFEST_APP_VERSION"],
    "serverVersion": os.environ["MANIFEST_SERVER_VERSION"],
    "protocolVersion": int(os.environ["MANIFEST_PROTOCOL_VERSION"]),
    "files": files,
}
revision = os.environ.get("MANIFEST_SOURCE_REVISION", "")
if revision:
    manifest["sourceRevision"] = revision

with open(manifest_path, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, sort_keys=True, indent=2)
count = len(files)
total = sum(
    os.path.getsize(os.path.join(root, rel.replace("/", os.sep)))
    for rel in files
)
print(f"make-server-payload.sh: {count} files, {total} bytes, "
      f"manifest at {manifest_path}")
PYEOF
