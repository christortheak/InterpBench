#!/usr/bin/env bash
# Build the embedded Results Explorer from source.
#
#   usage: build-results-explorer.sh [--output DIR] [--locked] [--if-stale]
#     --output DIR   where the built bundle is written
#                    (default: <repo>/web/results-explorer)
#     --locked       reinstall the dependencies from the committed lockfile
#                    first (`npm ci`), whatever is already installed — what a
#                    release build wants
#     --if-stale     do nothing when DIR already holds a build that is newer
#                    than every source file — what a development run wants
#
# Exit codes: 0 built (or already fresh) · 1 the build failed ·
#             2 usage · 3 npm is not installed
#
# WHY THIS EXISTS. `web/results-explorer/` is BUILD OUTPUT: a minified bundle
# produced from `results-explorer/` by `npm run build:embed`. It used to be
# committed, and the app build used it whenever the directory existed — so the
# app shipped whatever was last committed, which fell behind the source
# without anything noticing. It is no longer tracked. The two things that
# need the bundle each produce it:
#
#   * `build-app.sh` runs this with `--locked --output <bundle>/web/…`, every
#     time, straight into the bundle being assembled. The checkout's own copy
#     is never consulted, so a stale one cannot ship.
#   * `run-app.sh` runs this with `--if-stale`, so a development run from a
#     fresh clone gets a working Results Explorer and a later source change
#     is picked up.
#
# Without `--locked`, dependencies are installed only when `node_modules` is
# missing or older than the lockfile.
set -u

usage() { sed -n '/^#   usage:/,/^# Exit codes/p' "$0" | sed 's/^# \{0,3\}//'; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
SOURCE="$REPO/results-explorer"
OUTPUT="$REPO/web/results-explorer"
LOCKED=0
IF_STALE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --output)   [ $# -ge 2 ] || { usage >&2; exit 2; }; OUTPUT="$2"; shift 2 ;;
    --locked)   LOCKED=1; shift ;;
    --if-stale) IF_STALE=1; shift ;;
    -h|--help)  usage; exit 0 ;;
    *) echo "build-results-explorer.sh: unknown flag: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[ -f "$SOURCE/package.json" ] || {
  echo "build-results-explorer.sh: no results-explorer/package.json under $REPO — not a SteerLab checkout" >&2
  exit 1
}

# Fresh = the built index.html exists and no source file is newer than it.
# `node_modules` and the tool caches are not source.
if [ "$IF_STALE" -eq 1 ] && [ -f "$OUTPUT/index.html" ]; then
  newer="$(find "$SOURCE" \
      \( -name node_modules -o -name .wrangler -o -name .vinext -o -name .next -o -name dist \) -prune \
      -o -type f -newer "$OUTPUT/index.html" -print 2>/dev/null | head -n 1)"
  if [ -z "$newer" ]; then
    echo "results explorer: $OUTPUT is up to date"
    exit 0
  fi
fi

command -v npm >/dev/null 2>&1 || {
  echo "build-results-explorer.sh: npm is not installed. The embedded Results Explorer is built from" >&2
  echo "  results-explorer/ and is not kept in the repository. Install Node.js 22.13 or later, then run" >&2
  echo "  this again." >&2
  exit 3
}

cd "$SOURCE" || exit 1
if [ "$LOCKED" -eq 1 ] || [ ! -d node_modules ] \
    || [ package-lock.json -nt node_modules/.package-lock.json ]; then
  echo "results explorer: installing dependencies from the lockfile (npm ci)"
  npm ci --silent || { echo "build-results-explorer.sh: npm ci failed" >&2; exit 1; }
fi
echo "results explorer: building $OUTPUT from source"
# vite.embed.config.ts writes to STEERLAB_EMBED_OUT_DIR when it is set.
STEERLAB_EMBED_OUT_DIR="$OUTPUT" npm run --silent build:embed \
  || { echo "build-results-explorer.sh: the embedded build failed" >&2; exit 1; }
[ -f "$OUTPUT/index.html" ] || {
  echo "build-results-explorer.sh: the build reported success but wrote no $OUTPUT/index.html" >&2
  exit 1
}
