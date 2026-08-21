#!/usr/bin/env bash
# Assemble (and optionally sign / install) SteerLab.app — WP2, the packaged
# Mac instrument.
#
#   usage: build-app.sh [flags]
#     --output DIR        where SteerLab.app is written
#                         (default: ~/SteerLab/build — NOT inside the
#                         checkout; a signed bundle cannot live in iCloud
#                         Drive, see "iCloud" below)
#     --identity NAME     codesign identity; "-" is ad-hoc (default: "-").
#                         For a shippable build pass the Developer ID, e.g.
#                           --identity "Developer ID Application: … (TEAMID)"
#     --bundle-id ID      CFBundleIdentifier (default: org.steerlab.SteerLab)
#     --derived-data DIR  xcodebuild -derivedDataPath
#                         (default: <repo>/.dd-app.nosync)
#     --no-build          reuse an existing products directory
#     --no-verify         skip the post-assembly launch/codesign checks
#     --colocate-metallib  force the belt-and-braces copy of mlx.metallib
#                          beside the executable (see "Metal" below)
#     --install           move the finished bundle to ~/SteerLab/SteerLab.app
#     --force             replace an existing output or install target
#     --package           zip an already-built SteerLab.app (installed copy
#                         first, then the build dir) into a signature-
#                         preserving, version-named release artifact and exit.
#                         Run it AFTER stapling for the artifact you attach to
#                         a GitHub Release — stapling modifies the bundle, so
#                         a pre-staple zip is not the one to ship.
#     --notarize          print the notarization commands and exit (a STUB:
#                         it runs nothing and needs no credentials)
#
# Exit codes: 0 ok · 2 usage · 3 build failed · 4 products incomplete ·
#             5 assembly failed · 6 signing or verification failed
#
# ─────────────────────────────────────────────────────────────────────────────
# WHY A SCRIPT AND NOT AN .xcodeproj
#
# This repo is script-first: `install-cli.sh` and `make-server-payload.sh`
# already assemble distributable trees from an xcodebuild products directory,
# and the same approach works here because the app has no Xcode-only build
# phases to express — no asset catalog, no Interface Builder output, no
# entitlement-driven capability. The SPM executable `SteerLabApp` links
# statically and its resources are plain directories that CodeResources
# resolves at runtime. Checking in an .xcodeproj would add a second,
# drifting definition of the product for no capability gained. Revisit only
# if the app acquires something SwiftPM genuinely cannot emit (an asset
# catalog compiled by actool is the likeliest trigger — see "Follow-ups").
#
# ─────────────────────────────────────────────────────────────────────────────
# HARDENED RUNTIME: THE SPIKE VERDICT (2026-08-20)
#
# MLX inference WORKS under the hardened runtime with ZERO entitlements. The
# evidence, the two controls that make it a real result, and the reasoning
# for each entitlement deliberately NOT requested live in
# scripts/app-bundle/SteerLab.entitlements. Read that file before adding an
# entitlement to this build; the short version is that the executable
# statically links the whole graph (one Mach-O, no @rpath dylibs, so library
# validation has nothing to reject) and MLX's shaders are precompiled into
# mlx.metallib and loaded as DATA through Metal, never as JIT pages.
#
# This script therefore signs with `--options runtime` and passes the
# entitlements file to codesign ONLY when that file declares at least one
# key. While it stays empty, the signature carries no entitlement blob at
# all — the strongest posture, and the one that notarizes most cleanly.
#
# ─────────────────────────────────────────────────────────────────────────────
# iCloud
#
# This checkout lives in iCloud Drive, whose fileprovider attaches
# com.apple.FinderInfo to DIRECTORIES on its own schedule — a *.nosync suffix
# does not exempt them. codesign refuses to sign such a directory, and
# `codesign --verify --strict` refuses to validate one, so:
#
#   * the bundle is assembled and signed under TMPDIR, and
#   * the default output is ~/SteerLab/build, outside iCloud.
#
# Pointing --output back inside the checkout produces a bundle that signs but
# then fails its own verification, so the script warns when you do. This is
# measured behavior on this machine, not caution.
#
# ─────────────────────────────────────────────────────────────────────────────
# METAL
#
# Only Xcode produces mlx-swift_Cmlx.bundle (SwiftPM cannot build the Metal
# shaders — CLAUDE.md › Build & run), which is why this script drives
# xcodebuild and refuses to proceed without that bundle in the products
# directory. It is copied into Contents/Resources/, where MLX finds it
# through the main bundle exactly as it does in any packaged app.
#
# `--colocate-metallib` additionally drops a copy of the shader library at
# Contents/MacOS/mlx.metallib. MLX probes a COLOCATED mlx.metallib before
# any bundle lookup — that is the mechanism install-cli.sh relies on for the
# environment-free CLI install. It is off by default here because the
# in-bundle Resources lookup was verified to work on its own (see VERIFY
# below); the flag exists so a future MLX change that breaks bundle lookup
# has a one-word remedy rather than a debugging session.
#
# ─────────────────────────────────────────────────────────────────────────────
# THE BUNDLED CLI (Contents/Helpers/steerlab-cli)
#
# The distribution promise is "no Xcode required", and until this step the
# only way to get `steerlab-cli` was `install-cli.sh`, which builds it with
# xcodebuild. So the app now CARRIES the release binary and the docs point at
# it; install-cli.sh becomes the developer path, not the user path.
#
# Three facts shape the layout, and all three were measured:
#
#   * BUNDLE IDENTITY. CFBundle makes the enclosing .app the main bundle only
#     for an executable in Contents/MacOS/. From Contents/Helpers/,
#     `Bundle.main` is the HELPER'S OWN DIRECTORY, so nothing in
#     Contents/Resources is reachable through it. `CodeResources
#     .enclosingAppBundle` derives the .app from the layout instead and probes
#     its Contents/Resources — that seam is what makes this location work, and
#     `steerlab-cli install version` prints the family-by-family proof.
#   * METAL. MLX probes a COLOCATED mlx.metallib before any bundle lookup
#     (the mechanism install-cli.sh rests on), and the helper's bundle lookup
#     lands in Contents/Helpers. So the shader library is colocated THERE —
#     the one deliberate duplicate in this bundle, and the reason GPU verbs
#     work from the bundled CLI with no environment at all.
#   * SIGNING. A Mach-O under Contents/ is nested code: it is signed
#     explicitly, before the outer app, with the same hardened-runtime flags
#     as the main executable, or `--verify --deep --strict` fails on it.
#
# The documented way to reach it is a symlink on PATH, and BOTH halves of that
# were measured rather than hoped for. MLX's colocated lookup asks `dladdr`,
# which reports the RESOLVED path, so the shaders are found through a symlink
# (install-cli.sh's shim-not-symlink rule is about its own `bin/` layout, and
# is not evidence about this one). CFBundle, in the other direction, reports
# the path the process was LAUNCHED with, so a symlink would otherwise make
# `Bundle.main` the symlink's directory — `CodeResources.executableDirectory`
# is what closes that, and VERIFY below exercises the symlink shape, not just
# the direct one.
#
# The helper is NOT stamped with a resource-manifest.json: `install stamp`
# WRITES beside the binary, and the first write into a signed bundle breaks
# the seal. `install verify` is therefore an install-cli.sh answer; the
# bundle's own integrity answer is its code signature.
#
# ─────────────────────────────────────────────────────────────────────────────
# NOTARIZATION — WHAT THE RESEARCHER RUNS NEXT
#
# Not run here: it needs an Apple Developer account, a Developer ID
# Application certificate, and an app-specific password. `--notarize` prints
# the exact commands. Two hard prerequisites the ad-hoc default cannot meet:
# the identity must be a "Developer ID Application" certificate, and the
# signature must carry a SECURE TIMESTAMP (this script passes `--timestamp`
# automatically for any non-ad-hoc identity).
#
# ─────────────────────────────────────────────────────────────────────────────
# FOLLOW-UPS (none blocking)
#   * App icon: scripts/app-bundle/SteerLab.icns (generated by make-icon.swift
#     in the same directory), staged into Resources and named by
#     CFBundleIconFile. If an asset catalog is
#     wanted instead, that is the one thing that would justify an .xcodeproj.
#   * No Sparkle / update feed. The bundle id and version keys are already
#     shaped for one.
#   * The resource-manifest walk below duplicates the convention in
#     make-server-payload.sh; worth factoring into a shared helper once a
#     third caller appears.
#
# NOTE ON `pipefail`: several steps pipe a tool through `sed` to indent its
# output. Without pipefail the `|| die` after such a pipeline tests SED's exit
# status, not the tool's — which silently swallowed a codesign failure during
# development and produced a bundle that carried only the linker's ad-hoc
# signature while the script reported success. Do not remove it.
set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"
SUPPORT="$SCRIPT_DIR/app-bundle"

# Default OUTPUT deliberately sits OUTSIDE the checkout — see the iCloud note
# under "Assemble". A signed .app cannot live in iCloud Drive and still pass
# codesign verification, so the build product cannot default into the repo.
OUTPUT="$HOME/SteerLab/build"
IDENTITY="-"
BUNDLE_ID="org.steerlab.SteerLab"
DERIVED="$REPO/.dd-app.nosync"
DO_BUILD=1
DO_VERIFY=1
COLOCATE=0
DO_INSTALL=0
FORCE=0
NOTARIZE_ONLY=0
PACKAGE_ONLY=0

SCHEME="SteerLabApp"
EXECUTABLE="SteerLabApp"
# The headless runner, shipped inside the bundle so app users need no Xcode —
# see "THE BUNDLED CLI" in the header.
CLI_SCHEME="steerlab-cli"
CLI_EXECUTABLE="steerlab-cli"
HELPERS_DIR_NAME="Helpers"
APP_NAME="SteerLab.app"
INSTALL_DIR="$HOME/SteerLab"

usage() { sed -n '/^#   usage:/,/^# Exit codes/p' "$0" | sed 's/^# \{0,3\}//'; }
die() { echo "build-app.sh: $1" >&2; exit "${2:-5}"; }
step() { echo; echo "── $1"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --output)        OUTPUT="$2"; shift 2 ;;
    --identity)      IDENTITY="$2"; shift 2 ;;
    --bundle-id)     BUNDLE_ID="$2"; shift 2 ;;
    --derived-data)  DERIVED="$2"; shift 2 ;;
    --no-build)      DO_BUILD=0; shift ;;
    --no-verify)     DO_VERIFY=0; shift ;;
    --colocate-metallib) COLOCATE=1; shift ;;
    --install)       DO_INSTALL=1; shift ;;
    --force)         FORCE=1; shift ;;
    --notarize)      NOTARIZE_ONLY=1; shift ;;
    --package)       PACKAGE_ONLY=1; shift ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "build-app.sh: unknown flag: $1" >&2; usage >&2; exit 2 ;;
  esac
done

APP_FINAL="$OUTPUT/$APP_NAME"
# Everything is assembled and signed in APP (a staging path) and only then
# moved to APP_FINAL — see the staging note below.
APP="$APP_FINAL"

# ── --package: zip an existing bundle for release, and exit ──────────────────
# The artifact is the SAME shape notarytool submits (ditto -c -k --keepParent,
# which preserves the signature's extended attributes), named from the app's
# own Info.plist so a release asset states its identity:
#   SteerLab-<SLFullVersionString>+<SLSourceRevision>.zip
# Run it after `stapler staple` for the shippable zip — the ticket is stapled
# INTO the bundle, so only a post-staple zip opens promptless on a fresh Mac.
if [ "$PACKAGE_ONLY" -eq 1 ]; then
  PKG_APP=""
  for candidate in "$INSTALL_DIR/$APP_NAME" "$OUTPUT/$APP_NAME"; do
    if [ -d "$candidate" ]; then PKG_APP="$candidate"; break; fi
  done
  [ -n "$PKG_APP" ] || die "no $APP_NAME found in $INSTALL_DIR or $OUTPUT — build one first"
  # A broken signature must fail HERE, not on a stranger's Mac.
  codesign --verify --deep --strict "$PKG_APP" \
    || die "$PKG_APP fails signature verification — rebuild/re-sign before packaging"
  if codesign -dv "$PKG_APP" 2>&1 | grep -q "flags=.*adhoc"; then
    echo "build-app.sh: WARNING — $PKG_APP is ad-hoc signed; this zip will" >&2
    echo "  not open on other Macs. Fine for archiving, wrong for a release." >&2
  fi
  PKG_VERSION="$(plutil -extract SLFullVersionString raw "$PKG_APP/Contents/Info.plist" 2>/dev/null || echo unknown)"
  PKG_REV="$(plutil -extract SLSourceRevision raw "$PKG_APP/Contents/Info.plist" 2>/dev/null || echo unknown)"
  PKG_ZIP="$OUTPUT/SteerLab-$PKG_VERSION+$PKG_REV.zip"
  mkdir -p "$OUTPUT" || die "could not create $OUTPUT"
  rm -f "$PKG_ZIP"
  ditto -c -k --keepParent "$PKG_APP" "$PKG_ZIP" || die "ditto failed"
  if xcrun stapler validate "$PKG_APP" >/dev/null 2>&1; then
    STAPLE_NOTE="stapled — ships promptless"
  else
    STAPLE_NOTE="NOT stapled — fine for a notarytool submission, not yet the release asset"
  fi
  echo "packaged: $PKG_ZIP"
  echo "  from:   $PKG_APP ($STAPLE_NOTE)"
  echo "  $(du -sh "$PKG_ZIP" | cut -f1), sha256 $(shasum -a 256 "$PKG_ZIP" | cut -d' ' -f1)"
  echo "  release: gh release upload <tag> \"$PKG_ZIP\""
  exit 0
fi

# ── --notarize: print, run nothing ───────────────────────────────────────────
if [ "$NOTARIZE_ONLY" -eq 1 ]; then
  cat <<NOTARIZE
Notarization is a STUB in this script — the commands below are printed, never
run, because they need your Apple Developer credentials.

Prerequisites
  1. Re-assemble the bundle signed with a Developer ID Application
     certificate (ad-hoc and "Apple Development" are BOTH rejected by the
     notary service):

       scripts/build-app.sh --identity "Developer ID Application: NAME (TEAMID)"

     The secure timestamp notarization requires is added automatically for
     any non-ad-hoc identity.

  2. Store credentials once, so the password never sits in your shell
     history or in this repo:

       xcrun notarytool store-credentials "steerlab-notary" \\
         --apple-id "YOUR_APPLE_ID" \\
         --team-id "YOUR_TEAM_ID" \\
         --password "YOUR_APP_SPECIFIC_PASSWORD"

Submit, staple, confirm
       scripts/build-app.sh --package
       # prints the versioned zip path — the SAME artifact shape notarytool
       # takes (ditto -c -k --keepParent under the hood):

       xcrun notarytool submit "<the printed .zip>" \\
         --keychain-profile "steerlab-notary" --wait

       # On "Accepted" — staple the ticket INTO the .app, not the zip, so the
       # bundle validates offline:
       xcrun stapler staple "$APP"
       xcrun stapler validate "$APP"

       # The real acceptance test, and the one that fails before notarization:
       spctl --assess --type execute -vvv "$APP"

  If the submission is rejected, the log names every offending binary:
       xcrun notarytool log <SUBMISSION_ID> --keychain-profile "steerlab-notary"

Distribute the STAPLED .app: re-run  scripts/build-app.sh --package  AFTER
stapling — the ticket lives in the bundle, so only the post-staple zip opens
promptless on a fresh Mac. That zip (version+revision in its name, sha256
printed) is the GitHub Release asset:  gh release upload <tag> <zip>.
NOTARIZE
  exit 0
fi

# ── Preflight ────────────────────────────────────────────────────────────────
[ -f "$SUPPORT/Info.plist.template" ] || die "missing $SUPPORT/Info.plist.template"
[ -f "$SUPPORT/SteerLab.entitlements" ] || die "missing $SUPPORT/SteerLab.entitlements"
command -v python3 >/dev/null 2>&1 || die "python3 is required (Info.plist + manifest)"

# An output inside iCloud Drive signs but cannot then verify (see the iCloud
# note in the header). Warn rather than refuse: the bundle is still usable for
# inspection, and someone may want it there deliberately.
case "$OUTPUT" in
  *"/Library/Mobile Documents/"*)
    echo "build-app.sh: WARNING — $OUTPUT is inside iCloud Drive." >&2
    echo "  The fileprovider re-attaches com.apple.FinderInfo to the .app" >&2
    echo "  directory, so codesign verification and Gatekeeper WILL fail there" >&2
    echo "  even though signing itself succeeds. Prefer a path outside iCloud" >&2
    echo "  (the default is ~/SteerLab/build)." >&2
    ;;
esac

# This project needs Xcode 27 and `xcode-select` may point at a 26.x install.
if [ -z "${DEVELOPER_DIR:-}" ] && [ -d /Applications/Xcode-beta.app ]; then
  export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
fi

# ── Build ────────────────────────────────────────────────────────────────────
if [ "$DO_BUILD" -eq 1 ]; then
  step "Building $SCHEME (Release) — only Xcode can build the Metal shaders"
  # CLANG_COVERAGE_MAPPING=NO: the package has test targets, so Xcode's
  # auto-generated scheme turns on gather-coverage, and that instruments even
  # plain `build` actions — the shipped binaries then carry __llvm_prf
  # sections (megabytes of them) and write default.profraw into whatever cwd
  # they run from. A command-line setting outranks the scheme; the test lane
  # (`xcodebuild test`) is a different invocation and keeps its coverage.
  xcodebuild build -skipMacroValidation -scheme "$SCHEME" \
    -destination 'platform=macOS' -configuration Release \
    CLANG_COVERAGE_MAPPING=NO \
    -derivedDataPath "$DERIVED" >/dev/null \
    || die "the build failed — rerun the xcodebuild line without >/dev/null to see why" 3

  # The same derived-data directory, so the CLI links the module graph the app
  # build already produced — this is a link step, not a second full build.
  step "Building $CLI_SCHEME (Release) — the CLI the bundle carries"
  xcodebuild build -skipMacroValidation -scheme "$CLI_SCHEME" \
    -destination 'platform=macOS' -configuration Release \
    CLANG_COVERAGE_MAPPING=NO \
    -derivedDataPath "$DERIVED" >/dev/null \
    || die "the $CLI_SCHEME build failed — rerun the xcodebuild line without >/dev/null to see why" 3
fi

PRODUCTS="$DERIVED/Build/Products/Release"
[ -x "$PRODUCTS/$EXECUTABLE" ] || die "no built $EXECUTABLE at $PRODUCTS" 4
[ -x "$PRODUCTS/$CLI_EXECUTABLE" ] \
  || die "no built $CLI_EXECUTABLE at $PRODUCTS — the bundle ships it (drop --no-build, or build the $CLI_SCHEME scheme into this derived-data directory)" 4
CMLX="$PRODUCTS/mlx-swift_Cmlx.bundle"
METALLIB="$CMLX/Contents/Resources/default.metallib"
[ -f "$METALLIB" ] || die "no Metal shader library at $METALLIB (only Xcode produces it)" 4

# ── Versions ─────────────────────────────────────────────────────────────────
FULL_VERSION="$(sed -n 's/.*static let version = "\(.*\)"/\1/p' \
  "$REPO/Sources/ExperimentKit/SteerLabVersion.swift" 2>/dev/null | head -n1)"
[ -n "$FULL_VERSION" ] || die "could not read the version from SteerLabVersion.swift" 4
# Apple requires 1-3 dot-separated integers; "0.9.0-dev" -> "0.9.0".
SHORT_VERSION="$(printf '%s' "$FULL_VERSION" | sed 's/[^0-9.].*$//; s/\.$//')"
[ -n "$SHORT_VERSION" ] || SHORT_VERSION="0.0.0"
MIN_SYSTEM="$(sed -n 's/.*\.macOS("\([0-9.]*\)").*/\1/p' "$REPO/Package.swift" | head -n1)"
[ -n "$MIN_SYSTEM" ] || die "could not read the deployment target from Package.swift" 4

SOURCE_REVISION=""
if command -v git >/dev/null 2>&1; then
  SOURCE_REVISION="$(git -C "$REPO" rev-parse --short=8 HEAD 2>/dev/null || true)"
fi

echo "  version   $FULL_VERSION (CFBundleShortVersionString $SHORT_VERSION)"
echo "  revision  ${SOURCE_REVISION:-<unresolved>}"
echo "  min macOS $MIN_SYSTEM"

# ── Assemble ─────────────────────────────────────────────────────────────────
if [ -e "$APP_FINAL" ]; then
  [ "$FORCE" -eq 1 ] || die "$APP_FINAL already exists — pass --force to replace it"
fi

# ── Staging, and why it is NOT optional here ─────────────────────────────────
# This checkout lives in iCloud Drive, and the fileprovider re-applies
# com.apple.FinderInfo (plus com.apple.fileprovider.fpfs#P) to DIRECTORIES on
# its own schedule — the .app and the nested .bundle directories included,
# and a *.nosync path does not exempt them. codesign refuses any such
# directory outright ("resource fork, Finder information, or similar detritus
# not allowed"), and because the attributes come back while signing is still
# in progress, one `xattr -cr` sweep beforehand does not hold. Measured, not
# theorized: that is exactly how the first working version of this script
# failed.
#
# So the bundle is assembled and signed under TMPDIR, outside iCloud
# entirely, and only the finished, signed bundle is moved to the output path.
# That also buys the atomicity install-cli.sh has: nothing live is touched
# until the whole thing is built, signed, and sealed.
STAGE="${TMPDIR:-/tmp}/steerlab-app-build.$$"
cleanup() { rm -rf "$STAGE"; }
trap cleanup EXIT
rm -rf "$STAGE"
mkdir -p "$STAGE" || die "could not create the staging directory $STAGE"
APP="$STAGE/$APP_NAME"

step "Assembling $APP_NAME (staged outside iCloud, in $STAGE)"
CONTENTS="$APP/Contents"
RES="$CONTENTS/Resources"
mkdir -p "$CONTENTS/MacOS" "$RES" || die "could not create the bundle skeleton"

cp "$PRODUCTS/$EXECUTABLE" "$CONTENTS/MacOS/$EXECUTABLE" || die "could not copy the executable"
chmod +x "$CONTENTS/MacOS/$EXECUTABLE"

# SwiftPM resource bundles that exist beside the binary. mlx-swift_Cmlx is
# load-bearing (the shaders); the other two are tokenizer/crypto resources.
for bundle in mlx-swift_Cmlx.bundle swift-transformers_Hub.bundle swift-crypto_Crypto.bundle; do
  [ -d "$PRODUCTS/$bundle" ] && cp -R "$PRODUCTS/$bundle" "$RES/$bundle"
done
if [ "$COLOCATE" -eq 1 ]; then
  cp "$METALLIB" "$CONTENTS/MacOS/mlx.metallib" || die "could not colocate mlx.metallib"
  echo "  colocated Contents/MacOS/mlx.metallib"
fi

# ── The bundled CLI ──────────────────────────────────────────────────────────
# See "THE BUNDLED CLI" in the header for why each of these three lands here
# rather than being reached through Contents/Resources. The colocated
# mlx.metallib is NOT optional the way --colocate-metallib is for the app: the
# helper's own bundle lookup resolves to Contents/Helpers, so this copy is the
# only thing a GPU verb can find.
step "Staging $HELPERS_DIR_NAME/$CLI_EXECUTABLE (the CLI the app ships)"
HELPERS="$CONTENTS/$HELPERS_DIR_NAME"
mkdir -p "$HELPERS" || die "could not create $HELPERS"
cp "$PRODUCTS/$CLI_EXECUTABLE" "$HELPERS/$CLI_EXECUTABLE" || die "could not copy $CLI_EXECUTABLE"
chmod +x "$HELPERS/$CLI_EXECUTABLE"
cp "$METALLIB" "$HELPERS/mlx.metallib" || die "could not colocate the helper's mlx.metallib"
# Resolved through Bundle.main by the packages that own them, which for a
# helper means "beside the helper" — the same set install-cli.sh carries, and
# 40 KB between them.
for bundle in swift-transformers_Hub.bundle swift-crypto_Crypto.bundle; do
  [ -d "$PRODUCTS/$bundle" ] && cp -R "$PRODUCTS/$bundle" "$HELPERS/$bundle"
done
printf '  %-16s %s\n' "$HELPERS_DIR_NAME" "$(du -sh "$HELPERS" | cut -f1)"

# CodeResources families. The names are the raw values of
# CodeResources.Family and MUST match, because a packaged build resolves them
# as Bundle.main.resourceURL/<name> and fails closed when one is missing.
step "Staging CodeResources families"
cp -R "$REPO/WorkspaceSeed" "$RES/WorkspaceSeed" || die "could not stage WorkspaceSeed"
# web/ splits by nature: index.html is hand-written SOURCE and ships in
# the repo; results-explorer/ is BUILD OUTPUT the repo deliberately does
# not carry (the CI lane's rule — a cold clone must produce it), so the
# app build produces it here when absent. First caught building from a
# fresh clone (2026-08-20): every earlier build ran from a tree that
# happened to carry both.
mkdir -p "$RES/web"
cp "$REPO/web/index.html" "$RES/web/index.html" || die "web/index.html missing — it is checked-in source"
if [ ! -d "$REPO/web/results-explorer" ]; then
  step "Building the embedded results explorer (web/results-explorer is not checked in)"
  command -v npm >/dev/null 2>&1 || die "npm is required to build the results explorer (web/results-explorer is build output, produced from results-explorer/)"
  ( cd "$REPO/results-explorer" && npm ci --silent && npm run --silent build:embed ) || die "results-explorer build failed"
fi
cp -R "$REPO/web/results-explorer" "$RES/web/results-explorer" || die "could not stage the results explorer build"
cp "$SUPPORT/SteerLab.icns" "$RES/SteerLab.icns" || die "could not stage the app icon"

# AnalysisTools = the checkout's scripts/, minus generated caches.
rsync -a --exclude "__pycache__" --exclude "*.pyc" --exclude ".DS_Store" \
  "$REPO/scripts/" "$RES/AnalysisTools/" || die "could not stage AnalysisTools"

# ClusterPayload = exactly what ClusterProvisioner pushes (filtered Server/ +
# prompts/fixtures/) plus deployment-manifest.json. Reuse the existing staging
# tool rather than re-deriving its filter rules here — the payload must stay
# byte-for-byte what clusters already receive.
"$SCRIPT_DIR/make-server-payload.sh" --source "$REPO" --output "$RES/ClusterPayload" \
  --force >/dev/null || die "make-server-payload.sh failed"

# ServerPayload = the filtered Server/ tree, taken from the payload just
# staged so the two can never disagree.
cp -R "$RES/ClusterPayload/Server" "$RES/ServerPayload" || die "could not stage ServerPayload"

for family in WorkspaceSeed web AnalysisTools ClusterPayload ServerPayload; do
  printf '  %-16s %s\n' "$family" "$(du -sh "$RES/$family" | cut -f1)"
done

# ── Info.plist ───────────────────────────────────────────────────────────────
step "Writing Info.plist"
SL_TEMPLATE="$SUPPORT/Info.plist.template" \
SL_OUT="$CONTENTS/Info.plist" \
SL_BUNDLE_ID="$BUNDLE_ID" SL_EXECUTABLE="$EXECUTABLE" \
SL_SHORT_VERSION="$SHORT_VERSION" SL_BUNDLE_VERSION="$SHORT_VERSION" \
SL_FULL_VERSION="$FULL_VERSION" SL_SOURCE_REVISION="$SOURCE_REVISION" \
SL_MIN_SYSTEM="$MIN_SYSTEM" \
python3 - <<'PY' || die "Info.plist substitution failed"
import os, plistlib, re, sys
text = open(os.environ["SL_TEMPLATE"], encoding="utf-8").read()
for key, env in (
    ("__BUNDLE_ID__", "SL_BUNDLE_ID"),
    ("__EXECUTABLE__", "SL_EXECUTABLE"),
    ("__SHORT_VERSION__", "SL_SHORT_VERSION"),
    ("__BUNDLE_VERSION__", "SL_BUNDLE_VERSION"),
    ("__FULL_VERSION__", "SL_FULL_VERSION"),
    ("__SOURCE_REVISION__", "SL_SOURCE_REVISION"),
    ("__MIN_SYSTEM__", "SL_MIN_SYSTEM"),
):
    text = text.replace(key, os.environ[env])
leftover = sorted(set(re.findall(r"__[A-Z][A-Z_]*__", text)))
if leftover:
    sys.exit(f"unsubstituted placeholder(s): {leftover}")
out = os.environ["SL_OUT"]
open(out, "w", encoding="utf-8").write(text)
# Parse it back: a malformed Info.plist makes the bundle unlaunchable in a
# way that is tedious to diagnose later.
with open(out, "rb") as handle:
    plistlib.load(handle)
PY
echo "  $BUNDLE_ID"

# ── resource-manifest.json ───────────────────────────────────────────────────
# CodeResources.Family.buildManifest: a packaged build carries it, and
# SteerLabVersion.current prefers it over the runtime git read. Walk
# conventions match ResourceManifest.generate (sorted, dot-prefixed entries
# skipped, "/"-relative paths, lowercase hex) — the same convention
# make-server-payload.sh implements.
step "Stamping resource-manifest.json"
SL_ROOT="$RES" SL_APP_VERSION="$FULL_VERSION" \
SL_SOURCE_REVISION="$SOURCE_REVISION" \
python3 - <<'PY' || die "resource manifest generation failed"
import hashlib, json, os
root = os.environ["SL_ROOT"]
out = os.path.join(root, "resource-manifest.json")
if os.path.exists(out):
    os.remove(out)          # never let the manifest record a stale self
files = {}
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
    for name in sorted(filenames):
        if name.startswith("."):
            continue
        full = os.path.join(dirpath, name)
        if os.path.islink(full):
            continue
        rel = os.path.relpath(full, root).replace(os.sep, "/")
        digest = hashlib.sha256()
        with open(full, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        files[rel] = digest.hexdigest()
manifest = {
    "schemaVersion": 1,
    "appVersion": os.environ["SL_APP_VERSION"],
    "serverVersion": "bundled",
    "protocolVersion": 1,
    "files": files,
}
revision = os.environ.get("SL_SOURCE_REVISION", "")
if revision:
    manifest["sourceRevision"] = revision
with open(out, "w", encoding="utf-8") as handle:
    json.dump(manifest, handle, sort_keys=True, indent=2)
print(f"  {len(files)} file(s) hashed")
PY

# ── Sign ─────────────────────────────────────────────────────────────────────
# Staging in TMPDIR is what actually keeps the iCloud fileprovider's
# FinderInfo out of the way (see the staging note above); this sweep is the
# cheap belt-and-braces for anything the copies dragged in from the checkout.
step "Signing (hardened runtime)"
xattr -cr "$APP" 2>/dev/null || true

SIGN_ARGS=(--force --options runtime)
if [ "$IDENTITY" != "-" ]; then
  # Secure timestamp: required by the notary service, impossible for ad-hoc.
  SIGN_ARGS+=(--timestamp)
fi

# Pass entitlements only when at least one is declared — while the file is an
# empty dict (the spike's verdict) the signature carries no entitlement blob.
ENTITLEMENT_COUNT="$(SL_ENT="$SUPPORT/SteerLab.entitlements" python3 -c '
import os, plistlib
with open(os.environ["SL_ENT"], "rb") as handle:
    print(len(plistlib.load(handle) or {}))
' 2>/dev/null || echo 0)"
if [ "$ENTITLEMENT_COUNT" -gt 0 ]; then
  SIGN_ARGS+=(--entitlements "$SUPPORT/SteerLab.entitlements")
  echo "  entitlements: $ENTITLEMENT_COUNT declared"
else
  echo "  entitlements: none (spike verdict — see app-bundle/SteerLab.entitlements)"
fi

# Inside-out: nested bundles first, then the nested CLI, the app last, so each
# seal covers already-signed content.
for nested in "$RES"/*.bundle "$HELPERS"/*.bundle; do
  [ -e "$nested" ] || continue
  codesign "${SIGN_ARGS[@]}" -s "$IDENTITY" "$nested" 2>&1 | sed 's/^/  /' \
    || die "signing $(basename "$nested") failed" 6
done
# EVERYTHING under Contents/Helpers is nested code as far as codesign is
# concerned — `Helpers` is one of the directory names its built-in nested-code
# rules name, alongside MacOS, Frameworks, PlugIns and XPCServices — so the
# shader library needs its own signature too, and the outer signature refuses
# ("code object is not signed at all") until it has one. Measured: that is
# exactly how the first version of this step failed. `mlx.metallib` is a
# MetalLib executable and signs as a generic code object; it must be signed
# BEFORE the helper binary and the app, like any other nested content.
codesign "${SIGN_ARGS[@]}" -s "$IDENTITY" "$HELPERS/mlx.metallib" 2>&1 | sed 's/^/  /' \
  || die "signing the helper's mlx.metallib failed" 6
# The helper itself: same SIGN_ARGS as the main executable — same hardened
# runtime, same timestamp policy, same entitlements decision — because it is
# the same program graph, statically linked, and a weaker posture on the
# helper would be the bundle's weakest link.
codesign "${SIGN_ARGS[@]}" -s "$IDENTITY" "$HELPERS/$CLI_EXECUTABLE" 2>&1 | sed 's/^/  /' \
  || die "signing $CLI_EXECUTABLE failed" 6
codesign "${SIGN_ARGS[@]}" -s "$IDENTITY" \
  --identifier "$BUNDLE_ID" "$APP" 2>&1 | sed 's/^/  /' \
  || die "signing the app failed" 6

# ── Move the finished bundle into place ──────────────────────────────────────
# `ditto` rather than `mv`/`cp -R`: the staging directory and the output are
# usually on different volumes, and ditto is the tool that carries a signed
# bundle across one intact. Everything from here on verifies the bundle the
# caller actually gets, not the staged copy.
step "Placing the bundle at $APP_FINAL"
mkdir -p "$OUTPUT" || die "could not create $OUTPUT"
rm -rf "$APP_FINAL" || die "could not remove the existing $APP_FINAL"
ditto "$APP" "$APP_FINAL" || die "could not move the bundle into place"
rm -rf "$STAGE"
APP="$APP_FINAL"
CONTENTS="$APP/Contents"
RES="$CONTENTS/Resources"
HELPERS="$CONTENTS/$HELPERS_DIR_NAME"

# ── Verify ───────────────────────────────────────────────────────────────────
if [ "$DO_VERIFY" -eq 1 ]; then
  step "Verifying"

  codesign --verify --deep --strict --verbose=2 "$APP" 2>&1 | sed 's/^/  /' \
    || die "codesign --verify --deep --strict failed" 6

  echo "  linked libraries (must be system-only — the graph links statically):"
  otool -L "$CONTENTS/MacOS/$EXECUTABLE" | tail -n +2 | sed 's/^/    /'

  # The bundled CLI, run the way a user's symlink runs it: from the bundle,
  # with no DYLD_* environment and from a cwd that is not the checkout. This
  # is the check that would have caught the Contents/Helpers bundle-identity
  # problem (Bundle.main is the helper's own directory, not the .app), so it
  # asserts on the OUTPUT rather than merely on the exit code — `install
  # version` prints the family-by-family resolution and the Metal answer.
  # Through a SYMLINK, because that is what the docs tell people to make —
  # and it is the shape that fails differently from a direct invocation (see
  # "THE BUNDLED CLI"). The link lives in a scratch directory that is thrown
  # away; nothing is installed on this machine by verifying.
  echo "  bundled CLI ($HELPERS_DIR_NAME/$CLI_EXECUTABLE, reached by symlink):"
  CLI_LOG="$OUTPUT/cli-check.log"
  CLI_LINK_DIR="${TMPDIR:-/tmp}/steerlab-cli-check.$$"
  mkdir -p "$CLI_LINK_DIR" || die "could not create $CLI_LINK_DIR"
  ln -sf "$HELPERS/$CLI_EXECUTABLE" "$CLI_LINK_DIR/$CLI_EXECUTABLE"
  if ( cd / && env -u DYLD_FRAMEWORK_PATH -u DYLD_LIBRARY_PATH -u DYLD_INSERT_LIBRARIES \
        "$CLI_LINK_DIR/$CLI_EXECUTABLE" --version >"$CLI_LOG" 2>&1 ); then
    sed 's/^/    /' "$CLI_LOG"
  else
    sed 's/^/    /' "$CLI_LOG" >&2
    rm -rf "$CLI_LINK_DIR"
    die "the bundled $CLI_EXECUTABLE would not run out of the assembled bundle" 6
  fi
  rm -rf "$CLI_LINK_DIR"
  if grep -q "PROBLEMS:" "$CLI_LOG"; then
    die "the bundled $CLI_EXECUTABLE cannot resolve every resource family (see $CLI_LOG) — Contents/Resources is not reachable from Contents/$HELPERS_DIR_NAME" 6
  fi
  grep -q "release mode" "$CLI_LOG" \
    || die "the bundled $CLI_EXECUTABLE did not assert release mode — it is resolving out of a checkout on this machine, which a user's Mac will not have" 6
  grep -q "mlx.metallib colocated" "$CLI_LOG" \
    || die "the bundled $CLI_EXECUTABLE found no colocated mlx.metallib — GPU verbs would refuse to load shaders" 6

  # Gatekeeper. An ad-hoc or Apple Development signature FAILS here and that
  # is the expected, correct answer before notarization — report it, never
  # hide it. Captured rather than piped so the status read is spctl's own.
  echo "  spctl --assess:"
  SPCTL_OUT="$(spctl --assess --type execute -vv "$APP" 2>&1)"
  SPCTL_STATUS=$?
  printf '%s\n' "$SPCTL_OUT" | sed 's/^/    /'
  if [ "$SPCTL_STATUS" -eq 0 ]; then
    echo "    -> accepted"
  else
    echo "    -> rejected, exit $SPCTL_STATUS (expected until the bundle is"
    echo "       notarized and stapled; ad-hoc and Apple Development"
    echo "       identities can never pass Gatekeeper assessment)"
  fi

  # Launch it the way a user will: no DYLD_* environment whatsoever. The app
  # opens a window; a few seconds is enough to clear initialization, which is
  # where a missing resource or a bad install name would abort.
  echo "  launching with no DYLD_* environment…"
  LAUNCH_LOG="$OUTPUT/launch-check.log"
  ( cd / && env -u DYLD_FRAMEWORK_PATH -u DYLD_LIBRARY_PATH -u DYLD_INSERT_LIBRARIES \
      "$CONTENTS/MacOS/$EXECUTABLE" >"$LAUNCH_LOG" 2>&1 ) &
  LAUNCH_PID=$!
  sleep 8
  if kill -0 "$LAUNCH_PID" 2>/dev/null; then
    echo "    -> alive past initialization ✓"
    kill "$LAUNCH_PID" 2>/dev/null
    wait "$LAUNCH_PID" 2>/dev/null
  else
    wait "$LAUNCH_PID" 2>/dev/null
    echo "    -> EXITED EARLY. Output:" >&2
    sed 's/^/      /' "$LAUNCH_LOG" >&2
    die "the assembled app did not stay up" 6
  fi
  if [ -s "$LAUNCH_LOG" ]; then
    echo "    launch output ($LAUNCH_LOG):"
    sed 's/^/      /' "$LAUNCH_LOG"
  else
    echo "    no output on stdout/stderr ✓"
  fi
fi

# ── Install ──────────────────────────────────────────────────────────────────
if [ "$DO_INSTALL" -eq 1 ]; then
  step "Installing to $INSTALL_DIR/$APP_NAME"
  mkdir -p "$INSTALL_DIR" || die "could not create $INSTALL_DIR"
  if [ -e "$INSTALL_DIR/$APP_NAME" ]; then
    [ "$FORCE" -eq 1 ] || die "$INSTALL_DIR/$APP_NAME already exists — pass --force to replace it"
    rm -rf "$INSTALL_DIR/$APP_NAME" || die "could not remove the existing install"
  fi
  mv "$APP" "$INSTALL_DIR/$APP_NAME" || die "could not move the bundle into place"
  APP="$INSTALL_DIR/$APP_NAME"
fi

echo
echo "SteerLab.app ready: $APP"
echo "  $(du -sh "$APP" | cut -f1) total"
echo "  CLI: $APP/Contents/$HELPERS_DIR_NAME/$CLI_EXECUTABLE"
echo "       ln -s \"$APP/Contents/$HELPERS_DIR_NAME/$CLI_EXECUTABLE\" ~/.local/bin/steerlab-cli"
if [ "$IDENTITY" = "-" ]; then
  echo "  signed ad-hoc — fine for local use, NOT distributable."
  echo "  For a shippable build: --identity \"Developer ID Application: … (TEAMID)\","
  echo "  then scripts/build-app.sh --notarize for the next commands."
fi
