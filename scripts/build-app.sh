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
#     --build-root DIR    the NEUTRAL directory the Release build runs in. The
#                         package sources are staged to DIR/src and compiled
#                         there, so no binary embeds this checkout's path —
#                         see "IDENTIFYING STRINGS" below. Must not sit under
#                         a home folder and must not contain whitespace
#                         (default: /private/var/tmp/steerlab-build)
#     --derived-data DIR  xcodebuild -derivedDataPath
#                         (default: <build-root>/dd)
#     --no-build          reuse an existing products directory
#     --no-verify         skip the post-assembly launch/codesign checks (the
#                         launch check is OFFLINE — STEERLAB_LAUNCH_CHECK=1,
#                         scratch workspace; see app-bundle/launch-check.sh)
#     --colocate-metallib  force the belt-and-braces copy of mlx.metallib
#                          beside the executable (see "Metal" below)
#     --install           move the finished bundle to ~/SteerLab/SteerLab.app
#     --force             replace an existing output or install target
#     --package           zip the SteerLab.app in --output — the bundle a build
#                         wrote there, and the one you stapled — into a
#                         signature-preserving, version-named release artifact,
#                         write its checksum beside it (<zip>.sha256), and
#                         exit. Never the installed copy. Run it AFTER
#                         stapling for the artifact you attach to a GitHub
#                         Release — stapling modifies the bundle, so a
#                         pre-staple zip is not the one to ship. It refuses:
#                           * an ad-hoc signature (the zip would not open on
#                             another Mac) unless --allow-adhoc;
#                           * a checkout that is not the release — uncommitted
#                             or untracked changes, no `v<version>` tag on
#                             HEAD, or a bundle built from another commit —
#                             unless --allow-unreleased.
#     --allow-adhoc       with --package: zip an ad-hoc-signed bundle anyway
#                         (an archive, not a release asset)
#     --allow-unreleased  with --package: zip from a dirty, untagged, or
#                         moved-on checkout anyway (not a release asset)
#     --notarize          print the notarization commands and exit (a STUB:
#                         it runs nothing and needs no credentials)
#
# Exit codes: 0 ok · 2 usage · 3 build failed · 4 products incomplete ·
#             5 assembly failed · 6 signing or verification failed ·
#             7 the assembled bundle carries identifying strings ·
#             8 --package refused: the checkout is not a clean, tagged release
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
# IDENTIFYING STRINGS: WHY THE BUILD DOES NOT RUN IN THE CHECKOUT
#
# A compiler writes source-file paths into the binary it produces. Built in
# place, both shipped executables carried the build machine's home folder —
# and so the account name — dozens of times: C and C++ `__FILE__` in the
# package checkouts under the derived-data directory, Swift `#file` in the
# dependencies still in Swift 5 mode, and this project's own `#filePath`
# (`CodeResources.compiledCheckoutPath`, which is deliberate).
#
# What was measured with Xcode 27 on a Swift package scheme, and what this
# script therefore does:
#
#   * clang's `-ffile-prefix-map=<old>=<new>` DOES rewrite `__FILE__`, and it
#     reaches package targets from the xcodebuild command line when the value
#     starts with `$(inherited)`. Passed below for C, C++ and Objective-C.
#   * Swift's `-file-prefix-map` rewrites debug, coverage and index paths
#     ONLY. A `#filePath` or Swift-5 `#file` literal is compiled in exactly as
#     the path was handed to the compiler, and xcodebuild hands over absolute
#     paths. No build setting changes that (a symlinked working directory is
#     resolved before the compiler sees it). It is still passed, so that the
#     debug information is location-independent too.
#   * So the build itself runs from a NEUTRAL location: `--build-root`.
#     `Package.swift`, `Package.resolved`, `Sources/` and `Tests/` are staged
#     to `<build-root>/src` (rsync, so an unchanged file keeps its identity
#     and incremental builds stay incremental), xcodebuild runs there, and
#     derived data defaults to `<build-root>/dd`. Every compiled-in path then
#     begins with the build root, which names nobody.
#
# Consequence, stated because it is observable: in a build made this way
# `CodeResources.compiledCheckoutPath` is `<build-root>/src`, not the
# developer's checkout. A bundled build asserts release mode and never
# resolves resources through it.
#
# None of that is taken on trust. After assembly `ci/artifact_scan.py` reads
# every byte of the bundle — both executables, every bundled resource, and
# the wheel inside the client release — for a home-folder path or a term
# from the private-name list (`~/.steerlab/private-names.txt`, or the file
# `$STEERLAB_PRIVATE_NAMES_FILE` names), and the build stops, before anything
# is signed, on a finding. An ad-hoc build runs the same scan but tolerates a
# machine that has no list (it has nothing to leak); a build signed for
# distribution requires one.
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
# Refuse stale compiled client identities before compiling a distributable app.
python3 "$SCRIPT_DIR/ci/check-python-client-identity.py" || exit 1

# Default OUTPUT deliberately sits OUTSIDE the checkout — see the iCloud note
# under "Assemble". A signed .app cannot live in iCloud Drive and still pass
# codesign verification, so the build product cannot default into the repo.
OUTPUT="$HOME/SteerLab/build"
IDENTITY="-"
BUNDLE_ID="org.steerlab.SteerLab"
# The neutral build location — see "IDENTIFYING STRINGS" in the header. /var/tmp
# rather than /tmp so the derived data survives a restart; neither names a
# person. DERIVED is resolved after flag parsing (it defaults under this).
BUILD_ROOT="/private/var/tmp/steerlab-build"
DERIVED=""
DO_BUILD=1
DO_VERIFY=1
COLOCATE=0
DO_INSTALL=0
FORCE=0
NOTARIZE_ONLY=0
PACKAGE_ONLY=0
ALLOW_ADHOC=0
ALLOW_UNRELEASED=0

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
    --build-root)    BUILD_ROOT="$2"; shift 2 ;;
    --derived-data)  DERIVED="$2"; shift 2 ;;
    --no-build)      DO_BUILD=0; shift ;;
    --no-verify)     DO_VERIFY=0; shift ;;
    --colocate-metallib) COLOCATE=1; shift ;;
    --install)       DO_INSTALL=1; shift ;;
    --force)         FORCE=1; shift ;;
    --notarize)      NOTARIZE_ONLY=1; shift ;;
    --package)       PACKAGE_ONLY=1; shift ;;
    --allow-adhoc)   ALLOW_ADHOC=1; shift ;;
    --allow-unreleased) ALLOW_UNRELEASED=1; shift ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "build-app.sh: unknown flag: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[ -n "$DERIVED" ] || DERIVED="$BUILD_ROOT/dd"

APP_FINAL="$OUTPUT/$APP_NAME"
# Everything is assembled and signed in APP (a staging path) and only then
# moved to APP_FINAL — see the staging note below.
APP="$APP_FINAL"

# One exit handler for the whole script: the assembly staging directory and
# the build-root lock are both released however the script ends.
STAGE=""
BUILD_LOCK=""
cleanup() {
  [ -n "$STAGE" ] && rm -rf "$STAGE"
  [ -n "$BUILD_LOCK" ] && rmdir "$BUILD_LOCK" 2>/dev/null
  return 0
}
trap cleanup EXIT

# ── --package: zip an existing bundle for release, and exit ──────────────────
# The artifact is the SAME shape notarytool submits (ditto -c -k --keepParent,
# which preserves the signature's extended attributes), named from the app's
# own Info.plist so a release asset states its identity:
#   SteerLab-<SLFullVersionString>+<SLSourceRevision>.zip
# Run it after `stapler staple` for the shippable zip — the ticket is stapled
# INTO the bundle, so only a post-staple zip opens promptless on a fresh Mac.
if [ "$PACKAGE_ONLY" -eq 1 ]; then
  # THE BUILD OUTPUT, and only that. This used to prefer the installed copy
  # (~/SteerLab/SteerLab.app) whenever one existed, so the zip could hold a
  # different bundle from the one that was just built, notarized, and
  # stapled — an older install, or one signed some other way.
  PKG_APP="$OUTPUT/$APP_NAME"
  [ -d "$PKG_APP" ] || die "no $APP_NAME in $OUTPUT. Build one first (scripts/build-app.sh --identity …), or pass --output <the directory it was built into>. The installed copy is never packaged: the release is the bundle a build wrote and you stapled."
  # A broken signature must fail HERE, not on a stranger's Mac.
  codesign --verify --deep --strict "$PKG_APP" \
    || die "$PKG_APP fails signature verification — rebuild/re-sign before packaging" 6
  # Captured, not piped into grep: under pipefail a reader that stops early
  # can turn a match into a failed pipeline, and "ad-hoc" would read as "not".
  SIGN_INFO="$(codesign -dv "$PKG_APP" 2>&1 || true)"
  case "$SIGN_INFO" in
    *flags=*adhoc*)
      [ "$ALLOW_ADHOC" -eq 1 ] || die "$PKG_APP is ad-hoc signed, so this zip would not open on another Mac. Rebuild with --identity \"Developer ID Application: … (TEAMID)\" for a release, or pass --allow-adhoc to archive this bundle anyway." 6
      echo "build-app.sh: NOTE — packaging an ad-hoc-signed bundle (--allow-adhoc). An archive, not a release asset." >&2
      ;;
  esac
  PKG_VERSION="$(plutil -extract SLFullVersionString raw "$PKG_APP/Contents/Info.plist" 2>/dev/null || echo unknown)"
  PKG_REV="$(plutil -extract SLSourceRevision raw "$PKG_APP/Contents/Info.plist" 2>/dev/null || echo unknown)"

  # A release asset has to be traceable to one clean, tagged commit: the
  # checkout it is packaged from is that commit, unmodified, tagged with the
  # version the bundle reports, and the bundle was built from it.
  UNRELEASED=""
  HEAD_REV="$(git -C "$REPO" rev-parse --short=8 HEAD 2>/dev/null || true)"
  if [ -z "$HEAD_REV" ]; then
    UNRELEASED="this checkout has no git history to tie the bundle to"
  else
    if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
      UNRELEASED="the checkout has uncommitted or untracked changes"
    fi
    HEAD_TAGS="$(git -C "$REPO" tag --points-at HEAD 2>/dev/null | tr '\n' ' ')"
    case " $HEAD_TAGS" in
      *" v$PKG_VERSION "*) ;;
      *) UNRELEASED="${UNRELEASED:+$UNRELEASED; }the commit $HEAD_REV is not tagged v$PKG_VERSION (tags on it: ${HEAD_TAGS:-none})" ;;
    esac
    if [ "$PKG_REV" != "$HEAD_REV" ]; then
      UNRELEASED="${UNRELEASED:+$UNRELEASED; }the bundle was built from $PKG_REV, but the checkout is at $HEAD_REV"
    fi
  fi
  if [ -n "$UNRELEASED" ]; then
    [ "$ALLOW_UNRELEASED" -eq 1 ] || die "not packaging: $UNRELEASED. A release asset comes from one clean commit tagged v$PKG_VERSION: commit your changes, tag that commit, build from it, then package. To zip this bundle anyway — as an archive, not a release asset — pass --allow-unreleased." 8
    echo "build-app.sh: NOTE — $UNRELEASED (--allow-unreleased). An archive, not a release asset." >&2
  fi

  PKG_ZIP="$OUTPUT/SteerLab-$PKG_VERSION+$PKG_REV.zip"
  rm -f "$PKG_ZIP" "$PKG_ZIP.sha256"
  ditto -c -k --keepParent "$PKG_APP" "$PKG_ZIP" || die "ditto failed"
  # The checksum file, in the form `shasum -a 256 -c` reads, naming the zip by
  # its file name so the pair can be moved or downloaded together.
  ( cd "$OUTPUT" && shasum -a 256 "$(basename "$PKG_ZIP")" > "$(basename "$PKG_ZIP").sha256" ) \
    || die "could not write $PKG_ZIP.sha256"
  if xcrun stapler validate "$PKG_APP" >/dev/null 2>&1; then
    STAPLE_NOTE="stapled — ships promptless"
  else
    STAPLE_NOTE="NOT stapled — fine for a notarytool submission, not yet the release asset"
  fi
  echo "packaged: $PKG_ZIP"
  echo "  from:   $PKG_APP ($STAPLE_NOTE)"
  echo "  $(du -sh "$PKG_ZIP" | cut -f1), sha256 $(cut -d' ' -f1 "$PKG_ZIP.sha256")"
  echo "  checksum file: $PKG_ZIP.sha256"
  echo "  release: gh release upload <tag> \"$PKG_ZIP\" \"$PKG_ZIP.sha256\""
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
       # takes (ditto -c -k --keepParent under the hood). It zips the bundle
       # in --output, never the installed copy, and it refuses a checkout
       # that is not one clean commit tagged v<version>: tag the release
       # commit before this step (or pass --allow-unreleased for a
       # submission you do not intend to publish).

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
promptless on a fresh Mac. That zip (version+revision in its name) and the
<zip>.sha256 written beside it are the GitHub Release assets:
gh release upload <tag> <zip> <zip>.sha256.
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
  # ── Stage the package at the neutral build root ────────────────────────────
  # See "IDENTIFYING STRINGS" in the header for why the compiler must not see
  # this checkout's own path.
  step "Staging the package sources at the neutral build root ($BUILD_ROOT)"
  case "$BUILD_ROOT" in
    /*) ;;
    *) die "--build-root must be an absolute path (got '$BUILD_ROOT')" 2 ;;
  esac
  # A build setting's value is split on whitespace, so a path carrying any
  # could not be written into the prefix maps below.
  case "$BUILD_ROOT$DERIVED" in
    *[[:space:]]*) die "--build-root and --derived-data must not contain whitespace" 2 ;;
  esac
  # (`[U]sers` so this script never spells a home-folder prefix itself.)
  case "$BUILD_ROOT/" in
    "$HOME"/*|/[U]sers/*|/home/*)
      echo "build-app.sh: WARNING — the build root $BUILD_ROOT is under a home folder," >&2
      echo "  so the binaries will embed it and the artifact scan will stop the build." >&2
      echo "  Drop --build-root to use the neutral default." >&2
      ;;
  esac
  mkdir -p "$BUILD_ROOT" "$DERIVED" || die "could not create $BUILD_ROOT or $DERIVED" 3
  # One build at a time per build root: two checkouts staging into the same
  # directory would compile a mixture of both.
  BUILD_LOCK="$BUILD_ROOT/.build-app.lock"
  if ! mkdir "$BUILD_LOCK" 2>/dev/null; then
    held="$BUILD_LOCK"
    BUILD_LOCK=""          # not ours — the exit handler must leave it alone
    die "another build-app.sh is using $BUILD_ROOT, or an earlier one was interrupted. If no build is running, remove $held and retry. To build two checkouts at once, give each its own --build-root outside your home folder." 3
  fi
  SRC_STAGE="$BUILD_ROOT/src"
  mkdir -p "$SRC_STAGE" || die "could not create $SRC_STAGE" 3
  # rsync, not a fresh copy: an unchanged file keeps its inode and
  # modification time, which is what lets the next build stay incremental.
  # `Tests/` travels because the package manifest declares test targets and
  # the package does not load without their directories.
  rsync -a --delete --exclude ".DS_Store" \
    "$REPO/Sources" "$REPO/Tests" "$SRC_STAGE/" \
    || die "could not stage Sources/ and Tests/ at $SRC_STAGE" 3
  rsync -a "$REPO/Package.swift" "$REPO/Package.resolved" "$SRC_STAGE/" \
    || die "could not stage the package manifest at $SRC_STAGE" 3

  # Every spelling of a directory a compiler may be handed: as given, with
  # symlinks resolved, and that physical path without its /private prefix
  # (Xcode standardizes /private/var and /private/tmp away for Swift inputs,
  # while clang is handed the physical path — both were observed).
  path_spellings() {
    local given="$1" physical
    physical="$(cd "$given" 2>/dev/null && pwd -P)" || physical="$given"
    printf '%s\n' "$given"
    [ "$physical" = "$given" ] || printf '%s\n' "$physical"
    case "$physical" in
      /private/*) [ "${physical#/private}" = "$given" ] || printf '%s\n' "${physical#/private}" ;;
    esac
  }
  # Derived data first: when it sits inside the build root's own tree a later
  # map must not shadow it.
  REMAP_CLANG=""
  REMAP_SWIFT=""
  for spelling in $(path_spellings "$DERIVED"); do
    REMAP_CLANG="$REMAP_CLANG -ffile-prefix-map=$spelling=/steerlab/build"
    REMAP_SWIFT="$REMAP_SWIFT -file-prefix-map $spelling=/steerlab/build"
  done
  for spelling in $(path_spellings "$SRC_STAGE"); do
    REMAP_CLANG="$REMAP_CLANG -ffile-prefix-map=$spelling=/steerlab/src"
    REMAP_SWIFT="$REMAP_SWIFT -file-prefix-map $spelling=/steerlab/src"
  done
  # `$(inherited)` keeps each package target's own flags: a command-line
  # setting replaces a target's value outright without it.
  REMAP_SETTINGS=(
    "OTHER_CFLAGS=\$(inherited)$REMAP_CLANG"
    "OTHER_CPLUSPLUSFLAGS=\$(inherited)$REMAP_CLANG"
    "OTHER_SWIFT_FLAGS=\$(inherited)$REMAP_SWIFT"
  )

  step "Building $SCHEME (Release) — only Xcode can build the Metal shaders"
  # CLANG_COVERAGE_MAPPING=NO: the package has test targets, so Xcode's
  # auto-generated scheme turns on gather-coverage, and that instruments even
  # plain `build` actions — the shipped binaries then carry __llvm_prf
  # sections (megabytes of them) and write default.profraw into whatever cwd
  # they run from. A command-line setting outranks the scheme; the test lane
  # (`xcodebuild test`) is a different invocation and keeps its coverage.
  ( cd "$SRC_STAGE" && xcodebuild build -skipMacroValidation -scheme "$SCHEME" \
      -destination 'platform=macOS' -configuration Release \
      CLANG_COVERAGE_MAPPING=NO "${REMAP_SETTINGS[@]}" \
      -derivedDataPath "$DERIVED" >/dev/null ) \
    || die "the build failed — rerun the xcodebuild line from $SRC_STAGE without >/dev/null to see why" 3

  # The same derived-data directory, so the CLI links the module graph the app
  # build already produced — this is a link step, not a second full build.
  step "Building $CLI_SCHEME (Release) — the CLI the bundle carries"
  ( cd "$SRC_STAGE" && xcodebuild build -skipMacroValidation -scheme "$CLI_SCHEME" \
      -destination 'platform=macOS' -configuration Release \
      CLANG_COVERAGE_MAPPING=NO "${REMAP_SETTINGS[@]}" \
      -derivedDataPath "$DERIVED" >/dev/null ) \
    || die "the $CLI_SCHEME build failed — rerun the xcodebuild line from $SRC_STAGE without >/dev/null to see why" 3
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
STAGE="${TMPDIR:-/tmp}/steerlab-app-build.$$"   # removed by `cleanup` on exit
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
# DemoWorkspaces = the worked examples the app opens as a copy: one folder per
# backend the checkout carries (mlx, mps, cuda — any subset, possibly none)
# beside a README that always ships, so the family resolves in every build.
# Checked BEFORE it is staged: each demo's shape, the size limits (one demo,
# one file), and identifying strings. The scan of the assembled bundle below
# reads the staged copy again, and the copy of it inside the client wheel.
# Dot-prefixed entries are left out: a demo must not carry them, and neither
# the resource manifest's walk nor a wheel would ship them reliably.
python3 "$SCRIPT_DIR/ci/check-demo-workspaces.py" 2>&1 | sed 's/^/  /' \
  || die "a Demo Workspace under DemoWorkspaces/ is not fit to ship (see the lines above)"
rsync -a --exclude ".*" --exclude "__pycache__" --exclude "*.pyc" \
  "$REPO/DemoWorkspaces/" "$RES/DemoWorkspaces/" || die "could not stage DemoWorkspaces"
# web/ splits by nature: index.html is hand-written SOURCE and ships in
# the repo; results-explorer/ is BUILD OUTPUT, which the repo does not
# carry. It is produced HERE, from source, on every build, and written
# straight into the bundle — never copied from the checkout's own
# web/results-explorer. That copy is a development convenience
# (run-app.sh makes it), and copying it is how the app came to ship a
# bundle that was three days older than its source: the build used
# whatever directory happened to exist.
mkdir -p "$RES/web"
cp "$REPO/web/index.html" "$RES/web/index.html" || die "web/index.html missing — it is checked-in source"
step "Building the embedded results explorer from source"
"$SCRIPT_DIR/build-results-explorer.sh" --locked --output "$RES/web/results-explorer" 2>&1 | sed 's/^/  /' \
  || die "the results explorer build failed (see the lines above). It is built from results-explorer/ with npm, which needs Node.js 22.13 or later."
cp "$SUPPORT/SteerLab.icns" "$RES/SteerLab.icns" || die "could not stage the app icon"

# ── Licenses ─────────────────────────────────────────────────────────────────
# The app is distributed as a binary, so the terms it is offered under have
# to travel INSIDE it: SteerLab's own LICENSE and NOTICE, and the license
# texts of what the build linked in or embedded. Those are collected from
# what this build actually used — every package pinned in Package.resolved,
# read from its checkout in the derived data, and the npm packages the web
# bundle just reported it contains — and the step refuses rather than omits.
step "Staging the license, the notice, and third-party notices"
cp "$REPO/LICENSE" "$RES/LICENSE" || die "LICENSE is missing from the checkout"
cp "$REPO/NOTICE" "$RES/NOTICE" || die "NOTICE is missing from the checkout"
python3 "$SCRIPT_DIR/generate-third-party-notices.py" \
  --output "$RES/THIRD-PARTY-NOTICES.txt" \
  --package-resolved "$REPO/Package.resolved" \
  --checkouts "$DERIVED/SourcePackages/checkouts" \
  --web-bundle "$RES/web/results-explorer" \
  --web-node-modules "$REPO/results-explorer/node_modules" 2>&1 | sed 's/^/  /' \
  || die "could not collect the third-party license notices (see the line above)"

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
# Both the app and app-free client use the same reviewed installer.
python3 "$SCRIPT_DIR/build-client-release.py" --output "$RES/ServerPayload/client-release" \
  || die "could not build the lightweight client release (uv is required by the release builder)"

for family in WorkspaceSeed DemoWorkspaces web AnalysisTools ClusterPayload ServerPayload; do
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

# ── Scan what was assembled ──────────────────────────────────────────────────
# The source scan reads tracked source; this reads the BUILT thing, which is
# what a stranger downloads — see "IDENTIFYING STRINGS" in the header. Before
# signing, so a bundle with a finding is never sealed, placed, or packaged.
step "Scanning the assembled bundle for identifying strings"
SCAN_ARGS=()
if [ "$IDENTITY" = "-" ]; then
  # An ad-hoc build cannot be distributed. On a machine with no private-name
  # list it still checks home-folder paths; with a list it checks both.
  SCAN_ARGS+=(--allow-missing-list)
fi
python3 "$SCRIPT_DIR/ci/artifact_scan.py" ${SCAN_ARGS[@]+"${SCAN_ARGS[@]}"} "$APP" 2>&1 | sed 's/^/  /' \
  || die "the assembled bundle carries identifying strings, or could not be scanned (see the lines above) — nothing was signed or placed. A home-folder path means a binary was not built at the neutral build root: drop --derived-data and --build-root, or point them outside your home folder." 7

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
  #
  # OFFLINE, and against a scratch workspace (a live controller on
  # 2026-09-13): this launch used to be a real one — the researcher's saved
  # sites, defaults, Keychain, and whatever tunnel happened to be up. Through
  # a live tunnel the app connected, its evidence auto-import began fetching
  # every succeeded job's bundle the local ledger had never seen, and the
  # kill below landed mid-transfer; the controller then wedged for half an
  # hour. `STEERLAB_LAUNCH_CHECK=1` makes the app refuse every network path
  # and print a verdict; app-bundle/launch-check.sh FAILS the build on any
  # refused attempt (see that script for the exact lines). The scratch
  # workspace is bootstrapped by the bundled CLI so the app opens a real,
  # empty workspace rather than the researcher's; an app that cannot
  # bootstrap one still gets an empty directory, never the real one.
  LAUNCH_LOG="$OUTPUT/launch-check.log"
  LAUNCH_WORKSPACE="${TMPDIR:-/tmp}/steerlab-launch-check.$$/workspace"
  rm -rf "${TMPDIR:-/tmp}/steerlab-launch-check.$$"
  mkdir -p "$LAUNCH_WORKSPACE" || die "could not create $LAUNCH_WORKSPACE"
  if ! ( cd / && env -u DYLD_FRAMEWORK_PATH -u DYLD_LIBRARY_PATH -u DYLD_INSERT_LIBRARIES \
        "$HELPERS/$CLI_EXECUTABLE" workspace init "$LAUNCH_WORKSPACE" >/dev/null 2>&1 ); then
    echo "  (bundled CLI could not bootstrap the scratch workspace; launching against an empty directory)"
  fi
  "$SUPPORT/launch-check.sh" "$CONTENTS/MacOS/$EXECUTABLE" "$LAUNCH_LOG" \
      --seconds 10 --workspace "$LAUNCH_WORKSPACE"
  LAUNCH_STATUS=$?
  rm -rf "${TMPDIR:-/tmp}/steerlab-launch-check.$$"
  [ "$LAUNCH_STATUS" -eq 0 ] || die "the assembled app did not pass the offline launch check (see $LAUNCH_LOG)" 6
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
