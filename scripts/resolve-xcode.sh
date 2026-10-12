#!/bin/zsh
# Which Xcode builds SteerLab: print its developer directory, or say plainly
# what to install or select.
#
#   usage: resolve-xcode.sh
#
# The choice, in order:
#
#   1. DEVELOPER_DIR, when it is set. An explicit choice is always respected.
#   2. The Xcode that `xcode-select -p` names, when it is a full Xcode (the
#      Command Line Tools have no xcodebuild).
#   3. /Applications/Xcode.app.
#
# No beta is ever preferred: a beta builds only when DEVELOPER_DIR or
# xcode-select names it. These scripts used to reach for
# /Applications/Xcode-beta.app whenever it existed, so installing a newer beta
# would have silently changed the toolchain of every build.
#
# The chosen Xcode must be 27 or later, read from `xcodebuild -version` run
# with DEVELOPER_DIR set to the choice, which is the xcodebuild the caller then
# runs. A Metal compiler this Xcode cannot run is a warning, not a refusal:
# MLX's shaders need it, but a build whose shaders are already compiled does
# not.
#
# Prints the developer directory on stdout, and one line on stderr naming the
# Xcode and why it was chosen. Exit codes: 0 chosen · 1 no Xcode 27 or later
# to build with (the reason, and what to install or select, on stderr).
#
# Shared by build-app.sh, install-cli.sh, run-app.sh, and both Finder
# launchers (steerlab-launcher.swift, steerlab-server-launcher.swift), so the
# rule lives in one tested place: Server/tests/test_resolve_xcode_script.py.
set -u

NEEDED=27
RELEASE_XCODE=/Applications/Xcode.app/Contents/Developer

tell() { print -r -- "$*" >&2 }

# What to do about it, worded for how the choice was made.
repair() {
  if [[ "$1" == DEVELOPER_DIR ]]; then
    tell "  Point DEVELOPER_DIR at Xcode $NEEDED or later, or unset it to use the Xcode that"
    tell "  xcode-select selects."
  else
    tell "  Install Xcode $NEEDED or later (the App Store, or https://developer.apple.com/download/),"
    tell "  then select it:"
    tell "    sudo xcode-select -s $RELEASE_XCODE"
    tell "  or name it for one command with DEVELOPER_DIR=$RELEASE_XCODE."
  fi
}

if [[ -n "${DEVELOPER_DIR:-}" ]]; then
  developer="$DEVELOPER_DIR"
  chosen_by=DEVELOPER_DIR
  why="named by DEVELOPER_DIR"
  if [[ ! -d "$developer" ]]; then
    tell "DEVELOPER_DIR is set to $developer, which does not exist."
    repair DEVELOPER_DIR
    exit 1
  fi
  # DEVELOPER_DIR may name the .app itself; its tools are under Contents/Developer.
  [[ -d "$developer/Contents/Developer" ]] && developer="$developer/Contents/Developer"
else
  selected="$(xcode-select -p 2>/dev/null)" || selected=""
  if [[ -n "$selected" && -x "$selected/usr/bin/xcodebuild" ]]; then
    developer="$selected"
    chosen_by=xcode-select
    why="selected by xcode-select"
  elif [[ -x "$RELEASE_XCODE/usr/bin/xcodebuild" ]]; then
    developer="$RELEASE_XCODE"
    chosen_by=fallback
    why="the standard location (xcode-select names no full Xcode)"
  else
    tell "No Xcode to build with: xcode-select names ${selected:-no developer directory},"
    tell "which has no xcodebuild, and there is no $RELEASE_XCODE."
    repair xcode-select
    exit 1
  fi
fi

version_output="$(DEVELOPER_DIR="$developer" xcodebuild -version 2>&1)"
code=$?
first_line="${version_output%%$'\n'*}"
if (( code != 0 )) || [[ ! "$first_line" =~ '^Xcode ([0-9]+)' ]]; then
  tell "Could not read the version of the Xcode at $developer ($why)."
  tell "xcodebuild -version said:"
  print -r -- "$version_output" | sed 's/^/    /' >&2
  repair "$chosen_by"
  exit 1
fi
major="${match[1]}"
if (( major < NEEDED )); then
  tell "SteerLab needs Xcode $NEEDED or later to build. The Xcode at $developer,"
  tell "$why, is ${first_line}."
  repair "$chosen_by"
  exit 1
fi

build="$(print -r -- "$version_output" | sed -n 's/^Build version //p')"
tell "Using ${first_line}${build:+ ($build)} at $developer, $why."

if ! DEVELOPER_DIR="$developer" xcrun metal --version >/dev/null 2>&1; then
  tell "Warning: this Xcode cannot run the Metal compiler, so MLX's shaders will not"
  tell "compile. Install its Metal toolchain once:"
  tell "    DEVELOPER_DIR=$developer xcodebuild -downloadComponent MetalToolchain"
  if [[ -n "${TOOLCHAINS:-}" ]]; then
    tell "  TOOLCHAINS is set to $TOOLCHAINS; check that it names an installed"
    tell "  toolchain (xcodebuild -showComponent MetalToolchain prints the identifier)."
  fi
fi

print -r -- "$developer"
