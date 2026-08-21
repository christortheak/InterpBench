#!/bin/zsh
# Build and launch the SteerLab chat app from the shell.
# (GPU work requires xcodebuild-built products; DYLD_FRAMEWORK_PATH makes the
# Metal shader bundle visible to a bare executable — see CLAUDE.md › Build & run.)
set -e
cd "$(dirname "$0")/.."
if [[ -d /Applications/Xcode-beta.app ]]; then
  export DEVELOPER_DIR=/Applications/Xcode-beta.app/Contents/Developer
fi
# CLANG_COVERAGE_MAPPING=NO: the auto-generated scheme gathers coverage even
# for plain builds (the package has test targets) — an instrumented app drops
# default.profraw into its cwd, which is how a 6 MB one ended up at the repo
# root. The test lane keeps its own coverage.
xcodebuild build -skipMacroValidation -scheme SteerLabApp \
  -destination 'platform=macOS,arch=arm64' CLANG_COVERAGE_MAPPING=NO \
  -derivedDataPath .deriveddata.nosync | tail -2
# Re-sign the built executable with the stable Apple Development identity
# when one exists (2026-08-05): the default ad-hoc, linker-signed build gets
# a NEW code signature every rebuild, so the keychain treats each build as a
# different app and every launch re-prompts for the stored keys — "Always
# Allow" can never stick. A certificate-backed signature keeps the designated
# requirement stable across rebuilds, so one Always Allow per key is durable.
# Post-build (not via xcodebuild flags) because manual signing of the SPM
# package targets fails ("requires a development team" on swift-crypto).
# Falls back to ad-hoc (previous behavior, with the re-prompts) on machines
# without an identity.
if security find-identity -v -p codesigning 2>/dev/null | grep -q "Apple Development"; then
  codesign --force --sign "Apple Development" \
    .deriveddata.nosync/Build/Products/Debug/SteerLabApp
fi
export DYLD_FRAMEWORK_PATH=.deriveddata.nosync/Build/Products/Debug
exec ./.deriveddata.nosync/Build/Products/Debug/SteerLabApp
