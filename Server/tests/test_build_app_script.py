"""``scripts/build-app.sh`` — the parts of the app build that can be held to
their contract without Xcode.

The script drives ``xcodebuild``, ``codesign``, and friends. Each test here
puts stand-ins for those tools first on ``PATH`` (they record how they were
called and do nothing), runs the real script against scratch directories, and
reads what it did. Nothing is compiled, signed, installed, or launched, and
nothing is written outside ``tmp_path``.
"""

import os
import pathlib
import shutil
import stat
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build-app.sh"

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("bash") is None,
    reason="build-app.sh assembles a macOS app bundle")


def _stand_in(directory: pathlib.Path, name: str, body: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run(tmp_path, *arguments, stand_ins=None, env=None):
    """Run build-app.sh with stand-in tools first on PATH and a scratch HOME."""
    fakebin = tmp_path / "fakebin"
    log = tmp_path / "calls.log"
    tools = {
        # Records where it was run from and with what, and builds nothing —
        # so the script stops at "no built SteerLabApp" (exit 4), after the
        # build step has shown its whole hand.
        "xcodebuild": 'printf "cwd=%s\\n" "$PWD" >> "$CALLS"\n'
                      'for word in "$@"; do printf "arg=%s\\n" "$word" >> "$CALLS"; done\n',
    }
    tools.update(stand_ins or {})
    for name, body in tools.items():
        _stand_in(fakebin, name, body)
    # The interpreter running this suite, so the script's own python3 steps
    # do not depend on which developer tools the machine has selected.
    _stand_in(fakebin, "python3", f'exec "{sys.executable}" "$@"\n')
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    environment = {
        "PATH": f"{fakebin}:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(home),
        "TMPDIR": str(tmp_path / "tmp"),
        "CALLS": str(log),
    }
    (tmp_path / "tmp").mkdir(exist_ok=True)
    environment.update(env or {})
    result = subprocess.run(
        ["/bin/bash", str(SCRIPT), *map(str, arguments)],
        env=environment, text=True, capture_output=True, check=False, cwd=tmp_path)
    calls = log.read_text().splitlines() if log.exists() else []
    return result, calls


#: Stand-ins that let the script run PAST the build, through assembly, to a
#: placed bundle: an xcodebuild that "builds" two small executables and a
#: shader library into the derived-data directory it is given, a uv that
#: writes a small wheel, and a codesign that only records that it was called.
#: `$APP_BINARY_TEXT` is what the fake app executable contains, so a test can
#: plant a string in a "binary".
ASSEMBLING_STAND_INS = {
    "xcodebuild": r'''
printf "cwd=%s\n" "$PWD" >> "$CALLS"
derived=""
while [ $# -gt 0 ]; do
  case "$1" in -derivedDataPath) derived="$2"; shift 2 ;; *) shift ;; esac
done
products="$derived/Build/Products/Release"
mkdir -p "$products/mlx-swift_Cmlx.bundle/Contents/Resources"
printf '\xcf\xfa\xed\xfe a stand-in app executable\0%s\0' "${APP_BINARY_TEXT:-}" > "$products/SteerLabApp"
printf '\xcf\xfa\xed\xfe a stand-in CLI executable\0' > "$products/steerlab-cli"
chmod +x "$products/SteerLabApp" "$products/steerlab-cli"
printf 'MTLB stand-in shaders' > "$products/mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"
# What a real build leaves behind: one checkout per pinned package, each with
# its license. (The working directory is the staged package.)
python3 -c 'import json, pathlib, sys
for pin in json.load(open("Package.resolved"))["pins"]:
    checkout = pathlib.Path(sys.argv[1]) / pin["identity"]
    checkout.mkdir(parents=True, exist_ok=True)
    (checkout / "LICENSE").write_text("stand-in license of " + pin["identity"] + "\n")' \
  "$derived/SourcePackages/checkouts"
''',
    "uv": r'''
out=""
while [ $# -gt 0 ]; do
  case "$1" in --out-dir) out="$2"; shift 2 ;; *) shift ;; esac
done
python3 -c 'import sys, zipfile
with zipfile.ZipFile(sys.argv[1], "w") as wheel:
    wheel.writestr("steerlab_server/__init__.py", "VALUE = 1\n")' "$out/steerlab_server-0.0.0-py3-none-any.whl"
''',
    "codesign": 'printf "codesign %s\\n" "$*" >> "$CALLS"\n',
    # `npm ci`, then `npm run build:embed`: the second writes a bundle to the
    # directory the build asked for, marked so a test can tell it from any
    # copy that happens to sit in the checkout.
    "npm": r'''
printf "npm %s\n" "$*" >> "$CALLS"
case "$*" in
  *build:embed*)
    mkdir -p "$STEERLAB_EMBED_OUT_DIR/assets"
    printf '<!doctype html><title>built from source by this build</title>\n' > "$STEERLAB_EMBED_OUT_DIR/index.html"
    printf 'console.log("stand-in")\n' > "$STEERLAB_EMBED_OUT_DIR/assets/index.js"
    printf '[]\n' > "$STEERLAB_EMBED_OUT_DIR/bundled-packages.json"   # bundles no npm package
    ;;
esac
''',
}


def _assemble(tmp_path, *arguments, app_binary_text="", env=None):
    """Run the script through assembly with the stand-ins above. Returns the
    result, the recorded tool calls, and where the bundle would be placed."""
    output = tmp_path / "out"
    environment = {"APP_BINARY_TEXT": app_binary_text,
                   # Never the machine's own list unless a test names one.
                   "STEERLAB_PRIVATE_NAMES_FILE": str(tmp_path / "no-list.txt")}
    environment.update(env or {})
    result, calls = _run(
        tmp_path, "--build-root", tmp_path / "neutral", "--output", output,
        "--no-verify", *arguments, stand_ins=ASSEMBLING_STAND_INS, env=environment)
    return result, calls, output / "SteerLab.app"


# -- the Release build runs from a neutral location ---------------------------


def test_the_build_runs_from_the_neutral_build_root_not_the_checkout(tmp_path):
    """A compiler writes the paths it is handed into the binary, and no build
    setting rewrites a Swift `#filePath`. So the compile must not happen in
    the checkout: the sources are staged at the build root and xcodebuild is
    run THERE, with derived data beside them."""
    build_root = tmp_path / "neutral"
    result, calls = _run(tmp_path, "--build-root", build_root,
                         "--output", tmp_path / "out", "--no-verify")
    # The stand-in built nothing, so the script stops right after the build.
    assert result.returncode == 4, result.stdout + result.stderr
    assert "no built SteerLabApp" in result.stderr

    stage = build_root / "src"
    working_directories = [line[4:] for line in calls if line.startswith("cwd=")]
    assert len(working_directories) == 2                 # the app, then the CLI
    for directory in working_directories:
        assert pathlib.Path(directory).resolve() == stage.resolve()
        assert pathlib.Path(directory).resolve() != ROOT.resolve()

    # What the package needs to load and compile was staged, and only that.
    assert (stage / "Package.swift").read_bytes() == (ROOT / "Package.swift").read_bytes()
    assert (stage / "Package.resolved").is_file()
    assert (stage / "Sources/ExperimentKit/CodeResources.swift").is_file()
    assert (stage / "Tests/ExperimentKitTests").is_dir()
    assert not (stage / "Server").exists()
    assert not (stage / "docs").exists()
    assert not (stage / ".git").exists()

    arguments = [line[4:] for line in calls if line.startswith("arg=")]
    derived = arguments[arguments.index("-derivedDataPath") + 1]
    assert pathlib.Path(derived) == build_root / "dd"
    assert arguments.count("Release") == 2


def test_the_build_maps_the_build_root_out_of_compiled_paths(tmp_path):
    """Belt and braces for what a flag CAN rewrite: C-family `__FILE__` and
    all debug information. `$(inherited)` keeps each package target's own
    flags, which a bare command-line setting would replace."""
    build_root = tmp_path / "neutral"
    _, calls = _run(tmp_path, "--build-root", build_root,
                    "--output", tmp_path / "out", "--no-verify")
    arguments = [line[4:] for line in calls if line.startswith("arg=")]
    settings = {a.split("=", 1)[0]: a.split("=", 1)[1] for a in arguments if "=" in a
                and a.split("=", 1)[0].startswith("OTHER_")}
    assert set(settings) == {"OTHER_CFLAGS", "OTHER_CPLUSPLUSFLAGS", "OTHER_SWIFT_FLAGS"}
    for value in settings.values():
        assert value.startswith("$(inherited) ")
    physical = build_root.resolve()
    for name in ("OTHER_CFLAGS", "OTHER_CPLUSPLUSFLAGS"):
        assert f"-ffile-prefix-map={physical}/dd=/steerlab/build" in settings[name]
        assert f"-ffile-prefix-map={physical}/src=/steerlab/src" in settings[name]
    assert f"-file-prefix-map {physical}/dd=/steerlab/build" in settings["OTHER_SWIFT_FLAGS"]
    assert f"-file-prefix-map {physical}/src=/steerlab/src" in settings["OTHER_SWIFT_FLAGS"]
    # macOS spells one directory two ways (/private/var/… and /var/…), and
    # Xcode hands compilers both: each spelling gets its own map.
    if str(physical).startswith("/private/"):
        short = str(physical)[len("/private"):]
        assert f"-ffile-prefix-map={short}/src=/steerlab/src" in settings["OTHER_CFLAGS"]
        assert f"-file-prefix-map {short}/src=/steerlab/src" in settings["OTHER_SWIFT_FLAGS"]
    # The coverage setting the build already relied on is still there.
    assert "CLANG_COVERAGE_MAPPING=NO" in arguments


def test_a_second_build_in_the_same_root_is_refused_and_the_lock_is_released(tmp_path):
    build_root = tmp_path / "neutral"
    lock = build_root / ".build-app.lock"
    lock.mkdir(parents=True)
    result, calls = _run(tmp_path, "--build-root", build_root,
                         "--output", tmp_path / "out", "--no-verify")
    assert result.returncode == 3
    assert "another build-app.sh is using" in result.stderr
    assert "--build-root" in result.stderr
    assert calls == []                       # nothing was compiled
    assert lock.is_dir()                     # someone else's lock is left alone

    lock.rmdir()
    result, _ = _run(tmp_path, "--build-root", build_root,
                     "--output", tmp_path / "out", "--no-verify")
    assert result.returncode == 4
    assert not lock.exists()                 # ours is released on the way out


def test_a_build_root_the_prefix_maps_cannot_express_is_refused(tmp_path):
    result, calls = _run(tmp_path, "--build-root", tmp_path / "has space",
                         "--output", tmp_path / "out", "--no-verify")
    assert result.returncode == 2
    assert "must not contain whitespace" in result.stderr
    assert calls == []
    relative, calls = _run(tmp_path, "--build-root", "relative/root",
                           "--output", tmp_path / "out", "--no-verify")
    assert relative.returncode == 2
    assert "absolute path" in relative.stderr
    assert calls == []


# -- what was assembled is scanned before it is signed -------------------------

HOME_ROOT = "/" + "Users"                # never spelled whole in this file


def test_a_clean_bundle_is_scanned_then_signed_and_placed(tmp_path):
    result, calls, app = _assemble(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Scanning the assembled bundle for identifying strings" in result.stdout
    assert "artifact scan: clean" in result.stdout
    # This machine has no list (the test points at none) and the build is
    # ad-hoc, so the scan covered home-folder paths and said so.
    assert "home-folder paths ONLY" in result.stdout
    assert any(line.startswith("codesign ") for line in calls)
    assert (app / "Contents/MacOS/SteerLabApp").is_file()
    assert (app / "Contents/Helpers/steerlab-cli").is_file()


def test_a_bundle_with_a_home_folder_path_is_never_signed_or_placed(tmp_path):
    """The state the published app was in: a binary carrying the build
    machine's home folder. The build must stop before the seal."""
    leaked = f"{HOME_ROOT}/rosalind/SteerLab/Sources/ExperimentKit/CodeResources.swift"
    result, calls, app = _assemble(tmp_path, app_binary_text=leaked)
    assert result.returncode == 7, result.stdout + result.stderr
    assert "SteerLab.app/Contents/MacOS/SteerLabApp [home-path]" in result.stdout
    assert "nothing was signed or placed" in result.stderr
    assert "--build-root" in result.stderr            # the repair, in the refusal
    assert not any(line.startswith("codesign ") for line in calls)
    assert not app.exists()


def test_a_bundle_with_a_private_name_is_never_signed_or_placed(tmp_path):
    names = tmp_path / "private-names.txt"
    names.write_text("quuxcluster\n")               # made up
    result, calls, app = _assemble(
        tmp_path, app_binary_text="ssh login.QuuxCluster.example",
        env={"STEERLAB_PRIVATE_NAMES_FILE": str(names)})
    assert result.returncode == 7, result.stdout + result.stderr
    assert "SteerLab.app/Contents/MacOS/SteerLabApp [private-name]" in result.stdout
    assert "list entry 1 (11 letters)" in result.stdout
    assert "quuxcluster" not in (result.stdout + result.stderr).lower()
    assert not any(line.startswith("codesign ") for line in calls)
    assert not app.exists()


def test_the_results_explorer_is_built_from_source_into_the_bundle(tmp_path):
    """The app once shipped a results explorer three days older than its
    source, because the build copied `web/results-explorer` whenever that
    directory existed. The bundle's copy now comes from a build this run
    made — locked dependencies first — whatever the checkout holds."""
    result, calls, app = _assemble(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    npm = [line for line in calls if line.startswith("npm ")]
    assert npm[0].startswith("npm ci")                     # from the lockfile
    assert any("build:embed" in line for line in npm[1:])
    web = app / "Contents/Resources/web"
    assert "built from source by this build" in (web / "results-explorer/index.html").read_text()
    assert (web / "results-explorer/assets/index.js").is_file()
    # The hand-written page beside it is tracked source and is copied as is.
    assert (web / "index.html").read_bytes() == (ROOT / "web/index.html").read_bytes()


def test_the_bundle_carries_its_license_its_notice_and_third_party_notices(tmp_path):
    """The app is distributed as a binary, so its terms have to be inside it:
    SteerLab's own, and those of every package the build linked in."""
    import json
    result, _, app = _assemble(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    resources = app / "Contents/Resources"
    assert (resources / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()
    assert (resources / "NOTICE").read_bytes() == (ROOT / "NOTICE").read_bytes()
    notices = (resources / "THIRD-PARTY-NOTICES.txt").read_text()
    pins = json.loads((ROOT / "Package.resolved").read_text())["pins"]
    assert len(pins) >= 10
    for pin in pins:                       # every pinned package, none omitted
        assert f"\n{pin['identity']} " in notices
        assert f"stand-in license of {pin['identity']}" in notices
    # …and all three are in the resource manifest, so a tampered copy is caught.
    manifest = json.loads((resources / "resource-manifest.json").read_text())["files"]
    assert {"LICENSE", "NOTICE", "THIRD-PARTY-NOTICES.txt"} <= set(manifest)


def test_a_package_with_no_license_stops_the_build(tmp_path):
    """Refuse rather than omit: a notices file that silently left a linked
    package out would be worse than none."""
    stand_ins = dict(ASSEMBLING_STAND_INS)
    stand_ins["xcodebuild"] += (
        'rm -f "$derived/SourcePackages/checkouts/mlx-swift/LICENSE"\n')
    result, calls = _run(
        tmp_path, "--build-root", tmp_path / "neutral", "--output", tmp_path / "out",
        "--no-verify", stand_ins=stand_ins,
        env={"STEERLAB_PRIVATE_NAMES_FILE": str(tmp_path / "no-list.txt")})
    assert result.returncode == 5
    assert "the package 'mlx-swift' has no license file" in result.stdout
    assert "could not collect the third-party license notices" in result.stderr
    assert not any(line.startswith("codesign ") for line in calls)
    assert not (tmp_path / "out" / "SteerLab.app").exists()


def test_without_npm_the_build_stops_and_says_what_is_needed(tmp_path):
    stand_ins = {name: body for name, body in ASSEMBLING_STAND_INS.items() if name != "npm"}
    result, calls = _run(
        tmp_path, "--build-root", tmp_path / "neutral", "--output", tmp_path / "out",
        "--no-verify", stand_ins=stand_ins,
        env={"STEERLAB_PRIVATE_NAMES_FILE": str(tmp_path / "no-list.txt")})
    assert result.returncode == 5
    assert "npm is not installed" in result.stdout
    assert "Node.js 22.13 or later" in result.stderr
    assert not (tmp_path / "out" / "SteerLab.app").exists()


def test_a_build_signed_for_distribution_requires_the_private_name_list(tmp_path):
    """An ad-hoc build tolerates a machine with no list (the clean-bundle test
    above). A build that can be distributed does not: without the list the
    names cannot be checked, so it is not signed."""
    result, calls, app = _assemble(
        tmp_path, "--identity", "Developer ID Application: Example (TEAMID)")
    assert result.returncode == 7, result.stdout + result.stderr
    assert "private names cannot be checked" in result.stdout
    assert not any(line.startswith("codesign ") for line in calls)
    assert not app.exists()


# -- packaging: the bundle that was built, from a clean tagged commit ----------

#: Stand-ins for `--package`. `codesign -dv` reports the flags in
#: `$SIGNATURE_FLAGS`; `git` answers the three questions the script asks from
#: `$GIT_HEAD`, `$GIT_STATUS`, and `$GIT_TAGS`; `xcrun stapler` says "not
#: stapled". plutil, ditto, and shasum are the real tools.
PACKAGING_STAND_INS = {
    "codesign": r'''
case "$*" in
  *--verify*) exit 0 ;;
  *-dv*) printf 'Executable=%s\nCodeDirectory v=20500 flags=%s\n' "$2" "$SIGNATURE_FLAGS" >&2 ;;
esac
''',
    "git": r'''
case "$*" in
  *"rev-parse --short=8 HEAD"*) [ -n "${GIT_HEAD:-}" ] && printf '%s\n' "$GIT_HEAD" ;;
  *"status --porcelain"*) printf '%s' "${GIT_STATUS:-}" ;;
  *"tag --points-at HEAD"*) printf '%s' "${GIT_TAGS:-}" ;;
esac
''',
    "xcrun": "exit 1\n",
}

DEVELOPER_ID = "0x10000(runtime)"
AD_HOC = "0x10002(adhoc,runtime)"


def _bundle(directory: pathlib.Path, version: str, revision: str, marker: str) -> pathlib.Path:
    app = directory / "SteerLab.app"
    (app / "Contents").mkdir(parents=True)
    (app / "Contents" / "Info.plist").write_text(f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>SLFullVersionString</key><string>{version}</string>
<key>SLSourceRevision</key><string>{revision}</string>
</dict></plist>
""")
    (app / "Contents" / "marker.txt").write_text(marker)
    return app


def _package(tmp_path, *arguments, flags=DEVELOPER_ID, head="abcd1234", status="",
             tags="v9.9.9\n"):
    output = tmp_path / "out"
    result, _ = _run(
        tmp_path, "--package", "--output", output, *arguments,
        stand_ins=PACKAGING_STAND_INS,
        env={"SIGNATURE_FLAGS": flags, "GIT_HEAD": head, "GIT_STATUS": status,
             "GIT_TAGS": tags})
    return result, output / "SteerLab-9.9.9+abcd1234.zip"


def test_package_zips_the_build_output_not_the_installed_copy(tmp_path):
    """It used to prefer ~/SteerLab/SteerLab.app whenever one existed, so the
    zip could hold a different bundle from the one just built and stapled."""
    import hashlib
    import zipfile
    _bundle(tmp_path / "home" / "SteerLab", "0.0.1", "00000000", "the INSTALLED copy")
    _bundle(tmp_path / "out", "9.9.9", "abcd1234", "the bundle this build wrote")
    result, archive = _package(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    with zipfile.ZipFile(archive) as packed:
        assert packed.read("SteerLab.app/Contents/marker.txt") == b"the bundle this build wrote"
    # The checksum file sits beside the zip, in the form `shasum -c` reads.
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert (archive.parent / (archive.name + ".sha256")).read_text() == (
        f"{digest}  {archive.name}\n")
    assert f"{archive}.sha256" in result.stdout


def test_package_never_falls_back_to_the_installed_copy(tmp_path):
    _bundle(tmp_path / "home" / "SteerLab", "9.9.9", "abcd1234", "the INSTALLED copy")
    result, archive = _package(tmp_path)
    assert result.returncode == 5
    assert "The installed copy is never packaged" in result.stderr
    assert not archive.exists()


def test_package_refuses_an_ad_hoc_signature_unless_allowed(tmp_path):
    _bundle(tmp_path / "out", "9.9.9", "abcd1234", "built")
    refused, archive = _package(tmp_path, flags=AD_HOC)
    assert refused.returncode == 6
    assert "ad-hoc signed" in refused.stderr
    assert "--allow-adhoc" in refused.stderr          # the way through, in the refusal
    assert not archive.exists()
    allowed, archive = _package(tmp_path, "--allow-adhoc", flags=AD_HOC)
    assert allowed.returncode == 0, allowed.stdout + allowed.stderr
    assert archive.is_file()
    assert "not a release asset" in allowed.stderr


@pytest.mark.parametrize("state, reason", [
    ({"status": " M scripts/build-app.sh\n"}, "uncommitted or untracked changes"),
    ({"status": "?? notes.txt\n"}, "uncommitted or untracked changes"),
    ({"tags": ""}, "is not tagged v9.9.9 (tags on it: none)"),
    ({"tags": "v9.9.8\n"}, "is not tagged v9.9.9 (tags on it: v9.9.8"),
    ({"head": "ffff0000"}, "the bundle was built from abcd1234, but the checkout is at ffff0000"),
    ({"head": ""}, "no git history"),
])
def test_package_refuses_a_checkout_that_is_not_the_release(tmp_path, state, reason):
    """A release asset has to be traceable to one clean commit, tagged with
    the version the bundle reports, that the bundle was built from."""
    _bundle(tmp_path / "out", "9.9.9", "abcd1234", "built")
    refused, archive = _package(tmp_path, **state)
    assert refused.returncode == 8, refused.stdout + refused.stderr
    assert reason in refused.stderr
    assert "--allow-unreleased" in refused.stderr
    assert not archive.exists()
    assert not (archive.parent / (archive.name + ".sha256")).exists()

    allowed, archive = _package(tmp_path, "--allow-unreleased", **state)
    assert allowed.returncode == 0, allowed.stdout + allowed.stderr
    assert archive.is_file()
    assert reason in allowed.stderr                   # still said, as a note
