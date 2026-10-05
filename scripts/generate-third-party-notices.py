#!/usr/bin/env python3
"""Collect the license texts of everything a built SteerLab.app carries that
SteerLab did not write.

  usage: generate-third-party-notices.py --output FILE
           --package-resolved Package.resolved --checkouts DIR
           --web-bundle DIR --web-node-modules DIR

The repository's NOTICE covers what is VENDORED into the source tree. A built
app carries more: it links the Swift packages statically, so their licenses do
not "travel with them" the way they do in a source checkout, and it embeds a
minified web bundle. Most of those licenses require that their text accompany
a binary. This writes one file with all of them, and `scripts/build-app.sh`
stages it in the bundle beside LICENSE and NOTICE.

Two inputs, both read from what the build ACTUALLY used:

* Swift packages — every pin in `Package.resolved`, with the license,
  notice, and acknowledgment files found in its checkout under
  `<derived data>/SourcePackages/checkouts` (including the components a
  package vendors itself, such as the C++ libraries inside mlx-swift).
* The web bundle — the npm packages listed in `bundled-packages.json`, which
  the embedded build writes into the bundle from the modules it really
  included (results-explorer/vite.embed.config.ts), with each package's
  license from `node_modules`.

It REFUSES rather than omits: a pinned package with no checkout, or a package
with no license file, is an error (exit 1) — a notices file that silently
left something out would be worse than none.

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

#: File names that carry license terms or required attributions.
LICENSE_NAME = re.compile(r"^(LICEN[CS]E|COPYING|NOTICE|ACKNOWLEDG(E)?MENTS?)([.-].*)?$", re.I)
#: Directories that hold a package's own tests, documentation, and tooling —
#: not code that is compiled into the app.
PRUNED = {".git", ".github", ".build", "tests", "test", "docs", "documentation",
          "examples", "example", "benchmarks", "node_modules", "tools", "scripts",
          "python", "fuzz"}
#: How far below a checkout's root to look for the components it vendors.
MAX_DEPTH = 5
RULE = "=" * 78


def fail(message: str) -> "None":
    print(f"generate-third-party-notices.py: {message}", file=sys.stderr)
    raise SystemExit(1)


def license_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for directory, subdirectories, files in os.walk(root):
        depth = len(Path(directory).relative_to(root).parts)
        subdirectories[:] = sorted(
            name for name in subdirectories
            if name.lower() not in PRUNED and depth < MAX_DEPTH)
        found += [Path(directory) / name for name in sorted(files)
                  if LICENSE_NAME.match(name)]
    # The package's own terms first, then what it vendors.
    return sorted(found, key=lambda path: (len(path.relative_to(root).parts), str(path)))


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip("\n")


def swift_sections(resolved: Path, checkouts: Path) -> list[str]:
    pins = json.loads(resolved.read_text(encoding="utf-8"))["pins"]
    by_name = {entry.name.lower(): entry for entry in checkouts.iterdir() if entry.is_dir()}
    sections = []
    for pin in sorted(pins, key=lambda pin: pin["identity"]):
        location = pin["location"]
        candidates = (pin["identity"], location.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git"))
        checkout = next((by_name[name.lower()] for name in candidates
                         if name.lower() in by_name), None)
        if checkout is None:
            fail(f"no checkout for the pinned package '{pin['identity']}' under {checkouts} "
                 "— build the app into this derived-data directory first")
        files = license_files(checkout)
        if not any(path.parent == checkout for path in files):
            fail(f"the package '{pin['identity']}' has no license file at the top of {checkout}")
        state = pin["state"]
        version = state.get("version") or state.get("branch") or "unversioned"
        header = [RULE, f"{pin['identity']} {version}", location,
                  f"revision {state.get('revision', 'unknown')}", RULE]
        body = []
        for path in files:
            body += ["", f"--- {path.relative_to(checkout).as_posix()} ---", "", read(path)]
        sections.append("\n".join(header + body))
    return sections


def web_sections(bundle: Path, node_modules: Path) -> list[str]:
    listing = bundle / "bundled-packages.json"
    if not listing.is_file():
        fail(f"{listing} is missing — the embedded build writes it; rebuild the results explorer")
    sections = []
    names = sorted(json.loads(listing.read_text(encoding="utf-8")))
    if names and not node_modules.is_dir():
        fail(f"the web bundle contains npm packages, but {node_modules} does not exist — run npm ci")
    for name in names:
        package = node_modules / name
        if not (package / "package.json").is_file():
            fail(f"the web bundle contains '{name}', but {package} is not installed — run npm ci")
        manifest = json.loads((package / "package.json").read_text(encoding="utf-8"))
        files = [path for path in sorted(package.iterdir())
                 if path.is_file() and LICENSE_NAME.match(path.name)]
        if not files:
            fail(f"the npm package '{name}' has no license file in {package}")
        header = [RULE, f"{manifest['name']} {manifest['version']} (npm)",
                  f"declared license: {manifest.get('license', 'not declared')}", RULE]
        body = []
        for path in files:
            body += ["", f"--- {path.name} ---", "", read(path)]
        sections.append("\n".join(header + body))
    return sections


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--package-resolved", required=True, type=Path)
    parser.add_argument("--checkouts", required=True, type=Path,
                        help="<derived data>/SourcePackages/checkouts")
    parser.add_argument("--web-bundle", required=True, type=Path,
                        help="the built results explorer (holds bundled-packages.json)")
    parser.add_argument("--web-node-modules", required=True, type=Path)
    args = parser.parse_args()
    for path in (args.package_resolved, args.checkouts, args.web_bundle):
        if not path.exists():
            fail(f"{path} does not exist")

    swift = swift_sections(args.package_resolved, args.checkouts)
    web = web_sections(args.web_bundle, args.web_node_modules)
    text = "\n".join([
        "THIRD-PARTY NOTICES",
        "",
        "SteerLab itself is licensed under the Apache License, Version 2.0 (see",
        "LICENSE and NOTICE beside this file). This build also contains the",
        "software listed below, each under its own license. The texts are",
        "reproduced as their authors distribute them.",
        "",
        f"Part 1 — Swift packages linked into the app and its command-line tool ({len(swift)})",
        "",
        "\n\n".join(swift),
        "",
        f"Part 2 — JavaScript packages in the embedded Results Explorer ({len(web)})",
        "",
        "\n\n".join(web),
        "",
    ])
    args.output.write_text(text, encoding="utf-8")
    print(f"third-party notices: {len(swift)} Swift package(s), {len(web)} npm package(s) "
          f"-> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
