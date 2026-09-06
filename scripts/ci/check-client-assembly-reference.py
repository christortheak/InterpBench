#!/usr/bin/env python3
"""Check (or --write) the client assembly table from its declared specs."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Server"))
from steerlab_server import client_cli
from steerlab_server.client.study_assembly import VERB_SPECS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--write", action="store_true")
args = parser.parse_args()
path = ROOT / "docs/CLI-REFERENCE.md"
text = path.read_text()
begin = "<!-- BEGIN CLIENT-STUDY-ASSEMBLY -->"
end = "<!-- END CLIENT-STUDY-ASSEMBLY -->"
start = text.index(begin) + len(begin)
stop = text.index(end, start)
body = "\n```text\n" + "\n".join(client_cli.synopsis(s) for s in VERB_SPECS) + "\n```\n\n"
body += "All commands accept `--root <directory>` and `--json`; `--out` writes the envelope.\n"
if args.write:
    path.write_text(text[:start] + "\n" + body + "\n" + text[stop:])
else:
    assert text[start:stop].strip() == body.strip(), "Run scripts/ci/check-client-assembly-reference.py --write"
    print("Client study assembly reference matches declared flags.")
