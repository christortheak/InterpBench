#!/usr/bin/env python3
"""Check (or --write) the client assembly table from its declared specs."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Server"))
from steerlab_server import client_cli
from steerlab_server.client.study_assembly import VERB_SPECS as ASSEMBLY_SPECS
from steerlab_server.client.design_commands import VERB_SPECS as DESIGN_SPECS
from steerlab_server.client.authoring_commands import VERB_SPECS as AUTHORING_SPECS
MODEL_SPECS = tuple(s for s in client_cli.CLIENT_VERB_SPECS if s.family == "model" and s.verb in ("plan", "install", "status", "cancel"))
from steerlab_server.client.science_commands import VERB_SPECS as SCIENCE_SPECS
REMOTE_SPECS = tuple(s for s in client_cli.CLIENT_VERB_SPECS if s.family == "runner" and s.verb in ("science-stage", "science-export", "science-fetch", "science-call", "cleanup-plan", "cleanup-apply", "science-plan", "science-submit", "recovery", "recover", "resubmit", "reconcile"))
from steerlab_server.client.bootstrap_commands import VERB_SPECS as BOOTSTRAP_SPECS
from steerlab_server.client.setup_commands import VERB_SPECS as SETUP_SPECS
VERB_SPECS = (*SETUP_SPECS, *BOOTSTRAP_SPECS, *REMOTE_SPECS, *SCIENCE_SPECS, *ASSEMBLY_SPECS, *DESIGN_SPECS, *AUTHORING_SPECS, *MODEL_SPECS)

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
body += "All commands accept `--json`; `--out` writes the envelope. Workspace init takes its destination positionally; other commands accept `--root <directory>`.\n"
if args.write:
    path.write_text(text[:start] + "\n" + body + "\n" + text[stop:])
else:
    assert text[start:stop].strip() == body.strip(), "Run scripts/ci/check-client-assembly-reference.py --write"
    print("Client study assembly reference matches declared flags.")

from science_cli_census import check_catalog
import json
check_catalog(json.loads((Path(__file__).resolve().parents[2] / "WorkspaceSeed/prompts/method-guides/catalog.json").read_text()), (Path(__file__).resolve().parents[2] / "Server/steerlab_server/cli.py").read_text())

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'Server'))
from steerlab_server.api.route_roles import CENSUS
from science_cli_census import check_actions
check_actions(json.loads((Path(__file__).resolve().parents[2] / 'WorkspaceSeed/prompts/method-guides/catalog.json').read_text()), CENSUS)
