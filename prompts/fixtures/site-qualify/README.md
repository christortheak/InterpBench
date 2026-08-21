# Site-qualification fixtures

Read by `steerlab-server site qualify` — the one command that says whether a
freshly deployed node reproduces the committed structural contracts.

## `stimulus/`

A deliberately tiny contrastive stimulus set (`positive.jsonl` +
`negative.jsonl`, four neutral lines each) and `expected-hash.txt`, the SHA-256
these two files hash to under the pinned convention: the raw bytes of
`positive.jsonl` followed by the raw bytes of `negative.jsonl`
(`Server/steerlab_server/steering/stimulus_set.py`, Swift twin
`StimulusSet.swift`). That digest is what the circularity firewall pins a
concept by, and it is identical on both engines — so a node that computes a
different one has a filesystem, line-ending, or text-encoding problem that
would silently unpin every frozen experiment run there.

The sentences carry no concept: this fixture exists to exercise the hashing
contract, never to extract anything.

Regenerate the digest ONLY after deliberately changing the stimulus bytes:

```bash
Server/.venv.nosync/bin/python - <<'PY'
from steerlab_server.steering.stimulus_set import StimulusSet
s = StimulusSet.from_directory("prompts/fixtures/site-qualify/stimulus")
open("prompts/fixtures/site-qualify/stimulus/expected-hash.txt", "w").write(s.hash + "\n")
PY
```

`Server/tests/test_site_qualify.py` recomputes the digest from the committed
bytes on every run, so a stale `expected-hash.txt` fails loudly here before it
can pass falsely on a remote node.
