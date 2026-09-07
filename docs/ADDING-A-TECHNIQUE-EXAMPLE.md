# Disposable worked example: count declared input records

Companion to [ADDING-A-TECHNIQUE.md](ADDING-A-TECHNIQUE.md). This fictional CPU
operation counts nonblank JSON-object rows. It is a bookkeeping example, not a
new scientific measurement. Its claim is only the number of parsed input rows;
it establishes no dataset quality, independence or intervention efficacy.

Make the following changes only in a disposable checkout. Use a workspace under
temporary storage for fixture data. Keep the example out of the production
registry. A full release acceptance still needs both suites and an independent
agent trial; the small test here verifies the example's integration mechanics.

To reproduce the mechanical exercise without changing this checkout, run
`python scripts/ci/qualify-technique-example.py` with the test-capable environment.
It extracts the code blocks below, applies the stated registrations in temporary
storage, regenerates resources, runs the real-owner tests, and removes its copy.
It installs nothing and is not a substitute for an independent agent trial.

## Owner addition

Create `Server/steerlab_server/experiment/example_row_count.py`:

```python
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import uuid
from . import diagnostic_archives as archives


@dataclass(frozen=True)
class RowCountConfig:
    itemsFile: str

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {'itemsFile'}:
            raise archives.Refusal('Supply exactly itemsFile: a workspace-relative JSONL file.')
        if not isinstance(value['itemsFile'], str):
            raise archives.Refusal('itemsFile must be a workspace-relative path string.')
        archives.parts(value['itemsFile'])
        return cls(value['itemsFile'])

    def to_dict(self):
        return {'itemsFile': self.itemsFile}


def count(config, root, log=print):
    root = Path(root).resolve()
    source = archives.ordinary(root, config.itemsFile)
    data = source.read_bytes()
    rows = 0
    for index, line in enumerate(data.decode('utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError as exc:
            raise archives.Refusal(f'Row {index} is not JSON; repair that row in a new input version.') from exc
        if not isinstance(item, dict):
            raise archives.Refusal(f'Row {index} must be a JSON object; inspect the selected file.')
        rows += 1
    report = {
        'schemaVersion': 1, 'operation': 'example-row-count',
        'config': config.to_dict(), 'rowCount': rows,
        'inputSHA256': hashlib.sha256(data).hexdigest(),
        'claimBoundary': 'Parsed row count only; no scientific qualification.',
    }
    runs = archives.ordinary(root, 'runs', missing=True)
    runs.mkdir(exist_ok=True)
    target = runs / ('example-row-count-' + uuid.uuid4().hex)
    with tempfile.TemporaryDirectory(dir=runs) as staging:
        output = Path(staging) / 'output'
        output.mkdir()
        (output / 'report.json').write_bytes(archives.encoded(report))
        archives.publish_directory(output, target)
    log(f'Counted {rows} rows.')
    return {'runDirectory': str(target), 'rowCount': rows}
```

The report records the hash of the bytes actually read. Managed packaging and
queue-time verification additionally bind the declared request to its input
snapshot. It publishes a new `runs/<id>` directory, which the current export
adapter recognizes. It never changes a prior output.

## Registration and input closure

In `experiment/managed_methods.py`, add this entry to `METHODS`:

```python
'example-row-count': Method('example_row_count', 'RowCountConfig', 'count', 'cpu'),
```

No `managed_inputs.py` edit is needed: `itemsFile` is already in `FILES`.
For any other input key, inspect and test its closure explicitly. The config
is CPU-only and contains no `modelID`; it needs no model download or CUDA/MLX.

## Interview and catalog additions

Append this entry to `workflows.json`'s `operations`:

```json
{
  "id": "example-row-count",
  "method": "stability",
  "title": "Count input records (example)",
  "purpose": "Inspect the number of declared JSONL records.",
  "claimBoundary": "Counts parsed rows only; does not assess scientific quality.",
  "questions": ["Which file?", "What will the count be used for?", "What does it not establish?"],
  "fields": [
    {"id": "itemsFile", "label": "Input JSONL", "kind": "file", "required": true,
     "help": "Choose a workspace-relative JSONL file containing one object per row."}
  ]
}
```

For a concrete catalog delta, copy the existing `optvec-family` operation in
`catalog.json`, then replace these fields; keep its existing science plan/submit
actions and access envelope unchanged:

```json
{
  "id": "example-row-count",
  "method": "stability",
  "title": "Count input records (example)",
  "engineCLI": null,
  "compute": "cpu",
  "mac": "Research methods: Author request, then execution and evidence.",
  "outputs": ["A new run report containing rowCount and inputSHA256."],
  "restriction": "CPU bookkeeping only. This count is not scientific qualification."
}
```

This is a set of field replacements, not a complete second schema. `engineCLI`
is null because no engine-specific parser verb is added. The existing managed
HTTP routes provide execution. Add a short explanation to the stability guide.
The category is reused only for this disposable integration example.

Add `"example-row-count": "python-cpu"` to the `operations` object in
`docs/substrate-capabilities.json`. Run the guide's regeneration commands.

## Real-owner test addition

Add this test as `Server/tests/test_example_row_count.py`:

```python
import json
from pathlib import Path
import pytest
from steerlab_server.experiment import method_authoring, managed_methods
from steerlab_server.experiment import managed_inputs, diagnostic_archives as archives
from steerlab_server.api import managed_validation


def test_draft_validates_executes_and_preserves_prior_output(tmp_path):
    (tmp_path / 'items.jsonl').write_text('{"id":"a"}\n\n{"id":"b"}\n')
    answers = dict(purpose='Count rows', claim='Two parsed rows',
                   controls='Hand-counted fixture', selection='Declared fixture',
                   fields={'itemsFile': 'items.jsonl'}, advanced={})
    draft = method_authoring.draft('example-row-count', answers, tmp_path)
    request = draft['request']
    config = request['parameters']['config']
    assert managed_methods.validate('example-row-count', config, tmp_path).to_dict() == config
    assert managed_validation.validate(request, tmp_path)['effectiveConfig'] == config
    published = method_authoring.publish('example-row-count', answers, tmp_path,
                                        'requests/example', draft['planSHA256'])
    assert json.loads(Path(published['requestFile']).read_text()) == request
    result = managed_methods.execute('example-row-count', config, tmp_path)
    report_path = Path(result['runDirectory']) / 'report.json'
    before = report_path.read_bytes()
    assert json.loads(before)['rowCount'] == 2
    second = managed_methods.execute('example-row-count', config, tmp_path)
    assert second['runDirectory'] != result['runDirectory']
    assert report_path.read_bytes() == before
    pin = managed_inputs.plan(request, tmp_path)['planSHA256']
    (tmp_path / 'items.jsonl').write_text('{"id":"c"}\n')
    assert managed_inputs.plan(request, tmp_path)['planSHA256'] != pin
    with pytest.raises(archives.Refusal, match='changed'):
        method_authoring.publish('example-row-count', answers, tmp_path,
                                 'requests/changed', draft['planSHA256'])


def test_unknown_config_and_malformed_rows_explain_the_problem(tmp_path):
    with pytest.raises(archives.Refusal, match='itemsFile'):
        managed_methods.validate('example-row-count', {'typo': 'items.jsonl'}, tmp_path)
    (tmp_path / 'items.jsonl').write_text('[]\n')
    with pytest.raises(archives.Refusal, match='Row 1 must be a JSON object'):
        managed_methods.execute('example-row-count', {'itemsFile': 'items.jsonl'}, tmp_path)
```

Also add `example-row-count` to `OPERATIONS` in
`Server/tests/test_interview_validation.py` and a fixture branch setting
`fields['itemsFile'] = 'items.jsonl'`. Its existing helper already creates that
file. This ensures the full registry census and real-owner interview test cover
the new operation; updating a single isolated example test is insufficient.

Run `PYTHONPATH=Server python -m pytest Server/tests/test_example_row_count.py
Server/tests/test_interview_validation.py -q` as one command. Then the shared
resource checks and the actual science input package/stage/execute/export/import
tests. The snippet tests publication and direct execution, not queue scheduling
or HTTP custody. The guide's acceptance journey deliberately checks those too.

Inspect the entire delta, including generated files, and run both suites before
any real technique lands. Discard the disposable checkout after the exercise;
the real product registry must not gain this example operation.
