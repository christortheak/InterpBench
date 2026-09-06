"""Immutable semantic inputs and reviewed panel compilation for local authors."""
from pathlib import Path

from . import authoring_files as files, design_files, design_panels, panel_documents
from ..experiment import experiment_store as store, manifest_files


def validate(document: dict) -> dict:
    from ..experiment.multi_agent import Scenario, ScenarioError, validate as validate_scenario
    panel = panel_documents.normalized(document)
    if not panel['agents'] or len({a['id'] for a in panel['agents']}) != len(panel['agents']):
        files.refuse('A panel needs unique named seats.')
    if design_panels.semantic(panel) != panel:
        files.refuse('Author an unbound semantic panel; model settings and seat agents belong to a study.')
    # Semantic validation uses the execution validator with explicit throwaway
    # bindings. No rendered scientific input or model is loaded here.
    bound = design_panels.semantic(panel)
    bound['baseModelID'] = 'validation/model'
    for seat in bound['agents']:
        seat['baseModelID'] = 'validation/model'
    try:
        validate_scenario(Scenario.from_dict(bound))
    except (ScenarioError, ValueError, KeyError, TypeError) as exc:
        files.refuse(f'Invalid panel: {exc}')
    return panel


def inspect(path: str, *, root: Path) -> dict:
    if not path.startswith('prompts/panels/'):
        files.refuse('Select a workspace panel under prompts/panels/.')
    data = design_files.ordinary(root, path).read_bytes()
    digest = manifest_files.digest_bytes(data)
    panel = design_panels.load({'path': path, 'hash': digest}, root)
    return {'path': path, 'fileSHA256': digest, 'document': panel,
            'semantic': design_panels.semantic(panel) == panel,
            'seatIDs': [a['id'] for a in panel['agents']]}


def catalog(*, root: Path) -> dict:
    directory = design_files.ordinary(root, 'prompts/panels')
    entries, issues = [], []
    for path in sorted(directory.glob('*.json')) if directory.exists() else []:
        try:
            entries.append(inspect(str(path.relative_to(root)), root=root))
        except (OSError, ValueError, store.ExperimentStoreError) as exc:
            issues.append(f'Could not inspect {path.name}: {exc}')
    return {'panels': entries, 'issues': issues}


def publish(data: bytes, *, root: Path, expected: str) -> dict:
    if manifest_files.digest_bytes(data) != expected:
        files.refuse('The proposed panel changed after review.', gate='artifactPin')
    panel = validate(design_files.decode(data))
    ref, changed = design_panels.publish_pin(panel, root)
    return {**inspect(ref['path'], root=root), 'changed': changed}


def compile(name: str, path: str, casting: dict, *, root: Path, expected: str, file_sha256: str) -> dict:
    design_files.ordinary(root, str(files.study_path(root, name).relative_to(root)))
    with files.reviewed_draft(name, root, expected):
        review = inspect(path, root=root)
        if review['fileSHA256'] != file_sha256:
            files.refuse('The selected panel changed after review.', gate='artifactPin')
        validate(review['document'])
        if not isinstance(casting, dict) or set(casting) != {'seats'}:
            files.refuse('Panel compilation needs explicit seats, including null for every baseline.')
        document = store.load_raw(name, str(root))
        template = {'study': dict(document), 'semanticScenario': {'path': path, 'hash': file_sha256}}
        bound = design_panels.cast(template, casting, document, root)
        ref = design_panels.pin(bound, root, compiled=True)
        document.update(studyKind='multiAgent', studyType='multiAgent', multiAgentScenarioPath=ref['path'], multiAgentScenarioHash=ref['hash'],
                        multiAgentSemanticScenarioPath=path, multiAgentSemanticScenarioHash=file_sha256)
        store.save_raw(document, str(root))
        return {**files.snapshot(name, root), 'changed': document.source_digest != expected}
