"""Discover exact diagnostic input dependencies and package execution copies."""
from pathlib import Path
import json
from . import diagnostic_archives as archives


def plan(request, root):
    from ..api.scientific_execution import request_document
    from . import paths
    request = request_document(request)
    from . import managed_methods, managed_inputs
    if request['operation'] in managed_methods.OPERATIONS:
        return managed_inputs.plan(request, root)
    root = Path(root).resolve(); p = request['parameters']; files = set()
    def add(path):
        path = Path(path)
        relative = path.relative_to(root).as_posix() if path.is_absolute() else path.as_posix()
        files.update(archives.files_in(root, relative))
    if request['operation'] == 'battery':
        from . import battery_run, model_variant
        add(p['batteryFile'])
        agents = battery_run.resolve_agents(battery_run.parse_agents(p['agents']), root=str(root),
            model_id=p.get('modelID'), revision=p.get('revision'), alpha_units=p['alphaUnits'])
        for agent in agents:
            identity = agent.identity
            if identity.get('artifactPath'): add(identity['artifactPath'])
            references = [identity.get('vectorArtifactID')] + [i.get('vectorArtifactID') for i in identity.get('injections', [])]
            for reference in filter(None, references):
                archives.parts(reference)  # Portable copies must never resolve back to an absolute source.
                add(paths.resolve_artifact(reference, str(root)) + '.json')
                add(paths.resolve_artifact(reference, str(root)) + '.safetensors')
            if agent.variant:
                for adapter in agent.variant.adapters:
                    reference = adapter.get('adapterDirectory') or adapter.get('artifactPath') or ''
                    archives.parts(reference)
                    directory = paths.resolve_artifact(reference, str(root))
                    add(directory)
                    sidecar = Path(model_variant.adapter_sidecar_path(directory))
                    if sidecar.exists() or sidecar.is_symlink(): add(sidecar)
                if agent.variant.neutral_pc_basis_path:
                    archives.parts(agent.variant.neutral_pc_basis_path)
                    add(paths.resolve_artifact(agent.variant.neutral_pc_basis_path, str(root)))
    else:
        from . import extract_stability, experiment_store, multiconcept
        prepared = extract_stability.preflight(p['experiment'], p['concept'], root=str(root),
            resamples=p['resamples'], fraction=p['fraction'])
        document = experiment_store.load_raw(p['experiment'], str(root))
        add(document.source_path)
        ref = prepared['ref']
        if prepared['method'].is_designated_reference:
            add(multiconcept.stories_path(ref.name, str(root)))
            add(multiconcept.stories_path(ref.designated_reference['name'], str(root)))
        else:
            add(paths.concept_directory(ref.name, str(root)))
    result = {'schemaVersion': 1, 'request': request, 'sourceRoot': str(root), 'files': archives.snapshot(root, files)}
    return {**result, 'planSHA256': archives.digest(result)}


def package(request, root, destination, expected):
    reviewed = plan(request, root)
    if reviewed['planSHA256'] != expected: raise archives.Refusal('Input source plan changed; re-review before packaging.')
    context = {'request': reviewed['request'], 'sourcePlanSHA256': expected, 'sourceRoot': reviewed['sourceRoot']}
    return archives.package(root, [e['path'] for e in reviewed['files']], destination,
        kind='diagnosticInput', context=context, expected_entries=reviewed['files'])


def save_stage_reference(digest, root):
    """A local planning input naming the verified remote execution copy."""
    import re
    import tempfile
    if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
        raise archives.Refusal('Use the inputBundleSHA256 returned by staging.')
    root = Path(root).resolve(strict=True)
    directory = archives.ordinary(root, '.steerlab/diagnostic-requests', missing=True)
    directory.mkdir(parents=True, exist_ok=True)
    destination = archives.ordinary(directory, digest + '.json', missing=True)
    data = archives.encoded({'inputBundleSHA256': digest})
    with tempfile.TemporaryDirectory(dir=directory) as temp:
        source = Path(temp) / 'request.json'; source.write_bytes(data)
        try: archives.publish_file(source, destination)
        except FileExistsError:
            if destination.read_bytes() != data:
                raise archives.Refusal('The saved staged request differs; inspect it before planning.')
    return {'localRequestPath': str(destination), 'request': {'inputBundleSHA256': digest},
            'nextAction': 'Use localRequestPath with science-plan --request, then science-submit --request and the reviewed planSHA256 on the same controller.'}
