"""Discover exact diagnostic input dependencies and package execution copies."""
from pathlib import Path
import json
from . import diagnostic_archives as archives

#: A JSON input larger than this is not read for custom code: agents and
#: policies are small, and a large JSON file is data (a corpus, a lens table).
_CODE_SCAN_BYTES = 16 * 1024 * 1024


class CustomCodeRefusal(archives.Refusal):
    """Packaging declined: the inputs carry custom code nobody acknowledged."""

    code = 'missingPrerequisite'

    def __init__(self, message, repair_action):
        super().__init__(message)
        self.repair_action = repair_action


def _carried_code(files, root):
    """Every expert provider in the plan's JSON input files, as
    ``(providers, sources)`` in :func:`custom_code.providers` shape: the code
    an execution copy of these inputs would run."""
    from . import custom_code
    root = Path(root).resolve()
    names, sources = {}, {}
    for entry in files:
        if not entry['path'].endswith('.json') or entry.get('bytes', 0) > _CODE_SCAN_BYTES:
            continue
        try:
            document = json.loads((root / entry['path']).read_bytes())
        except (OSError, ValueError):
            continue
        for provider in custom_code.providers(document):
            names.setdefault(provider['sha256'], set()).update(provider['policyNames'])
        sources.update(custom_code.provider_sources(document))
    return [{'sha256': digest, 'policyNames': sorted(names[digest])} for digest in sorted(names)], sources


def _custom_code_block(reviewed, root):
    """The ``customCode`` block an input plan carries when its files hold
    custom code, or None. Outside the plan digest: acknowledging the code
    changes this block, not the inputs the digest pins."""
    from . import custom_code
    carried, sources = _carried_code(reviewed['files'], root)
    if not carried:
        return None
    try:
        seen = custom_code.acknowledged(root)
    except custom_code.CustomCodeError as exc:   # a damaged record is refused, never read as empty
        raise CustomCodeRefusal(str(exc), exc.repair_action) from exc
    rows = [{**row, 'sourceText': sources[row['sha256']], 'acknowledged': row['sha256'] in seen,
             'acknowledgedAt': seen.get(row['sha256'], {}).get('acknowledgedAt'),
             'acknowledgedBy': seen.get(row['sha256'], {}).get('acknowledgedBy')} for row in carried]
    pending = [row['sha256'] for row in rows if not row['acknowledged']]
    return {'notice': custom_code.DIAGNOSTIC_NOTICE if pending else None, 'providers': rows,
            'acknowledged': not pending, 'recordFile': custom_code.FILENAME,
            'acknowledgeFlag': '--custom-code-sha256 ' + ','.join(pending) if pending else None}


def plan(request, root):
    """The reviewed input plan: every file an execution copy carries, hashed.
    When those files hold custom code (a policy's expert provider), the plan
    also carries the notice and the code itself under ``customCode``."""
    reviewed = _plan(request, root)
    block = _custom_code_block(reviewed, root)
    return {**reviewed, 'customCode': block} if block else reviewed


def _plan(request, root):
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
    # Battery and stability closures are only ever planned to travel (package,
    # stage, verify a staged copy), so the transport bound applies at review.
    result = {'schemaVersion': 1, 'request': request, 'sourceRoot': str(root), 'files': archives.snapshot(root, files)}
    return {**result, 'planSHA256': archives.digest(result)}


def package(request, root, destination, expected, acknowledge=None):
    """Package the reviewed inputs. Inputs that carry custom code are packaged
    only once every piece of it is acknowledged in this workspace: earlier, or
    now, by naming each SHA-256 the plan shows in ``acknowledge`` (the
    comma-separated value of ``--custom-code-sha256``)."""
    from . import custom_code
    reviewed = plan(request, root)
    if reviewed['planSHA256'] != expected: raise archives.Refusal('Input source plan changed; re-review before packaging.')
    block = reviewed.get('customCode')
    named = [part.strip() for part in (acknowledge or '').split(',') if part.strip()]
    if named and not block:
        raise archives.Refusal('These inputs carry no custom code, so there is nothing to acknowledge; package without --custom-code-sha256.')
    if block and not block['acknowledged']:
        carried = [row['sha256'] for row in block['providers']]
        pending = [row for row in block['providers'] if not row['acknowledged']]
        unknown = [digest for digest in named if digest not in carried]
        if unknown or any(row['sha256'] not in named for row in pending):
            listed = '; '.join(f"SHA-256 {row['sha256']} (policy: {', '.join(row['policyNames']) or 'unnamed'})"
                               for row in pending)
            raise CustomCodeRefusal(
                ('The inputs carry no custom code with SHA-256 ' + ', '.join(unknown) + '. ' if unknown else '')
                + f'The diagnostic was not packaged: its inputs contain custom code nobody has acknowledged '
                f'in this workspace ({listed}). {custom_code.DIAGNOSTIC_NOTICE}',
                'Read the code under customCode.providers in `science input-plan <request.json>`. If you trust '
                'the source, repeat `science package` with ' + block['acknowledgeFlag'] + '.')
        custom_code.acknowledge_inputs(root, block['providers'], [row['sha256'] for row in pending],
                                       operation=reviewed['request'].get('operation', ''))
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
            'nextAction': 'Pass localRequestPath as the positional request file to science-plan, then science-submit with the reviewed planSHA256 on the same controller.'}
