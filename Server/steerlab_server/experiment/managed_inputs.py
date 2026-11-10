"""Portable dependency closure for managed scientific requests (no GPU imports)."""
from . import input_hashes
import json
from pathlib import Path
from . import diagnostic_archives as archives, managed_methods, paths

from .operation_bindings import INPUT_ROLES

#: How a lens's converted file enters a request's closure (`lens_reference`).
#: A lens is a record plus one converted file of several gigabytes; the record
#: (and its import receipt) is always a listed file. Public issues #13, #14.
#:
#: CARRIED: the request carries the converted file, listed and hashed with the
#: rest. Unchanged from before a site could keep lenses.
CARRIED = False
#: REFERRED: packaged with `--lens-reference` on the researcher's machine. The
#: workspace's converted file is checked against the record's hash exactly as
#: a carried one is, then left out and named in `libraryLenses`, so a
#: reference is only ever made to bytes that were checked.
REFERRED = True
#: WHERE_HELD: where a request runs. A converted file present in the workspace
#: is carried, exactly as above. One that is absent is taken from the
#: engine's lens library when that holds the record's hash (an existence and
#: size check here; `verify_library` hashes it, once, on the queued job), and
#: is otherwise the same missing input it always was.
WHERE_HELD = 'whereHeld'


def inventory(operation, config, root, *, lens_reference=CARRIED, referred=None, verify_library=False):
    root = Path(root).resolve(); files = set()
    referred = {} if referred is None else referred
    roles = INPUT_ROLES[operation]
    def add(reference, artifact=False):
        archives.parts(reference)
        candidates = [reference + '.json', reference + '.safetensors'] if artifact else [reference]
        for candidate in candidates: files.update(archives.files_in(root, candidate))
    def add_lens(value):
        if not isinstance(value, str): raise archives.Refusal('Lens IDs must be single components.')
        archives.parts(value)
        if '/' in value: raise archives.Refusal('Lens IDs must be single components.')
        directory = Path(paths.jlens_lens_directory(value, str(root)))
        # Ship the record, its import receipt, and the converted tensor the
        # engine reads; not the `source/` provenance copy of the original
        # bytes, which doubles a multi-gigabyte lens and is never read at
        # execution. The receipt still names the source hashes.
        lens_dir = directory.relative_to(root).as_posix()
        add(lens_dir + '/lens.json')
        if archives.ordinary(root, lens_dir + '/import-receipt.json', missing=True).is_file():
            add(lens_dir + '/import-receipt.json')
        record = json.loads((directory / 'lens.json').read_bytes())
        from ..jlens import artifact_paths
        converted = record.get('converted') or {}
        # The workspace's own file only: the site's lens library is consulted
        # below, by name, and never stands in for a carried file.
        selected = Path(artifact_paths.converted_file(value, converted['path'], str(root), library=False))
        if lens_reference == WHERE_HELD and not (selected.is_relative_to(root) and selected.is_file()):
            from ..jlens import lens_library
            held = lens_library.reference(value, converted.get('sha256'), verify=verify_library)
            if held is not None:
                referred[value] = held
                return
        if not selected.is_relative_to(root): raise archives.Refusal('Managed lenses require converted tensors in the workspace lens library.')
        relative = selected.relative_to(root).as_posix()
        if lens_reference is REFERRED:
            # The checks a carried file passes (an ordinary file in this
            # workspace whose bytes are the record's), without listing it.
            archives.parts(relative)
            if archives.files_in(root, relative) != [relative]:
                raise archives.Refusal('A lens\'s converted tensors must be one file: ' + relative)
            checked = archives.ordinary(root, relative)
            if archives.file_hash(checked) != converted.get('sha256'):
                raise archives.Refusal('Converted lens bytes differ from the imported hash; re-import or restore the lens.')
            referred[value] = {'lensID': value, 'convertedSHA256': converted['sha256'], 'bytes': checked.stat().st_size}
            return
        add(relative)
        if archives.file_hash(archives.ordinary(root, relative)) != converted.get('sha256'):
            raise archives.Refusal('Converted lens bytes differ from the imported hash; re-import or restore the lens.')
    def walk(value, key=''):
        if value is None: return
        if roles.get(key) == 'lens': add_lens(value)
        elif roles.get(key) == 'lenses' and isinstance(value, list):
            # Each distinct lens is inventoried once; `files` is a set.
            for item in value: add_lens(item)
        elif roles.get(key) == 'artifact' and isinstance(value, str): add(value, artifact=True)
        elif roles.get(key) == 'file' and isinstance(value, str):
            add(value)
            if key == 'gradients':
                for candidate in (str(Path(value).with_suffix('.json')), str(Path(value).parent / 'gradients.json')):
                    if archives.ordinary(root, candidate, missing=True).exists(): add(candidate)
        elif roles.get(key) == 'artifacts' and isinstance(value, list):
            for item in value:
                if isinstance(item, str): add(item, artifact=True)
                else: walk(item)
        elif roles.get(key) == 'trees' and isinstance(value, list):
            for item in value: add(item)
        elif isinstance(value, dict):
            if isinstance(value.get('path'), str) and value.get('sha256'):
                if archives.file_hash(archives.ordinary(root, value['path'])) != value['sha256']:
                    raise archives.Refusal('Declared data hash differs from the selected file: ' + value['path'])
            for name, item in value.items(): walk(item, name)
        elif isinstance(value, list):
            for item in value: walk(item)
    walk(config)
    # A nested closure is planned the way its parent is, into the same record
    # of lenses referred to.
    nested_mode = {'lens_reference': lens_reference, 'referred': referred, 'verify_library': verify_library}
    if operation == 'jlens-fit-round':
        from .jlens_round import RoundConfig, prepare
        for child in prepare(RoundConfig.from_dict(config), root)['shards']:
            files.update(e['path'] for e in inventory('jlens-fit', child['parameters']['config'], root, **nested_mode))
    if operation == 'jlens-fit-benchmark':
        from .jlens_benchmark import BenchmarkConfig, fitting_config
        nested = fitting_config(BenchmarkConfig.from_dict(config), root).to_dict()
        files.update(e['path'] for e in inventory('jlens-fit', nested, root, **nested_mode))
    if operation == 'jlens-fit':
        from .jlens_fit import checkpoint_files
        for relative in checkpoint_files(config, root): add(relative)
    if operation == 'jlens-probe-batch':
        # The request names only the agent JSON; the probe also reads what it
        # references (vectors, adapter, neutral basis), so those are pinned too.
        from .jlens_probe_batch import agent_files
        for relative in agent_files(config, root): add(relative)
    if operation == 'rescore-style':
        from . import experiment_store
        document = experiment_store.load_raw(config.get('experiment', ''), str(root))
        add(Path(document.source_path).relative_to(root).as_posix())
        taxonomy = document.get('reasoningStyleTaxonomyPath')
        if not taxonomy: raise archives.Refusal('Pin a taxonomy before authoring a style rescore.')
        add(taxonomy)
        # Verification can inspect declared prompt inputs; include the authored
        # prompt tree, displayed in review, without copying unrelated runs.
        if (root / 'prompts').exists(): add('prompts')
    if operation == 'sae-family-report':
        # The report discovers qualification pointers beside each vector; those
        # optional files influence its evidence columns and belong in review.
        for entry in config.get('artifacts', []):
            if isinstance(entry, dict) and isinstance(entry.get('reference'), str):
                reference = Path(entry['reference'])
                for name in (reference.name + '-sae-feature-qualification.json', 'sae-feature-qualification.json'):
                    candidate = (reference.parent / name).as_posix()
                    if archives.ordinary(root, candidate, missing=True).exists(): add(candidate)
    if operation == 'sae-family-report' and config.get('discoverPromotions', True):
        if (root / 'runs/model-variants').exists(): add('runs/model-variants')
    if not files: raise archives.Refusal('This operation has no resolvable scientific inputs.')
    # These inputs are pinned where they lie and read by a child on the same
    # filesystem; they are not an archive. When they must travel,
    # `diagnostic_inputs.package` applies the transport bound to this closure.
    return archives.pin(root, files)


@input_hashes.operation
def plan(request, root, lens_reference=CARRIED, verify_library=False):
    """The request's closure. `libraryLenses` lists each lens referred to by
    its hash and not carried, `{lensID, convertedSHA256, bytes}` sorted by
    `lensID`. The field is present whenever the plan was asked to refer
    (REFERRED), and otherwise only when a lens was in fact taken from the
    library, so a plan that carries every lens is the plan it always was."""
    normalized = managed_methods.request(request['operation'], request['parameters'])
    config = normalized['parameters']['config']
    referred = {}
    entries = inventory(normalized['operation'], config, root, lens_reference=lens_reference, referred=referred,
                        verify_library=verify_library)
    result = {'schemaVersion': 1, 'request': normalized, 'sourceRoot': str(Path(root).resolve()), 'files': entries}
    if lens_reference is REFERRED or referred:
        result['libraryLenses'] = [referred[name] for name in sorted(referred)]
    return {**result, 'planSHA256': archives.digest(result)}
