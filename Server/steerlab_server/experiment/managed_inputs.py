"""Portable dependency closure for managed scientific requests (no GPU imports)."""
from . import input_hashes
import json
from pathlib import Path
from . import diagnostic_archives as archives, managed_methods, paths

from .operation_bindings import INPUT_ROLES


def inventory(operation, config, root):
    root = Path(root).resolve(); files = set()
    roles = INPUT_ROLES[operation]
    def add(reference, artifact=False):
        archives.parts(reference)
        candidates = [reference + '.json', reference + '.safetensors'] if artifact else [reference]
        for candidate in candidates: files.update(archives.files_in(root, candidate))
    def walk(value, key=''):
        if value is None: return
        if roles.get(key) == 'lens':
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
            selected = Path(artifact_paths.converted_file(value, converted['path'], str(root)))
            if not selected.is_relative_to(root): raise archives.Refusal('Managed lenses require converted tensors in the workspace lens library.')
            relative = selected.relative_to(root).as_posix()
            add(relative)
            if archives.file_hash(archives.ordinary(root, relative)) != converted.get('sha256'):
                raise archives.Refusal('Converted lens bytes differ from the imported hash; re-import or restore the lens.')
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
    if operation == 'jlens-fit-round':
        from .jlens_round import RoundConfig, prepare
        for child in prepare(RoundConfig.from_dict(config), root)['shards']:
            files.update(e['path'] for e in inventory('jlens-fit', child['parameters']['config'], root))
    if operation == 'jlens-fit-benchmark':
        from .jlens_benchmark import BenchmarkConfig, fitting_config
        nested = fitting_config(BenchmarkConfig.from_dict(config), root).to_dict()
        files.update(e['path'] for e in inventory('jlens-fit', nested, root))
    if operation == 'jlens-fit':
        from .jlens_fit import checkpoint_files
        for relative in checkpoint_files(config, root): add(relative)
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
    return archives.snapshot(root, files)


@input_hashes.operation
def plan(request, root):
    normalized = managed_methods.request(request['operation'], request['parameters'])
    config = normalized['parameters']['config']
    entries = inventory(normalized['operation'], config, root)
    result = {'schemaVersion': 1, 'request': normalized, 'sourceRoot': str(Path(root).resolve()), 'files': entries}
    return {**result, 'planSHA256': archives.digest(result)}
