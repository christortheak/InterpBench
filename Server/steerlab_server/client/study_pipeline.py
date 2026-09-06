"""Reviewed pipeline declaration, with the execution owner's gate validation."""
from . import authoring_files as files, design_files
from ..experiment import experiment_store as store
from ..experiment.pipeline_spec import resolve_pipeline


def save(name, block, *, root, expected):
    try:
        resolve_pipeline(block)
    except (ValueError, TypeError) as exc:
        files.refuse(f'Invalid pipeline declaration: {exc}')
    design_files.ordinary(root, str(files.study_path(root, name).relative_to(root)))
    with files.reviewed_draft(name, root, expected):
        document = store.load_raw(name, str(root))
        if block is None:
            document.pop('pipeline', None)
        else:
            document['pipeline'] = block
        store.save_raw(document, str(root))
        return {**files.snapshot(name, root), 'changed': document.source_digest != expected}
