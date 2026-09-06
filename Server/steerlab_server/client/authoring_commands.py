"""Thin public adapters for remaining local study authorship."""
from pathlib import Path

from ..cli_envelope import CLIResult, VerbSpec

REVIEW = frozenset({'--manifest-sha256'})
VERB_SPECS = (
    VerbSpec('agent', 'list', purpose='Discover native and imported agent artifacts without changing evidence.'),
    VerbSpec('experiment', 'attach-agent', positional='<study>', purpose='Attach the exact reviewed agent to a reviewed draft.',
             value_flags=REVIEW | {'--artifact', '--artifact-sha256'}, required_flags=REVIEW | {'--artifact', '--artifact-sha256'}),
    VerbSpec('experiment', 'set-pipeline', positional='<study>', purpose='Replace or clear a reviewed pipeline declaration; does not execute it.',
             value_flags=REVIEW | {'--file'}, required_flags=REVIEW | {'--file'}),
    VerbSpec('panel', 'list', purpose='List local semantic panel inputs and catalog issues.'),
    VerbSpec('panel', 'inspect', positional='<path>', purpose='Inspect a panel and its exact file digest.'),
    VerbSpec('panel', 'check', positional='<file>', purpose='Validate proposed semantic panel JSON before publication.'),
    VerbSpec('panel', 'import', positional='<file>', purpose='Publish reviewed semantic panel bytes as a new immutable input.',
             value_flags=frozenset({'--file-sha256'}), required_flags=frozenset({'--file-sha256'})),
    VerbSpec('panel', 'compile', positional='<path>', purpose='Cast every seat and pin the compiled panel into a reviewed study.',
             value_flags=REVIEW | {'--experiment', '--casting', '--file-sha256'},
             required_flags=REVIEW | {'--experiment', '--casting', '--file-sha256'}),
)
EXPERIMENT_VERBS = frozenset(s.verb for s in VERB_SPECS if s.family == 'experiment')


def run(invocation):
    from ..client_cli import ClientRefusal
    from . import authoring_files as files, design_files, study_agents, study_panels, study_pipeline
    from ..experiment.manifest_files import digest_bytes
    spec, args, one = invocation.spec, invocation.positionals, invocation.one
    if (len(args) != (0 if spec.verb == 'list' else 1)
            or any(one(f) is None for f in spec.required_flags)
            or any(len(v) != 1 for v in invocation.flags.values())):
        raise ClientRefusal(code='usage', reason='Supply the declared arguments and each required flag once.',
                            repair_action=f'steerlab {spec.label} --help')
    root = files.root_path()
    if spec.family == 'agent':
        result = study_agents.catalog(root=root)
    elif spec.verb == 'attach-agent':
        result = study_agents.attach(args[0], one('--artifact'), root=root, expected=one('--manifest-sha256'), artifact_sha256=one('--artifact-sha256'))
    elif spec.verb == 'set-pipeline':
        raw = Path(one('--file')).read_bytes()
        block = None if raw.strip() == b'null' else design_files.decode(raw)
        result = study_pipeline.save(args[0], block, root=root, expected=one('--manifest-sha256'))
    elif spec.verb == 'list':
        result = study_panels.catalog(root=root)
    elif spec.verb == 'inspect':
        result = study_panels.inspect(args[0], root=root)
    elif spec.verb == 'compile':
        result = study_panels.compile(one('--experiment'), args[0], design_files.decode(Path(one('--casting')).read_bytes()),
            root=root, expected=one('--manifest-sha256'), file_sha256=one('--file-sha256'))
    else:
        data = Path(args[0]).read_bytes()
        if spec.verb == 'import':
            result = study_panels.publish(data, root=root, expected=one('--file-sha256'))
        else:
            result = {'document': study_panels.validate(design_files.decode(data)), 'fileSHA256': digest_bytes(data), 'valid': True}
    print(f'{spec.label}: complete; inspect the returned review before continuing')
    return CLIResult(message='Authoring review complete.', changed=result.get('changed', False), payload=result)
