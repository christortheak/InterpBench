"""Read-only discovery of shipped scientific workflows; never imports execution."""
from ..cli_envelope import CLIResult, VerbSpec
from ..experiment import science_catalog

VERB_SPECS = (
    VerbSpec('science', 'probe-list', purpose='List portable and legacy probes in the local workspace without changing artifacts.'),
    VerbSpec('science', 'probe-inspect', positional='<path>', purpose='Inspect exact probe bytes, score meaning, and provenance limitations.'),
    VerbSpec('science', 'corpus-preview', positional='<spec.json>', purpose='Read chosen data sources and capture a reproducible fitting corpus preview; public dataset files may download.'),
    VerbSpec('science', 'corpus-publish', positional='<preview-id>', purpose='Save the reviewed fitting corpus and provenance in a new directory.', value_flags=frozenset({'--plan-sha256', '--destination'}), required_flags=frozenset({'--plan-sha256', '--destination'})),
    VerbSpec('science', 'artifact-plan', positional='<description.json>', purpose='Inspect a custom lens or SAE decoder and hash its source files without publishing.'),
    VerbSpec('science', 'artifact-import', positional='<description.json>', purpose='Import the reviewed instrument into a fresh library destination.', value_flags=frozenset({'--plan-sha256'}), required_flags=frozenset({'--plan-sha256'})),
    VerbSpec('science', 'sae-check', positional='<roster-path>', purpose='Inspect the SAE roster and surface qualification warnings without a model.'),
    VerbSpec('science', 'sae-show', positional='<qualification-path>', purpose='Inspect an existing qualification record without changing its scientific status.'),
    VerbSpec('science', 'sae-pin-plan', positional='<roster-path>', purpose='Review roster and draft bytes before pinning.', value_flags=frozenset({'--experiment'}), required_flags=frozenset({'--experiment'})),
    VerbSpec('science', 'sae-pin', positional='<roster-path>', purpose='Pin the reviewed roster to the unchanged draft.', value_flags=frozenset({'--experiment', '--plan-sha256'}), required_flags=frozenset({'--experiment', '--plan-sha256'})),
    VerbSpec('science', 'interview', positional='<operation>', purpose='Read the shared method-specific interview and exact form fields.'),
    VerbSpec('science', 'draft', positional='<operation>', purpose='Resolve conceptual answers into a reviewed request and input hashes without publication.', value_flags=frozenset({'--answers'}), required_flags=frozenset({'--answers'})),
    VerbSpec('science', 'publish', positional='<operation>', purpose='Publish the reviewed request and rationale together in a new requests directory.', value_flags=frozenset({'--answers', '--destination', '--plan-sha256'}), required_flags=frozenset({'--answers', '--destination', '--plan-sha256'})),
    VerbSpec('science', 'input-plan', positional='<request.json>', purpose='Discover and hash local standalone diagnostic inputs without execution.'),
    VerbSpec('science', 'package', positional='<request.json>', purpose='Package exactly the reviewed local diagnostic inputs for permitted transport.', value_flags=frozenset({'--archive', '--plan-sha256'}), required_flags=frozenset({'--archive', '--plan-sha256'})),
    VerbSpec('science', 'import', positional='<archive.tar.gz>', purpose='Verify and import diagnostic evidence into the captured local workspace without replacing outputs.', value_flags=frozenset({'--sha256'}), required_flags=frozenset({'--sha256'})),
    VerbSpec('science', 'custody', purpose='Reverify and list retained diagnostic evidence receipts for offline inspection.'),
    VerbSpec('science', 'verify-custody', positional='<receipt-sha256>', purpose='Re-read the retained archive and every expanded local output before reporting custody.'),
    VerbSpec('science', 'list', purpose='List shipped methods, public operation paths and engine restrictions; does not execute.'),
    VerbSpec('science', 'guide', positional='<method>', purpose='Read the shared method guide, dataset schemas and coworker/reviewer instructions.'),
    VerbSpec('science', 'operation', positional='<operation>', purpose='Inspect exact public execution paths, outputs and restrictions for one operation.'),
)


def run(invocation):
    from ..client_cli import ClientRefusal
    verb, args = invocation.spec.verb, invocation.positionals
    if verb in {'probe-list', 'probe-inspect', 'corpus-preview', 'corpus-publish', 'artifact-plan', 'artifact-import', 'sae-check', 'sae-show', 'sae-pin-plan', 'sae-pin', 'interview', 'draft', 'publish', 'input-plan', 'package', 'import', 'custody', 'verify-custody'}:
        from .diagnostic_commands import local
        return local(invocation)
    if len(args) != (0 if verb == 'list' else 1):
        raise ClientRefusal(code='usage', reason='Supply exactly the declared arguments.', repair_action=f'steerlab science {verb} --help')
    try:
        result = science_catalog.catalog() if verb == 'list' else getattr(science_catalog, verb)(args[0])
    except science_catalog.ScienceRefusal as exc:
        raise ClientRefusal(code='usage', reason=str(exc), repair_action=exc.repair_action, state='blocked') from exc
    import json
    print(result['text'] if verb == 'guide' else json.dumps(result, indent=2, ensure_ascii=False))
    return CLIResult(message='Scientific workflow reference read; no execution performed.', payload=result)
