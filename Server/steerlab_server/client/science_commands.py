"""Read-only discovery of shipped scientific workflows; never imports execution."""
from ..cli_envelope import CLIResult, VerbSpec
from ..experiment import science_catalog

VERB_SPECS = (
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
    if verb in {'input-plan', 'package', 'import', 'custody', 'verify-custody'}:
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
