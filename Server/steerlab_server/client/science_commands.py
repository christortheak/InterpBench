"""Read-only discovery of shipped scientific workflows; never imports execution."""
from ..cli_envelope import CLIResult, VerbSpec
from ..experiment import science_catalog

VERB_SPECS = (
    VerbSpec('science', 'list', purpose='List shipped methods, public operation paths and engine restrictions; does not execute.'),
    VerbSpec('science', 'guide', positional='<method>', purpose='Read the shared method guide, dataset schemas and coworker/reviewer instructions.'),
    VerbSpec('science', 'operation', positional='<operation>', purpose='Inspect exact public execution paths, outputs and restrictions for one operation.'),
)


def run(invocation):
    from ..client_cli import ClientRefusal
    verb, args = invocation.spec.verb, invocation.positionals
    if len(args) != (0 if verb == 'list' else 1):
        raise ClientRefusal(code='usage', reason='Supply exactly the declared arguments.', repair_action=f'steerlab science {verb} --help')
    try:
        result = science_catalog.catalog() if verb == 'list' else getattr(science_catalog, verb)(args[0])
    except science_catalog.ScienceRefusal as exc:
        raise ClientRefusal(code='usage', reason=str(exc), repair_action=exc.repair_action, state='blocked') from exc
    import json
    print(result['text'] if verb == 'guide' else json.dumps(result, indent=2, ensure_ascii=False))
    return CLIResult(message='Scientific workflow reference read; no execution performed.', payload=result)
