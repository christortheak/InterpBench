"""Read-only discovery of shipped scientific workflows; never imports execution."""
from ..cli_envelope import CLIResult, VerbSpec
from ..experiment import science_catalog

#: Swift twin: the `science report` entry in ``ExperimentCLIParser.specs``.
REPORT_PURPOSE = ('Turn a stored J-lens assessment report into one self-contained HTML page a person can read. '
                  'Give the run folder or its assessment-report.json. A run folder is never written to: the page '
                  'goes to reports/ in the workspace, or to --out (read from the workspace unless absolute).')

VERB_SPECS = (
    VerbSpec('science', 'evidence-analyze', positional='<path>', purpose='Compare retained probe readings and requested/applied policy actions in a run.'),
    VerbSpec('science', 'policy-list', purpose='List saved intervention policies.'),
    VerbSpec('science', 'policy-inspect', positional="<path>", purpose='Inspect a policy and its exact input bindings.'),
    VerbSpec('science', 'policy-review', positional="<settings.json>", purpose='Review policy settings and embed their exact input bytes.'),
    VerbSpec('science', 'policy-publish', positional="<settings.json>", purpose='Publish the reviewed policy as an immutable artifact.', value_flags=frozenset({'--plan-sha256'}), required_flags=frozenset({'--plan-sha256'})),
    VerbSpec('science', 'policy-attach-review', positional="<settings.json>", purpose='Review a new agent version with the selected policies.'),
    VerbSpec('science', 'policy-attach', positional="<settings.json>", purpose='Create the reviewed agent version without editing its source.', value_flags=frozenset({'--plan-sha256'}), required_flags=frozenset({'--plan-sha256'})),
    VerbSpec('science', 'measurements-review', positional='<experiment>', purpose='Review probe measurement settings and pinned inputs for a draft study.', value_flags=frozenset({'--settings'}), required_flags=frozenset({'--settings'})),
    VerbSpec('science', 'measurements-save', positional='<experiment>', purpose='Save reviewed probe measurement settings to the unchanged draft.', value_flags=frozenset({'--settings', '--plan-sha256'}), required_flags=frozenset({'--settings', '--plan-sha256'})),
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
    VerbSpec('science', 'package', positional='<request.json>', purpose='Package exactly the reviewed local diagnostic inputs for permitted transport. Inputs that carry custom code need each SHA-256 the input plan shows, in --custom-code-sha256.', value_flags=frozenset({'--archive', '--plan-sha256', '--custom-code-sha256'}), required_flags=frozenset({'--archive', '--plan-sha256'})),
    VerbSpec('science', 'import', positional='<archive.tar.gz>', purpose='Verify and import diagnostic evidence into the captured local workspace without replacing outputs.', value_flags=frozenset({'--sha256'}), required_flags=frozenset({'--sha256'})),
    VerbSpec('science', 'custody', purpose='Reverify and list retained diagnostic evidence receipts for offline inspection.'),
    VerbSpec('science', 'verify-custody', positional='<receipt-sha256>', purpose='Re-read the retained archive and every expanded local output before reporting custody.'),
    # `--out` is this verb's own argument (the page), so the envelope has no file spelling here.
    VerbSpec('science', 'report', positional='<run-folder-or-report.json>', purpose=REPORT_PURPOSE, value_flags=frozenset({'--out'})),
    VerbSpec('science', 'list', purpose='List shipped methods, public operation paths and engine restrictions; does not execute. With --brief, return a short index instead: ids, titles, one line of purpose each, and where each operation runs.', boolean_flags=frozenset({'--brief'})),
    VerbSpec('science', 'guide', positional='<method>', purpose='Read the shared method guide, dataset schemas and coworker/reviewer instructions.'),
    VerbSpec('science', 'operation', positional='<operation>', purpose='Inspect exact public execution paths, outputs, restrictions, and where it runs on each backend for one operation.'),
)


def run(invocation):
    from ..client_cli import ClientRefusal
    verb, args = invocation.spec.verb, invocation.positionals
    if verb in {'evidence-analyze', 'policy-list', 'policy-inspect', 'policy-review', 'policy-publish', 'policy-attach-review', 'policy-attach', 'measurements-review', 'measurements-save', 'probe-list', 'probe-inspect', 'corpus-preview', 'corpus-publish', 'artifact-plan', 'artifact-import', 'sae-check', 'sae-show', 'sae-pin-plan', 'sae-pin', 'interview', 'draft', 'publish', 'input-plan', 'package', 'import', 'custody', 'verify-custody', 'report'}:
        from .diagnostic_commands import local
        return local(invocation)
    if len(args) != (0 if verb == 'list' else 1):
        raise ClientRefusal(code='usage', reason='Supply exactly the declared arguments.', repair_action=f'steerlab science {verb} --help')
    brief = verb == 'list' and '--brief' in invocation.flags
    try:
        if verb == 'list':
            result = science_catalog.brief() if brief else science_catalog.catalog()
        else:
            result = getattr(science_catalog, verb)(args[0])
    except science_catalog.ScienceRefusal as exc:
        raise ClientRefusal(code='usage', reason=str(exc), repair_action=exc.repair_action, state='blocked') from exc
    if invocation.json:
        # Under --json this print lands on stderr, and the document on stdout
        # already carries the whole result. Echoing the body again doubled what
        # a caller reading both streams paid for the catalog, so one line says
        # what was read and where it is.
        print(summary(verb, result))
    else:
        import json
        print(result['text'] if verb == 'guide' else json.dumps(result, indent=2, ensure_ascii=False))
    from ..cli_envelope import next_action
    return CLIResult(message='Scientific workflow reference read; no execution performed.', payload=result,
                     next_action=next_action(BRIEF_NEXT_VERB, detail=BRIEF_NEXT_DETAIL) if brief else None)


#: Where a caller goes after the short index. Swift twins:
#: ``ScienceCatalog.briefNextVerb`` and ``ScienceCatalog.briefNextDetail``.
BRIEF_NEXT_VERB = 'science guide <method>'
BRIEF_NEXT_DETAIL = ('Read one method with science guide <method>, or one operation with science operation <operation>. '
                     'science list without --brief is the full catalog.')


def summary(verb, result):
    """One stderr line for a --json read: what was read, never the body."""
    if verb == 'list':
        return (f"science list: {len(result['methods'])} methods and {len(result['operations'])} operations "
                f"(catalog {result['catalogSHA256'][:12]}…); the document is on stdout")
    if verb == 'guide':
        return (f"science guide {result['method']['id']}: {len(result['text'])} characters "
                f"(guide {result['guideSHA256'][:12]}…); the document is on stdout")
    return f"science operation {result['id']}: the document is on stdout"
