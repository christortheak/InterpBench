"""CPU CLI adapters; scientific admission and publication stay in their owners."""
import os
import sys
from ..cli_envelope import CLIResult


def refusal(exc, *, repair):
    from . import lifecycle_gates
    gate = lifecycle_gates.gate_of(exc)
    state = 'notFound' if isinstance(exc, FileNotFoundError) else 'refused' if gate or isinstance(exc, ValueError) else 'failed'
    sys.stderr.write(str(exc) + '\n')
    return CLIResult(message=str(exc), state=state, exit_code={'notFound': 66, 'failed': 70, 'refused': 65}[state], code=gate or ('scientificOperationFailed' if state == 'failed' else 'scientificOperationRefused'), gate=gate,
                     repair_action=lifecycle_gates.repair_of(exc) or repair)


def complete_judgment(name, rest, *, root, sweep, load_judgments):
    from . import tasks, judgment_evidence, experiment_store
    flags = ('--awaiting-run', '--judgments')
    verb = 'complete-sweep-judgment' if sweep else 'complete-judgment'
    repair = f'steerlab-server experiment {verb} {name} --awaiting-run <issued-run> --judgments <reviewed-file> --json'
    if len(rest) != 5 or any(rest.count(key) != 1 or rest.index(key) + 1 >= len(rest) for key in flags):
        sys.stderr.write(f'usage: experiment {verb} <name> --awaiting-run <run> --judgments <file>\n')
        return CLIResult(message=f'usage: experiment {verb} <name> --awaiting-run <run> --judgments <file>',
                         state='blocked', exit_code=64, code='usage', repair_action=repair)
    def flag(key):
        return rest[rest.index(key) + 1]
    awaiting = os.path.basename(os.path.normpath(flag('--awaiting-run')))
    try:
        rows, instructions_sha = load_judgments(flag('--judgments'))
    except ValueError as exc:
        sys.stderr.write(str(exc) + '\n')
        return CLIResult(message=str(exc), state='blocked', exit_code=64, code='invalidJudgmentInput', repair_action=repair)
    except OSError as exc:
        return refusal(exc, repair=repair)
    if sweep and instructions_sha is not None:
        sys.stderr.write('Sweep has no evaluation instructionsSha256 field.\n')
        return CLIResult(message='Sweep has no evaluation instructionsSha256 claim field; supply the issued sweep judgment rows.',
                         state='blocked', exit_code=64, code='invalidJudgmentInput', repair_action=repair)
    try:
        lookup = judgment_evidence.find_judgment_run if sweep else judgment_evidence.find_evaluate_judgment_run
        existing = lookup(name, awaiting, root)
        before = experiment_store.load_raw(name, root).source_digest
        if sweep:
            output = tasks.complete_sweep_judgment(name, awaiting, rows, root=root)
        else:
            output = tasks.complete_evaluate_judgment(name, awaiting, rows, root=root, instructions_sha256=instructions_sha)
        after = experiment_store.load_raw(name, root).source_digest
    except (ValueError, RuntimeError, OSError) as exc:
        return refusal(exc, repair='Inspect the original awaiting packets, study epoch and full declared judge coverage; repair the named refusal before explicit completion. ' + repair)
    print(output)
    return CLIResult(message='Deferred judgment completion verified.', changed=existing is None or before != after,
        payload={'runDirectory': output, 'awaitingRun': awaiting, 'reused': existing is not None,
                 'kind': 'sweep' if sweep else 'evaluate'})


def rescore_style(name, *, root, source, allow_unverified_epoch):
    from . import tasks
    try:
        output = tasks.rescore_style(name, root, source, allow_unverified_epoch=allow_unverified_epoch)
    except (ValueError, RuntimeError, OSError) as exc:
        return refusal(exc, repair='Inspect the pinned taxonomy and source epoch; select the matching source with --source. Do not change historical run bytes to bypass the refusal.')
    print(output)
    return CLIResult(message='Style rescoring complete.', changed=True, payload={'runDirectory': output})
