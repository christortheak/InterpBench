"""Root-scoped campaign adapter; numerical training remains in OptVec's owner."""
import contextlib
import json
import os
from pathlib import Path
import sys
from ..experiment import diagnostic_archives as archives, managed_inputs, manifest_files
from . import executors


def materialize(config, root):
    from ..experiment import optvec_campaign as owner
    parsed = owner.OptVecCampaignConfig.from_dict(config)
    cells = owner.plan(parsed)
    request = {'operation': 'optvec-campaign', 'parameters': {'config': config}}
    source = managed_inputs.plan(request, root)
    directory = Path(owner.materialize(parsed, root=root))
    packet = directory / 'managed-campaign.json'
    files = [str((directory / owner.CAMPAIGN_FILENAME).relative_to(root))]
    files += [str((directory / owner.CELLS_DIRNAME / cell.cell_id / owner.CELL_CONFIG_FILENAME).relative_to(root)) for cell in cells]
    document = {'root': str(root), 'source': source, 'files': archives.snapshot(root, files)}
    packet.write_bytes(archives.encoded(document)); packet_hash = archives.file_hash(packet)
    for cell in cells:
        cell_dir = directory / owner.CELLS_DIRNAME / cell.cell_id
        resources = parsed.slurm.resources(job_name=owner.job_name_for(parsed, cell.cell_id, str(directory)))
        command = [sys.executable, '-m', __name__, 'cell', str(packet), packet_hash, cell.cell_id]
        bundle = executors.SlurmExecutor().create_bundle(str(cell_dir / 'managed-job'), command,
            env={'STEERLAB_ROOT': str(root), 'STEERLAB_RUN_ROOT': str(cell_dir / 'outputs')}, resources=resources)
        # This is a newly materialized campaign, before any submission. The
        # existing campaign scheduler continues to own queue occupancy/retries.
        target = cell_dir / owner.CELL_SCRIPT_FILENAME
        target.write_text(Path(bundle.script_path).read_text())
        files.extend(str(p.relative_to(root)) for p in (target, Path(bundle.script_path), Path(bundle.manifest_path)))
    files.append(str(packet.relative_to(root)))
    return {'campaignDirectory': str(directory), 'campaignExecution': 'materializedOnly',
            'managedCampaign': {'packetSHA256': packet_hash, 'staticFiles': archives.snapshot(root, files)}}


def verify(directory, pin):
    directory = Path(directory); packet = directory / 'managed-campaign.json'
    if archives.file_hash(packet) != pin['packetSHA256']: raise archives.Refusal('Campaign packet changed.')
    data = json.loads(packet.read_bytes()); root = data['root']
    if archives.snapshot(root, [f['path'] for f in pin['staticFiles']]) != pin['staticFiles']:
        raise archives.Refusal('Campaign config or scheduler script changed after materialization.')
    if managed_inputs.plan(data['source']['request'], root) != data['source']:
        raise archives.Refusal('Campaign input bytes changed after materialization.')
    return data


def operate(directory, pin, action, expected=None, *, runner=None):
    from ..experiment import optvec_campaign as owner
    directory = Path(directory)
    # Managed requests serialize both review checks and the owner's complete
    # scheduler cycle. This lock remains outside scientific content hashes.
    with manifest_files.transaction(str(directory / 'managed-action'), workspace_root=str(directory.parent.parent)):
        data = verify(directory, pin)
        report = owner.status(str(directory), runner=runner)
        state = owner.read_state(str(directory))
        plan = {'campaignDirectory': str(directory), 'pin': pin, 'stateSHA256': archives.digest(state),
                'status': report, 'actions': ['submit', 'cancel'], 'cancellationScope': 'All cells, including allocations recovered by exact per-cell scheduler name', 'changed': False}
        plan['planSHA256'] = archives.digest(plan)
        if action in ('status', 'plan'): return plan
        if expected != plan['planSHA256']: raise archives.Refusal('Campaign state changed; review a fresh action plan.')
        if action == 'submit':
            with contextlib.redirect_stdout(sys.stderr):
                result = owner.submit(str(directory), runner=runner, sbatch_command=executors.scheduler_commands().submit)
            return {'changed': archives.digest(owner.read_state(str(directory))) != archives.digest(state), 'response': result}
        if action == 'cancel':
            selected = set()
            config = owner.OptVecCampaignConfig.from_dict(owner.read_campaign(str(directory))['config'])
            for row in report['cells']:
                identity = row['jobID']
                if not identity:
                    identity = executors.SlurmExecutor().find_job_by_name(owner.job_name_for(config, row['cellID'], str(directory)))
                if identity: selected.add(identity)
            # Resolve every unknown identity before sending any cancellation;
            # query failure never means the whole campaign has stopped.
            outcomes, errors = {}, {}
            for job in sorted(selected):
                try: outcomes[job] = executors.SlurmExecutor().cancel(job)
                except Exception as exc: outcomes[job] = False; errors[job] = str(exc)
            return {'changed': any(outcomes.values()), 'cancellationRequested': outcomes, 'cancellationErrors': errors,
                    'nextAction': 'Read status to establish termination. Cancellation acceptance is not completion; another submit is an explicit top-up.'}
        raise archives.Refusal('Unknown campaign action.')


def cell(packet, expected, cell_id):
    from ..experiment import optvec_campaign as owner, optvec_train
    packet = Path(packet)
    if archives.file_hash(packet) != expected: raise archives.Refusal('Queued campaign packet changed.')
    data = json.loads(packet.read_bytes()); root = data['root']
    if archives.snapshot(root, [f['path'] for f in data['files']]) != data['files']:
        raise archives.Refusal('Queued campaign configuration changed.')
    if managed_inputs.plan(data['source']['request'], root) != data['source']:
        raise archives.Refusal('Queued campaign inputs changed before model load.')
    archives.parts(cell_id)
    if '/' in cell_id: raise archives.Refusal('Invalid campaign cell.')
    directory = packet.parent / owner.CELLS_DIRNAME / cell_id
    if (directory / owner.COMPLETION_MARKER).exists() or (directory / 'managed-result.json').exists():
        raise archives.Refusal('This cell already has completed evidence; inspect it instead of rerunning in place.')
    config_path = directory / owner.CELL_CONFIG_FILENAME
    if str(config_path.relative_to(root)) not in {f['path'] for f in data['files']}:
        raise archives.Refusal('Cell is absent from the reviewed campaign.')
    os.environ['STEERLAB_ROOT'] = root
    os.environ['STEERLAB_RUN_ROOT'] = str(directory / 'outputs')
    os.chdir(root)
    with contextlib.redirect_stdout(sys.stderr):
        result = optvec_train.train(optvec_train.load_config(str(config_path)))
    with (directory / 'managed-result.json').open('xb') as handle:
        handle.write(archives.encoded(result))
    (directory / owner.COMPLETION_MARKER).touch(exist_ok=False)
    return result


def main():
    try:
        if len(sys.argv) > 1 and sys.argv[1] == 'cell': result = cell(*sys.argv[2:])
        else:
            request = json.load(sys.stdin)
            result = operate(**request)
        print(json.dumps({'ok': True, 'result': result}, allow_nan=False)); return 0
    except Exception as exc:
        print(json.dumps({'ok': False, 'reason': str(exc)})); return 65


if __name__ == '__main__': raise SystemExit(main())
