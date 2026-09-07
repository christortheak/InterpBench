"""Run config owners in a root-scoped validation process, without model loading."""
import json
from dataclasses import asdict, is_dataclass
import os
from pathlib import Path
import subprocess
import sys
from ..experiment import diagnostic_archives as archives


def validate(request, root):
    environment = {**os.environ, 'STEERLAB_ROOT': str(Path(root).resolve()), 'STEERLAB_RUN_ROOT': str(Path(root).resolve() / 'runs'), 'PYTHONPATH': str(Path(__file__).resolve().parents[2])}
    try:
        result = subprocess.run([sys.executable, '-m', __name__], input=archives.encoded({'request': request, 'root': str(root)}),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=root, env=environment, timeout=120)
    except subprocess.TimeoutExpired:
        raise archives.Refusal('Scientific config validation timed out; inspect the engine environment before retrying.') from None
    try: response = json.loads(result.stdout)
    except (ValueError, UnicodeError): raise archives.Refusal('Scientific config validation returned no result; inspect the engine environment.') from None
    if result.returncode or not response.get('ok'): raise archives.Refusal(response.get('reason', 'Scientific config refused.'))
    return {'effectiveConfig': response['effectiveConfig'], 'models': response['models']}


def main():
    try:
        from ..experiment import managed_methods
        data = json.load(sys.stdin)
        request = data['request']; root = str(Path(data['root']).resolve())
        os.environ['STEERLAB_ROOT'] = root
        parsed = managed_methods.validate(request['operation'], request['parameters']['config'], root)
        effective = parsed.to_dict() if hasattr(parsed, 'to_dict') else (asdict(parsed) if is_dataclass(parsed) else request['parameters']['config'])
        config = request['parameters']['config']
        if request['operation'] == 'optvec-campaign':
            from ..experiment import optvec_campaign
            identities = {(cell.config['modelID'], cell.config['revision']) for cell in optvec_campaign.plan(parsed)}
        else:
            identities = {(config['modelID'], config.get('revision'))} if config.get('modelID') else set()
        models = [{'modelID': model, 'revision': revision} for model, revision in sorted(identities)]
        print(json.dumps({'ok': True, 'effectiveConfig': effective, 'models': models}, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({'ok': False, 'reason': str(exc)})); return 65


if __name__ == '__main__': raise SystemExit(main())
