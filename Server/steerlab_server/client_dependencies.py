"""CPU client capabilities shared by readiness and managed-runtime activation."""
import importlib.metadata
import json
import subprocess
import sys

UPGRADE_REPAIR = ('In the app, open Workspace → Research Setup, review the setup plan, and approve the client environment update. '
                  'For the app-free client, use the matching current release: steerlab setup plan --release <current-client-release>, '
                  'then approve setup apply with that release and the returned plan hash. Basic authoring and plain-text preparation can still be used when available.')
DEPENDENCIES = ('numpy', 'safetensors', 'httpx', 'pyarrow', 'huggingface-hub', 'transformers')
CAPABILITIES = {
    'basicAuthoring': {'label': 'Basic study authoring', 'modules': ['numpy', 'safetensors', 'httpx']},
    'parquet': {'label': 'Parquet corpus files', 'modules': ['pyarrow.parquet']},
    'huggingFace': {'label': 'Public dataset downloads', 'modules': ['huggingface_hub.hf_api', 'huggingface_hub.file_download']},
    'tokenPreview': {'label': 'Offline token previews', 'modules': ['transformers.models.auto.tokenization_auto']},
}
PROBE = '''
import importlib,json,os,sys
# Readiness exercises CPU utilities only, even in an engine environment.
os.environ['USE_TORCH']='0'
os.environ['USE_TF']='0'
os.environ['USE_FLAX']='0'
result={}
for name, spec in json.loads(sys.argv[1]).items():
    try:
        for module in spec['modules']: importlib.import_module(module)
        result[name]={'ready':True,'diagnostic':None}
    except Exception as exc:
        result[name]={'ready':False,'diagnostic':str(exc)}
print(json.dumps(result))
'''


def versions():
    result = {}
    for name in DEPENDENCIES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def probe():
    """One isolated process, no source downloads, model loads, or parent imports."""
    try:
        process = subprocess.run([sys.executable, '-I', '-c', PROBE, json.dumps(CAPABILITIES)],
                                 capture_output=True, text=True, timeout=30)
        if process.returncode:
            raise ValueError(process.stderr.strip() or 'The client import probe did not complete.')
        results = json.loads(process.stdout)
        if not isinstance(results, dict) or set(results) != set(CAPABILITIES) or any(
                not isinstance(value, dict) or type(value.get('ready')) is not bool for value in results.values()):
            raise ValueError('The client import probe returned an incomplete report.')
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        results = {name: {'ready': False, 'diagnostic': str(exc)} for name in CAPABILITIES}
    return {name: {**result, 'label': CAPABILITIES[name]['label'],
                   'repairAction': None if result['ready'] else UPGRADE_REPAIR} for name, result in results.items()}
