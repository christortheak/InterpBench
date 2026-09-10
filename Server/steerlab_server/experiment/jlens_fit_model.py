"""Local prepared-checkpoint adapter. Heavy dependencies are execution-only."""
import hashlib
from importlib import metadata
import json
from pathlib import Path

from .jlens_fit import FitError
from ..jlens.backend import REFERENCE_COMMIT


def load(config, log):
    import torch
    import transformers
    import jlens
    from jlens import fitting
    from huggingface_hub import snapshot_download
    from ..steering.model_loader import resolve_device, hf_dtype_kwargs
    direct = json.loads(metadata.distribution('jlens').read_text('direct_url.json') or '{}')
    if direct.get('vcs_info', {}).get('commit_id') != REFERENCE_COMMIT:
        raise FitError('Install the engine’s pinned jlens extra before fitting; the reference kernel version differs.')
    device = resolve_device(config.device)
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise FitError('No CUDA device is visible. Select a GPU allocation, or choose mps/cpu for a suitable local pilot.')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise FitError('MPS is unavailable on this host. Choose a supported execution device.')
    if device != config.device and config.device not in ('cuda',):
        raise FitError('The requested device is unavailable; choose an available compute target.')
    # Explicitly offline: authoring and fitting never acquire an unreviewed model.
    try:
        snapshot = snapshot_download(config.modelID, revision=config.revision, local_files_only=True)
    except Exception as exc:
        raise FitError('Prepare this exact model checkpoint in the engine cache before fitting.') from exc
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    hf_config = AutoConfig.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    from .jlens_fit_review import estimate
    text_config = hf_config.get_text_config()
    cost = estimate(config, getattr(text_config, 'hidden_size', None), getattr(text_config, 'num_hidden_layers', None))
    if cost:
        log('Fitting cost before loading weights: ' + cost['summary'])
        log(cost['limitations'])
    if getattr(hf_config, 'quantization_config', None):
        raise FitError('Use unquantized model weights for Jacobian fitting; this checkpoint declares quantization.')
    architecture = getattr(hf_config, 'architectures', []) or []
    factory = AutoModelForImageTextToText if any('ForConditionalGeneration' in x for x in architecture) else AutoModelForCausalLM
    log(f'Loading the prepared checkpoint on {device} in {config.dtype}; no weights will be downloaded.')
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    hf = factory.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False,
        attn_implementation='eager', **hf_dtype_kwargs(getattr(torch, config.dtype)))
    hf.to(device)
    hf.config.use_cache = False
    # Preserve the tokenizer's recorded BOS behavior; no implicit chat rendering.
    model = jlens.from_hf(hf, tokenizer, compile=False, force_bos=False)
    text_config = hf.config.get_text_config()
    text_config.use_cache = False
    actual = str(next(hf.parameters()).dtype).replace('torch.', '')
    if actual != config.dtype:
        raise FitError('The loaded model dtype differs from the requested fitting precision.')
    tokenizer_material = (tokenizer.backend_tokenizer.to_str() if hasattr(tokenizer, 'backend_tokenizer')
                          else json.dumps(tokenizer.get_vocab(), sort_keys=True))
    tokenizer_material += json.dumps(tokenizer.special_tokens_map, sort_keys=True)
    runtime = {
        'referenceCommit': REFERENCE_COMMIT,
        'kernelSHA256': hashlib.sha256(Path(fitting.__file__).read_bytes()).hexdigest(),
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'modelConfigSHA256': hashlib.sha256((Path(snapshot) / 'config.json').read_bytes()).hexdigest(),
        'tokenizerSHA256': hashlib.sha256(tokenizer_material.encode()).hexdigest(),
        'dtype': actual, 'device': device, 'attention': 'eager',
        'modelClass': type(hf).__name__, 'forceBOS': False,
        'float32MatmulPrecision': torch.get_float32_matmul_precision(),
        'cudaTF32': torch.backends.cuda.matmul.allow_tf32,
        'cudnnTF32': torch.backends.cudnn.allow_tf32,
        'deterministicAlgorithms': torch.are_deterministic_algorithms_enabled(),
        'driverSHA256': hashlib.sha256(b''.join(
            Path(__file__).with_name(name + '.py').read_bytes()
            for name in ('jlens_fit', 'jlens_fit_execution', 'jlens_fit_model', 'jlens_fit_identity'))).hexdigest(),
    }
    return model, runtime
