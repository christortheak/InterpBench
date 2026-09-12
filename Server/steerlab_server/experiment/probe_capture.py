"""Managed text-to-activation capture, distinct from fitting and live measurement."""
from dataclasses import dataclass, asdict
import hashlib
from . import diagnostic_archives as archives, probe_artifacts as artifact, probe_data as data
from .probe_artifacts import ProbeError


@dataclass(frozen=True)
class CaptureConfig:
    modelID: str
    revision: str
    examples: dict
    layer: int
    site: str = 'residualPost'
    rendering: str = 'raw'
    population: str = 'prompt'
    position: str = 'lastNonPadding'
    splitPolicy: str = 'explicit'
    seed: int = 0
    maxExamples: int = 128
    maxSeqLen: int = 256
    maxRecords: int = 4096
    dtype: str = 'float32'
    device: str = 'cuda'

    @classmethod
    def from_dict(cls,value):
        if not isinstance(value,dict) or value.keys()-cls.__dataclass_fields__.keys(): raise ProbeError('Use the documented activation capture fields.')
        try:c=cls(**value)
        except TypeError as exc: raise ProbeError('Select a model version, labeled text file, and layer.') from exc
        artifact.text(c.modelID,'modelID');artifact.digest(c.revision,'revision',size=40)
        data.file_ref(c.examples,'examples');artifact.integer(c.layer,'layer');artifact.integer(c.seed,'seed')
        if c.seed>2**53-1:raise ProbeError('Choose a seed no larger than 2^53-1.')
        for key,limit in (('maxExamples',4096),('maxSeqLen',8192),('maxRecords',65536)):
            artifact.integer(getattr(c,key),key,1)
            if getattr(c,key)>limit:raise ProbeError(f'{key} exceeds the bounded capture limit of {limit}. Use a smaller pilot.')
        for key,choices in (('site',('residualPre','residualPost')),('rendering',('raw','chatTemplate')),('population',('prompt','completeText')),('position',('lastNonPadding','eachNonPadding')),('splitPolicy',('explicit','groupHash')),('dtype',('float32','bfloat16','float16'))):
            artifact.choice(getattr(c,key),choices,key)
        artifact.text(c.device,'device')
        if c.rendering=='chatTemplate' and c.population!='prompt':raise ProbeError('Chat capture renders a single user prompt with a generation cue. Use raw completeText for already-completed text.')
        return c
    def to_dict(self):return asdict(self)


def decoder_layers(model):
    """Explicit block paths only; do not guess a ModuleList by its size."""
    import torch
    for path in ('model.layers','model.language_model.layers','language_model.model.layers','transformer.h'):
        value=model
        for part in path.split('.'):value=getattr(value,part,None)
        if isinstance(value,torch.nn.ModuleList) and len(value):return value,path
    raise ProbeError('This model’s decoder block path is unknown. Add and test its residual-site mapping, or import activations with an explicit binding.')


def observe(model, block, encoded, config):
    """Read a single unpadded sequence; hooks are always removed, never replace output."""
    import torch
    captured=[]
    def read(tensor):
        if not isinstance(tensor,torch.Tensor) or tensor.ndim!=3 or tensor.shape[0]!=1:raise ProbeError('Expected one sequence of residual activations with shape [1, tokens, hidden].')
        if tensor.shape[1]!=encoded['input_ids'].shape[1]:raise ProbeError('Residual positions do not match the input token sequence.')
        positions=encoded['attention_mask'][0].nonzero().flatten()
        if not len(positions):raise ProbeError('The rendered example has no non-padding tokens.')
        if config.position=='lastNonPadding':positions=positions[-1:]
        values=tensor[0,positions.to(tensor.device)].detach().float().cpu()
        if not bool(torch.isfinite(values).all()):raise ProbeError('Captured activations are non-finite.')
        captured.append((positions.cpu().tolist(),values.tolist(),str(tensor.dtype).removeprefix('torch.')))
    if config.site=='residualPre':
        def pre(module,args,kwargs):read(args[0] if args else kwargs.get('hidden_states'))
        handle=block.register_forward_pre_hook(pre,with_kwargs=True)
    else:
        def post(module,args,output):read(output[0] if isinstance(output,tuple) else output)
        handle=block.register_forward_hook(post)
    try:
        with torch.inference_mode():model(**encoded,use_cache=False)
    finally:handle.remove()
    if len(captured)!=1:raise ProbeError('The declared decoder site must execute exactly once per capture pass.')
    return captured[0]


def tokenizer_identity(tokenizer):
    """A vocabulary alone does not identify a slow tokenizer's normalization."""
    backend=getattr(tokenizer,'backend_tokenizer',None)
    if backend is None: return None
    material={'backend':backend.to_str(),'specialTokens':tokenizer.special_tokens_map,
              'behavior':{key:getattr(tokenizer,key,None) for key in ('add_bos_token','add_eos_token','padding_side','truncation_side')}}
    return archives.digest(material)


def load(config,log):
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    from ..steering.model_loader import hf_dtype_kwargs
    try:snapshot=snapshot_download(config.modelID,revision=config.revision,local_files_only=True)
    except Exception as exc:raise ProbeError('Prepare this exact model version in the execution host’s cache before capture.') from exc
    hfconfig=AutoConfig.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    if getattr(hfconfig,'quantization_config',None):raise ProbeError('This capture recipe uses unquantized weights. Select a prepared unquantized checkpoint.')
    factory=AutoModelForImageTextToText if any('ForConditionalGeneration' in x for x in (getattr(hfconfig,'architectures',None) or [])) else AutoModelForCausalLM
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    log(f'Loading prepared model on {config.device} in {config.dtype}; no download or weight update.')
    model=factory.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,
        attn_implementation='eager',**hf_dtype_kwargs(getattr(torch,config.dtype))).to(config.device).eval()
    template=tokenizer.get_chat_template() if config.rendering=='chatTemplate' else None
    runtime={'torch':torch.__version__,'transformers':transformers.__version__,
        'tokenizerSHA256':tokenizer_identity(tokenizer),
        'templateSHA256':hashlib.sha256(template.encode()).hexdigest() if template else None,
        'modelClass':type(model).__name__,'device':config.device,'attention':'eager',
        'driverSHA256':archives.file_hash(__import__('pathlib').Path(__file__)),
        'configSHA256':archives.file_hash(__import__('pathlib').Path(snapshot)/'config.json')}
    return model,tokenizer,runtime


def capture(config, *, root, log=print, on_run_created=None):
    rows=data.text_rows(data.read(config.examples,root),config.splitPolicy,config.seed)
    chosen=rows[:config.maxExamples]
    run=data.new_run(root,'probe-capture',on_run_created)
    data.save(run,'capture-request.json',{'operation':'probe-capture','config':config.to_dict(),'availableRows':len(rows)})
    model,tokenizer,runtime=load(config,log)
    blocks,path=decoder_layers(model)
    if config.layer>=len(blocks):raise ProbeError(f'This model has {len(blocks)} decoder blocks; choose a layer from zero to {len(blocks)-1}.')
    grouped={role:[] for role in data.ROLES}; alignment=[]; binding=None; total_bytes=0; count=0
    for index,row in enumerate(chosen):
        if config.rendering=='chatTemplate':
            text=tokenizer.apply_chat_template([{'role':'user','content':row['text']}],tokenize=False,add_generation_prompt=True)
            encoded=tokenizer(text,add_special_tokens=False,truncation=True,max_length=config.maxSeqLen,return_tensors='pt')
        else:encoded=tokenizer(row['text'],truncation=True,max_length=config.maxSeqLen,return_tensors='pt')
        if 'attention_mask' not in encoded:encoded['attention_mask']=encoded['input_ids'].new_ones(encoded['input_ids'].shape)
        encoded={k:v.to(config.device) for k,v in encoded.items() if k in ('input_ids','attention_mask')}
        positions,values,precision=observe(model,blocks[config.layer],encoded,config)
        current={'modelID':config.modelID,'revision':config.revision,'substrate':'pytorch',
            'coordinateConvention':'hf-decoder-block-v1/'+path,'precision':precision,'hiddenSize':len(values[0]),
            'site':{'kind':config.site,'layer':config.layer},
            'reading':{'rendering':config.rendering,'position':config.position,'population':config.population},
            'tokenizerSHA256':runtime['tokenizerSHA256'],'templateSHA256':runtime['templateSHA256']}
        if binding is not None and binding!=current:raise ProbeError('The activation binding changed during capture.')
        binding=artifact.validate_input(current)
        for position,value in zip(positions,values):
            count+=1
            if count>config.maxRecords:raise ProbeError('Capture exceeded maxRecords. Increase the reviewed cap or capture fewer examples/positions; no rows were silently dropped.')
            record={'id':archives.digest([row['id'],position]),'group':row['group'],'sourceSHA256':row['sourceSHA256'],'label':row['label'],'activation':value}
            total_bytes+=len(archives.encoded(record))
            if total_bytes>data.MAX_BYTES:raise ProbeError('Captured activation JSON exceeds 64 MiB. Reduce the pilot size or use fewer positions.')
            grouped[row['split']].append(record)
        alignment.append({'id':row['id'],'split':row['split'],'group':row['group'],'positions':positions,
                          'atTokenLimit':encoded['input_ids'].shape[1]==config.maxSeqLen,
                          'tokenIDs':encoded['input_ids'][0].cpu().tolist(),'sourceSHA256':row['sourceSHA256']})
        log(f'Captured example {index+1}/{len(chosen)} at {len(positions)} position(s).')
    files={}
    for role,records in grouped.items():
        if records:files[role]=data.save(run,role+'-activations.json',{'artifactType':'activation-dataset','schemaVersion':1,'input':binding,'rows':records,
            'provenance':{'config':config.to_dict(),'runtime':runtime,'role':role,'examples':config.examples,'recipe':'probe-capture-v1'}})
    report={'schemaVersion':1,'operation':'probe-capture','config':config.to_dict(),'runtime':runtime,
        'input':binding,'files':files,'counts':{r:data.labels(v) for r,v in grouped.items()},'rowsAvailable':len(rows),
        'rowsCaptured':len(chosen),'alignment':alignment,'activationJSONBytes':total_bytes,
        'limitations':['Text is replayed once without generation or interventions. This does not measure a live response.',
                      'Tokenization is capped at maxSeqLen; inspect token IDs and positions in this report. Raw text uses native special tokens; chat uses one user message and a generation cue.',
                      'Group hashing is 60/20/20 in expectation; small groups may leave a split or class empty. Fitting validates its actual class counts.']}
    if config.position == 'eachNonPadding':
        report['limitations'].append('Each observed prefix inherits the whole example label. Decide whether that label is meaningful before the full example has been read.')
    data.save(run,'capture-report.json',report);(run/'COMPLETED').write_text('probe-capture\n')
    return {'runDirectory':str(run),'reportPath':str(run/'capture-report.json'),'datasets':files}


def preflight(config, root):
    rows=data.text_rows(data.read(config.examples,root),config.splitPolicy,config.seed)
    return {'examples':min(len(rows),config.maxExamples),'maxRecords':config.maxRecords,
            'maxActivationJSONBytes':data.MAX_BYTES,'modelLoaded':False,
            'limitations':['Dataset limits do not bound model memory. Multimodal checkpoints can load vision components even for text-only capture; allow memory for the full checkpoint.']}
