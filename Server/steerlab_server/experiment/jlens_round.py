"""Materialize a fixed global row budget as disjoint, independently resumable fits."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import uuid
from . import diagnostic_archives as archives, jlens_fit_selection as selection
from .jlens_fit import FitConfig, FitError, file_ref, read_pinned, corpus_rows


@dataclass(frozen=True)
class RoundConfig:
    modelID: str
    revision: str
    fittingRequest: dict
    shards: int = 4
    maxConcurrent: int = 1
    startRow: int = 0
    layout: str = 'interleaved'

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or value.keys() - cls.__dataclass_fields__.keys():
            raise FitError('Use the documented fitting-round fields.')
        try: cfg = cls(**{**value, 'fittingRequest': file_ref(value.get('fittingRequest'), 'fittingRequest')})
        except TypeError as exc: raise FitError('Supply modelID, revision, and a published fitting request.') from exc
        if any(type(v) is not int or v < 1 for v in (cfg.shards, cfg.maxConcurrent)) or cfg.maxConcurrent > cfg.shards:
            raise FitError('Choose positive shard and concurrency counts, with concurrency no greater than shard count and within site policy.')
        if type(cfg.startRow) is not int or cfg.startRow < 0 or cfg.layout not in ('interleaved','contiguous'):
            raise FitError('Choose a nonnegative starting row and interleaved or contiguous partitioning.')
        return cfg

    def to_dict(self): return asdict(self)


def prepare(config, root):
    document = json.loads(read_pinned(config.fittingRequest, root))
    if not isinstance(document, dict) or document.get('operation') != 'jlens-fit':
        raise FitError('Choose a published J-lens fitting request.')
    base = FitConfig.from_dict(document.get('parameters', {}).get('config'))
    if (base.modelID, base.revision) != (config.modelID, config.revision):
        raise FitError('The round and fitting request name different model versions.')
    if base.checkpoint or base.rowIndices is not None or base.shard is not None or getattr(base, 'stopping', None) is not None:
        raise FitError('Start a round from a fresh fixed-budget fit. Continue individual shards separately; stopping is assessed between rounds.')
    rows = corpus_rows(read_pinned(base.corpus, root))
    count = min(base.maxPrompts, len(rows) - config.startRow)
    if count < config.shards:
        raise FitError('The selected corpus range must give every shard at least one row.')
    children = []
    for index in range(config.shards):
        shard = dict(index=index, count=config.shards, startRow=config.startRow, rowCount=count, layout=config.layout)
        shard['planSHA256'] = selection.stamp(base, shard)
        chosen = selection.partition(config.startRow, count, config.shards, index, config.layout)
        child = FitConfig.from_dict({**base.to_dict(), 'maxPrompts':len(chosen), 'rowIndices':chosen, 'shard':shard})
        children.append({'operation':'jlens-fit','parameters':{'config':child.to_dict()}})
    from .jlens_fit_review import review
    costs = review(base.to_dict(), root)
    matrix = costs.get('estimate', {}).get('matrixSetBytes')
    result = {'schemaVersion':1, 'config':config.to_dict(), 'globalRowBudget':count,
              'rowBudgetIncludesSkipped':True, 'shards':children, 'fittingReview':costs,
              'minimumLensAndCheckpointBytes':2*matrix*config.shards if matrix else None,
              'guidance':'This is one global row budget, divided among shards. Each shard retains a lens and a checkpoint. Queue top-ups and continuation are explicit reviewed actions.'}
    return {**result,'planSHA256':archives.digest(result)}


def preflight(config, root): return prepare(config, root)


def materialize(config, *, root, log=print, on_run_created=None):
    plan = prepare(config, root)
    root = Path(root).resolve(strict=True)
    parent = archives.ordinary(root,'runs',missing=True); parent.mkdir(exist_ok=True)
    run = parent/('jlens-round-'+uuid.uuid4().hex); run.mkdir()
    if on_run_created: on_run_created(str(run))
    (run/'round-plan.json').write_bytes(archives.encoded(plan))
    for i, request in enumerate(plan['shards']):
        (run/f'shard-{i}.json').write_bytes(archives.encoded(request))
    (run/'COMPLETED').write_text('jlens-fit-round materialized\n')
    return {'runDirectory':str(run), 'roundPlanSHA256':archives.file_hash(run/'round-plan.json'),
            'roundState':'materializedOnly', 'shardCount':len(plan['shards']), 'globalRowBudget':plan['globalRowBudget'],
            'nextAction':'Review the fitting-round queue plan, then submit its pending shards. Materialization does not execute a fit.'}
