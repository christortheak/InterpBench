"""Scheduler walltime for managed science jobs: execution shape, never scientific identity.

A walltime rides beside the request exactly as a GPU type does (reviewed at plan
time, bound into ``planSHA256`` through the resources block, refused above the
site cap) and is never written into the published request. When no walltime is
requested, an operation whose plan carries a bounded, calibrated workload gets a
default sized from that workload with margin so it can backfill; everything
else, including every ``jlens-fit`` shard, keeps the site default.
"""
import math
import re

from .executors import SlurmResources, _parse_walltime

#: ``HH:MM:SS`` (hours unbounded, as Slurm accepts) or ``D-HH:MM:SS``.
_SHAPE = re.compile(r'(?:(?P<days>\d{1,3})-)?(?P<hours>\d{1,4}):(?P<minutes>[0-5]\d):(?P<seconds>[0-5]\d)')

#: Default rule for an estimated operation: ``MARGIN_FACTOR × estimate +
#: LOAD_SECONDS`` (model load and staging), rounded up to ``ROUND_SECONDS``,
#: never below ``FLOOR_SECONDS``, never above the site cap.
MARGIN_FACTOR = 3
LOAD_SECONDS = 30 * 60
ROUND_SECONDS = 15 * 60
FLOOR_SECONDS = 60 * 60

#: Assessment allowance per unit of work, one unit being one held-out row read
#: through one source layer for one lens comparison (the readout, the two lens
#: matrix loads amortized over the rows, and the row's share of activation
#: capture). Calibrated on a recorded acceptance run: 16 rows × 63 source
#: layers × 1 comparison took five minutes on an H100 for a 27B model
#: INCLUDING model load, which is under 0.3 s per unit. One second per unit is
#: therefore already a threefold allowance before the rule's own margin; the
#: float32 readout option doubles the readout and doubles the units.
ASSESSMENT_SECONDS_PER_UNIT = 1.0

RULE = (f'{MARGIN_FACTOR} × estimate + {LOAD_SECONDS // 60} min for model load, rounded up to '
        f'{ROUND_SECONDS // 60} min, at least {FLOOR_SECONDS // 3600} h, at most the site cap')


def _refusal(reason, repair):
    from .scientific_execution import ScientificRefusal
    error = ScientificRefusal(reason)
    error.repair_action = repair
    return error


def parse(value):
    """Seconds for a requested walltime, refusing anything but the two documented shapes."""
    text = value.strip() if isinstance(value, str) else ''
    match = _SHAPE.fullmatch(text)
    if not match:
        raise _refusal('Walltime ' + repr(value) + ' is not HH:MM:SS or D-HH:MM:SS.',
                       'Request the walltime as HH:MM:SS (for example 02:30:00) or D-HH:MM:SS, or omit it for the operation default.')
    seconds = (int(match['days'] or 0) * 86400 + int(match['hours']) * 3600
               + int(match['minutes']) * 60 + int(match['seconds']))
    if seconds <= 0:
        raise _refusal('Walltime must be positive.', 'Request a positive walltime, or omit it for the operation default.')
    return seconds


def render(seconds):
    """Canonical ``HH:MM:SS`` (hours may exceed 24; Slurm accepts hours:minutes:seconds unbounded)."""
    seconds = int(seconds)
    return f'{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}'


def cap_seconds(resources):
    """The site cap: the configured default walltime (``STEERLAB_SLURM_WALLTIME``), in seconds."""
    text = (resources.walltime or '').strip()
    match = _SHAPE.fullmatch(text)
    if match:
        return parse(text)
    return int(_parse_walltime(text).total_seconds())


def validate(value, resources):
    """A requested walltime normalized to ``HH:MM:SS``, refused above the site cap."""
    seconds = parse(value)
    cap = cap_seconds(resources)
    if seconds > cap:
        raise _refusal('Walltime ' + value.strip() + ' exceeds the site cap of ' + render(cap) + '.',
                       'Request at most the site cap (' + render(cap) + '), or omit the walltime for the operation default.')
    return render(seconds)


def estimate(plan):
    """A calibrated runtime estimate for the reviewed plan, or None when the operation has none.

    Only ``jlens-fit-assess`` is estimated: its work is a bounded number of
    readouts with no backward passes and a recorded calibration. Fitting keeps
    the long default by the researcher's rule (its checkpoints resume across
    walltime kills); benchmarks are the throughput measurement itself, so no
    estimate precedes them; batteries and stability sweeps carry no workload
    review at all.
    """
    operation = (plan.get('request') or {}).get('operation')
    review = plan.get('operationReview') or {}
    if operation != 'jlens-fit-assess' or not isinstance(review, dict):
        return None
    layers = review.get('sourceLayers')
    layer_count = len(layers) if isinstance(layers, (list, tuple)) else layers
    if type(layer_count) is not int or layer_count < 1:
        return None
    if isinstance(review.get('corpora'), list):
        rows = [corpus.get('rows') for corpus in review['corpora'] if isinstance(corpus, dict)]
        candidates = review.get('candidateLensIDs')
        candidate_count = len(candidates) if isinstance(candidates, list) else None
    else:
        rows, candidate_count = [review.get('rows')], 1
    if not rows or any(type(count) is not int or count < 0 for count in rows) or type(candidate_count) is not int or candidate_count < 1:
        return None
    config = ((plan.get('request') or {}).get('parameters') or {}).get('config') or {}
    readout_multiplier = 2 if config.get('readoutDtype') == 'float32' else 1
    units = sum(rows) * layer_count * candidate_count * readout_multiplier
    return {'operation': operation, 'seconds': int(math.ceil(units * ASSESSMENT_SECONDS_PER_UNIT)),
            'units': units, 'rows': sum(rows), 'sourceLayers': layer_count, 'candidates': candidate_count,
            'readoutMultiplier': readout_multiplier, 'secondsPerUnit': ASSESSMENT_SECONDS_PER_UNIT,
            'unit': 'one held-out row through one source layer for one lens comparison',
            'calibration': 'A recorded acceptance run of 16 rows × 63 source layers × 1 comparison took five minutes on an H100 for a 27B model including model load; the allowance is more than three times that rate before the margin.',
            'limitation': 'An allowance, not a measurement of this workload: larger models, longer rows, slower storage, or contention change the rate. The scheduler kills a job at its walltime; request a longer one explicitly when in doubt.'}


def default_seconds(estimate_seconds, cap):
    sized = MARGIN_FACTOR * estimate_seconds + LOAD_SECONDS
    rounded = int(math.ceil(sized / ROUND_SECONDS)) * ROUND_SECONDS
    return min(max(rounded, FLOOR_SECONDS), cap)


def review(plan, resources, requested=None):
    """The walltime this plan submits with and the rule that chose it.

    Mutates ``resources.walltime`` and returns ``{'walltime', 'basis', ...}``;
    ``basis`` is ``requested``, ``estimated``, or ``siteDefault``.
    """
    site_default = resources.walltime
    cap = cap_seconds(resources)
    result = {'siteDefault': site_default, 'cap': render(cap)}
    if requested is not None:
        resources.walltime = validate(requested, resources)
        # Only the canonical form is recorded, so 0-02:30:00 and 02:30:00 review
        # to the same plan hash and either spelling submits it.
        return {**result, 'walltime': resources.walltime, 'basis': 'requested',
                'summary': 'Requested explicitly; bound into the plan hash. Anything that runs longer is killed by the scheduler at this limit.'}
    estimated = estimate(plan)
    if estimated is None:
        return {**result, 'walltime': site_default, 'basis': 'siteDefault',
                'summary': 'No calibrated estimate exists for this operation, so the site default applies. Request a shorter walltime explicitly for a job known to be short.'}
    seconds = default_seconds(estimated['seconds'], cap)
    if seconds >= cap:
        return {**result, 'walltime': site_default, 'basis': 'siteDefault', 'estimate': estimated, 'rule': RULE,
                'summary': 'The estimate with margin reaches the site cap, so the site default applies.'}
    resources.walltime = render(seconds)
    return {**result, 'walltime': resources.walltime, 'basis': 'estimated', 'estimate': estimated, 'rule': RULE,
            'summary': f'Sized from the plan’s own workload: {RULE}. Request a walltime explicitly to override.'}


def walltime_of(resources_document):
    """The walltime a recorded resources block carries, if it is a Slurm block."""
    if not isinstance(resources_document, dict) or resources_document.get('executor') == 'local':
        return None
    try:
        return SlurmResources(**resources_document).walltime
    except TypeError:
        return resources_document.get('walltime')
