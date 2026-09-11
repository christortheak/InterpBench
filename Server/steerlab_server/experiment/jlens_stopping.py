"""Optional consecutive-update stopping; never a test of readout accuracy."""
import math
from .jlens_fit import FitError

STATISTIC = 'max-layer-relative-frobenius-running-mean-change'


def validate(rule):
    if rule is None: return
    if (not isinstance(rule,dict) or set(rule)!={'threshold','window','minPrompts'}
            or type(rule['threshold']) not in (int,float) or not math.isfinite(rule['threshold']) or rule['threshold'] <= 0
            or any(type(rule[k]) is not int or rule[k]<1 for k in ('window','minPrompts'))):
        raise FitError('Stopping requires a positive finite threshold, a positive consecutive-update window, and a minimum fitted-prompt count.')


class Control:
    def __init__(self, rule, saved=None, count=0):
        validate(rule)
        self.rule, self.count, self.values, self.available = rule,count,[],False
        self.last_value = None
        if rule is not None and saved is not None:
            if (not isinstance(saved,dict) or set(saved)!={'statistic','values','count','lastValue'} or saved['statistic']!=STATISTIC
                    or type(saved['count']) is not int or saved['count']!=count or not isinstance(saved['values'],list) or len(saved['values'])>min(rule['window'],max(0,count-1))
                    or any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<rule['threshold'] for v in saved['values'])):
                raise FitError('Checkpoint stopping history is invalid; retain the matching checkpoint state.')
            last=saved['lastValue']
            if last is not None and (type(last) not in (int,float) or not math.isfinite(last) or last<0):
                raise FitError('Checkpoint stopping statistic is invalid.')
            if saved['values'] and last!=saved['values'][-1]:
                raise FitError('Checkpoint stopping window and last statistic differ.')
            self.values=list(saved['values']);self.last_value=last
        elif rule is not None and count:
            raise FitError('A continuation with stopping needs the original stopping history; do not infer a window from a final mean.')

    def before(self, sums, count):
        self.available = count>0 and all(math.isfinite(float((v/count).norm())) and float((v/count).norm())>0 for v in sums.values())

    def observe(self, event, count):
        if event['status']!='fitted': return
        value=event.get('meanRelativeChangeMax')
        valid=self.available and isinstance(value,(float,int)) and math.isfinite(value)
        event['stoppingStatisticAvailable']=valid
        if not valid: event['meanRelativeChangeMax']=None
        self.last_value=value if valid else None
        self.count=count
        if self.rule is None: return
        if valid and value<self.rule['threshold']:
            self.values.append(value);self.values=self.values[-self.rule['window']:]
        else:self.values=[]

    @property
    def reached(self):
        return self.rule is not None and self.count>=self.rule['minPrompts'] and len(self.values)==self.rule['window']

    @property
    def state(self):
        return None if self.rule is None else {'statistic':STATISTIC,'values':self.values,'count':self.count,'lastValue':self.last_value}

    def report(self):
        return {'rule':self.rule,'statistic':STATISTIC,'endedBy':'stabilityWindow' if self.reached else 'rowBudget',
                'windowSemantics':'N consecutive available fitted-row updates strictly below threshold; skipped rows do not advance the window; an unavailable statistic clears it.',
                'state':self.state,'finalValue':self.last_value,'limitation':'A stable mean is not evidence of readout accuracy. Zero-norm layer means make the statistic unavailable; fitting continues to its row cap.'}
