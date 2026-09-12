#!/usr/bin/env python3
"""Prove extraction of the input validator preserves every prior artifact body."""
import ast
import copy
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
PATH='Server/steerlab_server/experiment/probe_artifacts.py'
BASE='d25c9a2'


def audit(before,after):
    old,new=ast.parse(before),ast.parse(after)
    helper=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='validate_input')
    assert ast.dump(helper.body[0])==ast.dump(ast.parse('finite_json(value)').body[0])
    assert ast.dump(helper.body[-1])==ast.dump(ast.parse('return copy.deepcopy(binding)').body[0])
    body=copy.deepcopy(helper.body[1:-1])
    assert ast.dump(body[0].value.args[0])==ast.dump(ast.Name(id='value',ctx=ast.Load()))
    body[0].value.args[0]=ast.parse("d['input']",mode='eval').body
    validator=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='validate')
    expected=ast.dump(ast.parse("binding = validate_input(d['input'])").body[0])
    matches=[i for i,n in enumerate(validator.body) if ast.dump(n)==expected]
    assert len(matches)==1
    i=matches[0];validator.body[i:i+1]=body
    new.body.remove(helper)
    assert ast.dump(old)==ast.dump(new),'Artifact semantics changed beyond input-validator extraction'


before=subprocess.check_output(['git','show',BASE+':'+PATH],cwd=ROOT,text=True)
after=(ROOT/PATH).read_text();audit(before,after)
for source,target in [("('residualPre', 'residualPost')","('residualPost',)"),("return float(value)","return int(value)")]:
    assert after.count(source)==1
    try:audit(before,after.replace(source,target))
    except AssertionError:pass
    else:raise AssertionError('Semantic mutation control accepted')
print('Probe artifact bodies preserved after input-validator extraction; site and numerical mutation controls rejected.')
