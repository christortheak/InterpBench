#!/usr/bin/env python3
"""Prove managed adapters leave the existing scientific owners' ASTs unchanged."""
import argparse
import ast
from pathlib import Path
import subprocess
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base',default='6a94a8f')
args=parser.parse_args()
root=Path(__file__).resolve().parents[2]
modules=['optvec_train','optvec_eval','optvec_geometry','optvec_interpret','optvec_gradient','optvec_jspace','optvec_campaign','family_report','sae_qualification','sae_candidates','analysis_workflow']
def tree(source):return ast.dump(ast.parse(source),include_attributes=False)
for name in modules:
    path='Server/steerlab_server/experiment/'+name+'.py'
    before=subprocess.check_output(['git','show',args.base+':'+path],cwd=root,text=True)
    after=(root/path).read_text()
    assert tree(before)==tree(after),path+' changed scientific AST'
    assert tree(after+'\nAUDIT_NEGATIVE_CONTROL = True\n')!=tree(before),'negative control did not detect a changed body'
print(str(len(modules))+' scientific owner ASTs unchanged; negative controls passed.')

for name in ('lens_store', 'qualification', 'g0'):
    path='Server/steerlab_server/jlens/'+name+'.py'
    before=subprocess.check_output(['git','show',args.base+':'+path],cwd=root,text=True)
    after=(root/path).read_text()
    normalized=after.replace('from . import artifact_paths\n','').replace('artifact_paths.converted_file(record.lensID, record.converted.path, root)','paths.resolve(record.converted.path, root)')
    assert tree(normalized)==tree(before),path+' differs beyond the declared path resolver substitution'
    assert tree(normalized+'\nAUDIT_NEGATIVE_CONTROL = True\n')!=tree(before)
print('Three lens consumers differ only by the declared path resolver and its import; negative controls passed.')
