"""Lossless panel authoring decode defaults shared with the Mac document model.

Execution validity is still decided by multi_agent.validate. These checks avoid
coercing malformed JSON into a different panel on the lightweight client.
"""
from copy import deepcopy
from . import authoring_files as files


def normalized(document: dict) -> dict:
    panel = deepcopy(document)
    def fields(obj, rules):
        if not isinstance(obj, dict):
            files.refuse('A panel, seat, turn or contract must be an object.')
        for key, kind in rules.items():
            if key not in obj or not isinstance(obj[key], kind) or (kind in (int, float) and isinstance(obj[key], bool)):
                files.refuse(f"Panel field '{key}' has a missing or invalid value.")
    def defaults(obj, values):
        for key, value in values.items():
            if obj.get(key) is None:
                obj[key] = deepcopy(value)
    fields(panel, {'name': str})
    defaults(panel, {'description':'', 'baseModelID':'', 'sharedMaterials':'', 'temperature':0,
                     'maxTokens':2048, 'agents':[], 'turns':[]})
    fields(panel, {'description':str, 'baseModelID':str, 'sharedMaterials':str, 'agents':list, 'turns':list, 'maxTokens':int})
    if type(panel['temperature']) not in (int, float):
        files.refuse('Panel temperature must be a number.')
    for seat in panel['agents']:
        fields(seat, {'id':str, 'name':str, 'baseModelID':str, 'systemPrompt':str})
        for key in ('variantArtifactPath','variantArtifactHash','role'):
            if seat.get(key) is None:
                seat.pop(key,None)
            elif not isinstance(seat[key],str):
                files.refuse(f"Seat {key} must be a string.")
        if 'role' in seat and not seat['role'].strip():
            seat.pop('role')
    for turn in panel['turns']:
        fields(turn, {'id':str, 'title':str, 'speakerAgentID':str, 'outputLabel':str, 'routing':str,
                      'routedAgentIDs':list, 'includeScenarioMaterials':bool, 'includeSpeakerContext':bool})
        defaults(turn, {'promptTemplate':''})
        fields(turn, {'promptTemplate':str})
        if any(not isinstance(v,str) for v in turn['routedAgentIDs']):
            files.refuse('Routed agent IDs must be strings.')
        if turn['routing'] not in ('all','speakerOnly','selected','none'):
            files.refuse('Unknown panel routing declaration.')
        for key in ('maxTokens','endpoint','contract','acknowledgedInputs'):
            if turn.get(key) is None:
                turn.pop(key,None)
        if 'maxTokens' in turn and type(turn['maxTokens']) is not int:
            files.refuse('Turn maxTokens must be an integer.')
        if 'acknowledgedInputs' in turn:
            labels=turn['acknowledgedInputs']
            if not isinstance(labels,list) or any(not isinstance(v,str) for v in labels):
                files.refuse('acknowledgedInputs must be a list of strings.')
            if not labels:
                turn.pop('acknowledgedInputs')
        if 'contract' in turn:
            contract=turn['contract']
            fields(contract,{})
            defaults(contract, {'stage':'','task':'','format':'','inputs':[],'ownVoice':True,'materialsTitle':''})
            fields(contract, {'stage':str,'task':str,'format':str,'inputs':list,'ownVoice':bool,'materialsTitle':str})
            if any(not isinstance(v,str) for v in contract['inputs']):
                files.refuse('Contract inputs must be output-label strings.')
            contract['materialsTitle']=contract['materialsTitle'].strip() or 'SHARED MATERIALS'
    if panel.get('schemaVersion') is None:
        panel['schemaVersion']=2 if any(t.get('contract') is not None for t in panel['turns']) else 1
    fields(panel, {'schemaVersion':int})
    return panel
