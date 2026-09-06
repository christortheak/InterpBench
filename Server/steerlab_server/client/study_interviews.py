"""The same shipped research interview the app and Mac CLI emit, without I/O writes."""
from importlib.resources import files


def prompt(intent: str) -> str:
    if intent not in ('conceptStudy', 'agentComparison', 'multiAgent'):
        raise ValueError('Choose conceptStudy, agentComparison or multiAgent.')
    return files('steerlab_server.experiment').joinpath(
        'seed/prompts/study-interviews', f'study-{intent}.md').read_text(encoding='utf-8').rstrip('\n')
