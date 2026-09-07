"""An operation can add an input role without changing another operation."""
from steerlab_server.experiment import managed_inputs
from steerlab_server.experiment.operation_bindings import INPUT_ROLES
from steerlab_server.experiment import diagnostic_archives as archives
import pytest


def test_operation_specific_input_role_is_included_and_isolated(tmp_path, monkeypatch):
    (tmp_path / 'observations.jsonl').write_text('{"id":"one"}\n')
    monkeypatch.setitem(INPUT_ROLES, 'optvec-family',
                        {**INPUT_ROLES['optvec-family'], 'observations': 'file'})
    config = {'observations': 'observations.jsonl'}
    inventory = managed_inputs.inventory('optvec-family', config, tmp_path)
    assert inventory == archives.snapshot(tmp_path, {'observations.jsonl'})
    with pytest.raises(archives.Refusal, match='no resolvable scientific inputs'):
        managed_inputs.inventory('optvec-geometry', config, tmp_path)
    (tmp_path / 'observations.jsonl').write_text('{"id":"changed"}\n')
    assert managed_inputs.inventory('optvec-family', config, tmp_path) != inventory
