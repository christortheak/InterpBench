"""Generation kwargs contract: every model.generate call carries an explicit
all-ones attention mask (batch of one, no padding — identical behavior, no HF
'attention mask is not set' warning when pad == eos). No model, no GPU."""

from types import SimpleNamespace

import torch

from steerlab_server.experiment.generate import _generation_kwargs


def test_generation_kwargs_pass_all_ones_attention_mask():
    ids = torch.tensor([[5, 6, 7]])
    tok = SimpleNamespace(pad_token_id=None, eos_token_id=2)
    kwargs = _generation_kwargs(ids, 8, 0.0, tok)
    mask = kwargs["attention_mask"]
    assert mask.shape == ids.shape
    assert bool((mask == 1).all())
    assert kwargs["input_ids"] is ids
    assert kwargs["pad_token_id"] == 2  # eos reused as pad — the warning trigger
    assert kwargs["do_sample"] is False


def test_generation_kwargs_sampled_path_keeps_mask():
    ids = torch.tensor([[1, 2]])
    tok = SimpleNamespace(pad_token_id=0, eos_token_id=2)
    kwargs = _generation_kwargs(ids, 4, 0.7, tok)
    assert kwargs["do_sample"] is True
    assert kwargs["attention_mask"].shape == ids.shape
    assert bool((kwargs["attention_mask"] == 1).all())
