"""Per-record RNG isolation (2026-07-13): seeded generation runs inside
``torch.random.fork_rng`` with the ``manual_seed`` INSIDE the fork, so seeding
is generation-local — two concurrent seeded studies interleaving records draw
exactly what they would serially, and a seeded record never perturbs the
process-global RNG stream around it."""

import torch

from steerlab_server.experiment import tasks


def _draw(seed: int, temperature: float = 0.7) -> list[float]:
    """One 'record': seed generation-locally, draw from the global RNG the
    way HF sampling does (torch.rand hits the same default CPU generator)."""
    with tasks._seeded_generation(temperature, seed):
        return torch.rand(4).tolist()


def test_same_seed_reproduces_same_draws():
    assert _draw(123) == _draw(123)
    assert _draw(123) != _draw(124)


def test_interleaved_sequences_equal_serial_sequences():
    seeds_a = [11, 12, 13]
    seeds_b = [21, 22, 23]
    serial_a = [_draw(s) for s in seeds_a]
    serial_b = [_draw(s) for s in seeds_b]

    interleaved_a, interleaved_b = [], []
    for sa, sb in zip(seeds_a, seeds_b):
        interleaved_a.append(_draw(sa))
        interleaved_b.append(_draw(sb))

    assert interleaved_a == serial_a
    assert interleaved_b == serial_b


def test_seeded_generation_is_local_to_the_record():
    # The global stream is untouched by a seeded record in its midst.
    torch.manual_seed(777)
    torch.rand(2)
    expected = torch.rand(3).tolist()

    torch.manual_seed(777)
    torch.rand(2)
    _draw(999)  # a seeded record between the global draws
    assert torch.rand(3).tolist() == expected


def test_greedy_records_never_touch_the_rng():
    torch.manual_seed(555)
    expected = torch.rand(3).tolist()
    torch.manual_seed(555)
    with tasks._seeded_generation(0.0, 42):
        pass  # greedy: no seeding, no fork
    assert torch.rand(3).tolist() == expected
