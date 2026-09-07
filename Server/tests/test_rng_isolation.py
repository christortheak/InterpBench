"""Per-record RNG isolation (2026-07-13): seeded generation runs inside
``torch.random.fork_rng`` with the ``manual_seed`` INSIDE the fork, so seeding
is generation-local — two concurrent seeded studies interleaving records draw
exactly what they would serially, and a seeded record never perturbs the
process-global RNG stream around it."""

import torch
import pytest

from steerlab_server.experiment import tasks
import steerlab_server.experiment.sampling as sampling


def _draw(seed: int, temperature: float = 0.7) -> list[float]:
    """One 'record': seed generation-locally, draw from the global RNG the
    way HF sampling does (torch.rand hits the same default CPU generator)."""
    with sampling.seeded_generation(temperature, seed):
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
    with sampling.seeded_generation(0.0, 42):
        pass  # greedy: no seeding, no fork
    assert torch.rand(3).tolist() == expected


def test_overlapping_worker_records_are_serialized():
    from concurrent.futures import ThreadPoolExecutor
    import time
    def draw(seed):
        with sampling.seeded_generation(.7, seed):
            first = torch.rand(3).tolist()
            time.sleep(.01)  # Give the other worker a chance to enter its scope.
            return first + torch.rand(3).tolist()
    expected = [draw(10), draw(20)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(draw, [10, 20])) == expected


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='actual MPS required')
def test_mps_state_restores_on_success_failure_and_nested_scopes():
    original = torch.mps.get_rng_state().clone()
    try:
        for fails in (False, True):
            before = torch.mps.get_rng_state().clone()
            try:
                with sampling.seeded_generation(.7, 99):
                    first = torch.rand(16, device='mps').cpu()
                    with sampling.seeded_generation(.7, 12):
                        torch.rand(32, device='mps').cpu()
                    second = torch.rand(16, device='mps').cpu()
                    if fails:
                        raise ValueError('fixture')
            except ValueError:
                pass
            assert torch.equal(before, torch.mps.get_rng_state())
            with sampling.seeded_generation(.7, 99):
                assert torch.equal(first, torch.rand(16, device='mps').cpu())
                assert torch.equal(second, torch.rand(16, device='mps').cpu())
    finally:
        torch.mps.set_rng_state(original)
