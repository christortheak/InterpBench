#!/usr/bin/env python3
"""Standalone reproduction for the per-generation memory growth seen on MPS.

Why this exists
---------------
A panel run on an M5 Pro climbed from ~13 GiB to 75 GiB and died with

    MPS backend out of memory (MPS allocated: 9.02 GiB,
    other allocations: 79.03 GiB, max allowed: 88.13 GiB)

Four hypotheses were wrong before this script existed — a growing KV cache,
an MLX model still resident, allocator accounting drift, and size-class
fragmentation — so it measures rather than assumes.

What it found (gemma-3-4b-it, bfloat16, MPS, climbing to a 10,207-token
context over twelve generations):

    sdpa, no trim      driver 8.1 -> 72.4 GiB   (footprint 74 GB: the OOM)
    sdpa, trim         driver 8.1 -> 62.4 GiB
    eager, trim        driver 8.1 ->  8.1 GiB   (footprint 15 GB)

Not a leak. ``allocated`` — torch's own count of LIVE memory — sat at
8.01 GiB, the model weights, in every row of every variant. Nor is it this
project's code: a bare ``model.generate()`` (``--path raw``) grows
identically to the full generation driver.

It is the attention kernel. On MPS, ``sdpa`` leaves a per-generation peak the
caching allocator will not return, so any run whose prompt length VARIES
ratchets the driver reservation up; a constant prompt length is flat forever
(``--prompt-growth 0``), and the ratchet happens whether prompts grow or
SHRINK (``--prompt-growth -900``), which is what rules out size-class
fragmentation. Under ``eager`` the same ``empty_cache()`` call returns the
whole peak (+13.46 GiB at turn 12) and the reservation never moves. Eager is
also ~10% FASTER here, so on this backend it is not a tradeoff.

Decode length is exonerated (tested 2026-07-29 after a field run spiked
from 51 to 96 GB during one final 2048-token draft): a FORCED 2048-token
decode at a 12K prompt (``--path raw --force-full-output``) reclaims
19.27 GiB against the 96-token control's 19.17 — decode adds ~0.1 GiB.
Whatever produced the field spike needs the long-lived multi-turn process
state (it ended holding ~45 GB more GPU pool than a fresh process doing
identical work, plus 23 GB of CPU heap); no fresh-process single mechanism
tested here reproduces it. The per-turn allocated/driver log from a real
run is the missing evidence — which is why dropping those log lines was a
bug worth its own fix.

Eager's own wall (calibrated at 16K/20K, same model and machine): the
during-turn transient is ~16 GiB x (N/10K)^2 — 40 GiB at 16K, 51 GiB at
20K, both fully returned by the trim afterward. On a 64 GB machine that
means comfortable to ~12K, pressured to ~20K, and past the MPS watermark
around ~24-25K. Long-materials runs need chunked prefill (linear in N for
a fixed chunk), not a bigger kernel budget.

12B calibration (2026-07-30, gemma-3-12b-it bf16, 22.7 GiB weights):
single-pass transient 7.52 GiB @ 5,005 tok and 13.59 @ 6,484 → 30-32.3
fit at 10K, exactly the 2x-the-4B the query-head arithmetic predicted —
the head-scaling fallback in memory_preflight is validated, not hoped.
Chunked at 16K: 13.03 GiB working set, driver flat (the x3.5 chunk
factor measures 2.49 here; kept at 3.5 — an advisory should over-warn).

None of this transfers to CUDA, where sdpa dispatches real flash-attention
kernels and eager would be a serious pessimization: any fix must be
device-conditional, and the cluster's behaviour has to be measured there.

What it separates
-----------------
The two numbers that matter are printed after every generation:

* ``allocated`` — what torch's caching allocator has handed out and still
  considers LIVE.
* ``driver`` — what the process has taken from the system.

Their divergence is the diagnosis. Driver climbing while allocated stays flat
is the allocator (or driver) failing to return blocks — a trim problem.  Both
climbing together is genuine retention of live tensors, and then the census
(``--census``) names the shapes and the objects holding them.

``--path`` bisects the stack, from the project's real generation driver down
to a bare ``model.generate``, so a leak can be attributed to project code or
to transformers:

    generate  the project's generate.generate() — streamer thread, hook
              session, logits guard: what a panel turn actually runs
    raw       bare model.generate() on the HF model, no project code

Usage (from the project root, NOT from Server/)::

    Server/.venv.nosync/bin/python Server/scripts/memory_growth_repro.py \
        --model google/gemma-3-4b-it --turns 12 --census

Loads a model and generates; it is not part of the test suite and is not
imported by anything.
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import time
import warnings

GIB = 1024 ** 3

# Import the package the same way the server does, from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--turns", type=int, default=12,
                        help="number of sequential generations")
    parser.add_argument("--prompt-start", type=int, default=300,
                        help="approx prompt tokens for turn 1")
    parser.add_argument("--prompt-growth", type=int, default=450,
                        help="approx prompt tokens added per turn (the panel's "
                             "context accumulation; 0 holds prompt length fixed, "
                             "which separates prompt length from call count)")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--path", choices=["generate", "raw"], default="generate")
    parser.add_argument("--trim", action="store_true",
                        help="call torch.mps.empty_cache() between generations, "
                             "as the panel runner does")
    parser.add_argument("--census", action="store_true",
                        help="after each generation, sum live torch tensors and "
                             "report the largest non-parameter ones")
    parser.add_argument("--force-full-output", action="store_true",
                        help="raw path only: set min_new_tokens = max_tokens "
                             "so decode runs its full length instead of "
                             "stopping at EOS. The panel's 2048-token drafts "
                             "were the regime the short-output calibration "
                             "never exercised — this flag exercises it")
    parser.add_argument("--attn", default=None,
                        help="override the attention implementation (sdpa, "
                             "eager, flash_attention_2). The peak is quadratic "
                             "in context length, which is the signature of a "
                             "kernel that MATERIALIZES the NxN score matrix — "
                             "so which kernel runs is the question this flag "
                             "asks")
    parser.add_argument("--reset-at", type=int, default=None,
                        help="turn index at which to reset the prompt to its "
                             "starting length (simulates the replicate boundary)")
    return parser.parse_args()


# ---------------------------------------------------------------- measurement


def accelerator_reading(device: str) -> tuple[float, float] | None:
    """(allocated, driver) in GiB, or None when the device has no probe."""
    import torch

    try:
        if device.startswith("mps") and hasattr(torch.mps, "driver_allocated_memory"):
            return (torch.mps.current_allocated_memory() / GIB,
                    torch.mps.driver_allocated_memory() / GIB)
        if device.startswith("cuda") and torch.cuda.is_available():
            return (torch.cuda.memory_allocated() / GIB,
                    torch.cuda.memory_reserved() / GIB)
    except Exception:
        return None
    return None


def process_footprint_gib() -> float | None:
    """macOS phys_footprint — what Activity Monitor calls Memory, and the
    number the OOM killer and the MPS watermark actually respond to. RSS is
    not it: unified-memory GPU allocations are counted in footprint."""
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(["footprint", "-p", str(os.getpid())],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if "phys_footprint" in line.lower() or "footprint" in line.lower():
            for token in line.replace("=", " ").split():
                cleaned = token.replace(",", "")
                if cleaned.replace(".", "", 1).isdigit():
                    value = float(cleaned)
                    if "MB" in line:
                        return value / 1024
                    if "GB" in line:
                        return value
    return None


def tensor_census(parameter_ids: set[int]) -> tuple[float, list[str]]:
    """(GiB held by live tensors, descriptions of the largest non-parameters).

    Walks the GC's object graph rather than asking torch, so it sees tensors
    the allocator counts as live and reports WHAT they are — the step past
    "memory is growing" to "this is growing".
    """
    import torch

    total = 0
    entries: list[tuple[int, str]] = []
    # Walking every live object touches deprecated module attributes that warn
    # on access (torch.distributed.reduce_op and friends) — noise from the
    # probe, not from the workload.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        objects = gc.get_objects()
    for obj in objects:
        try:
            if not isinstance(obj, torch.Tensor):
                continue
            nbytes = obj.untyped_storage().nbytes()
        except Exception:
            continue
        total += nbytes
        if id(obj) in parameter_ids or nbytes < 32 * 1024 * 1024:
            continue
        holders = sorted({type(r).__name__ for r in gc.get_referrers(obj)})[:3]
        entries.append((nbytes, f"{nbytes / GIB:.2f} GiB {tuple(obj.shape)} "
                                f"{obj.dtype} held by {holders}"))
    entries.sort(key=lambda pair: pair[0], reverse=True)
    return total / GIB, [text for _, text in entries[:5]]


# ---------------------------------------------------------------- the workload


def build_prompt(tokenizer, target_tokens: int) -> str:
    """A prompt of roughly ``target_tokens`` tokens, shaped like accumulated
    panel context (repeated prior-turn blocks) rather than random text."""
    block = ("[opening] Opening statement — Judge A\nThe question before this "
             "panel is whether the doctrine as applied produces a result the "
             "statute's text will bear, and if not, which of the two must give "
             "way. I would begin with the text.\n\n")
    unit = len(tokenizer(block, add_special_tokens=False)["input_ids"])
    repeats = max(1, target_tokens // max(1, unit))
    return (block * repeats) + "\nGive your view in two sentences."


def run_generate_path(model, prompt: str, max_tokens: int) -> str:
    from steerlab_server.experiment import generate as generate_mod

    return generate_mod.generate(model, prompt, model_id=model.model_id,
                                 max_tokens=max_tokens, temperature=0.0)


def run_raw_path(model, prompt: str, max_tokens: int,
                 force_full: bool = False) -> str:
    import torch

    tokenizer = model.tokenizer
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    extra = {"min_new_tokens": max_tokens} if force_full else {}
    with torch.no_grad():
        out = model.model.generate(input_ids=input_ids,
                                   attention_mask=torch.ones_like(input_ids),
                                   max_new_tokens=max_tokens, do_sample=False,
                                   use_cache=True,
                                   pad_token_id=tokenizer.pad_token_id
                                   or tokenizer.eos_token_id, **extra)
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)


def main() -> int:
    args = parse_args()
    import torch

    from steerlab_server.steering import model_loader

    print(f"loading {args.model} …", flush=True)
    model = model_loader.load(args.model, args.revision)
    device = str(model.device)
    if args.attn:
        model.model.set_attn_implementation(args.attn)
    effective = getattr(model.model.config, "_attn_implementation", "?")
    print(f"loaded on {device} ({model.dtype}), attention: {effective}", flush=True)

    parameter_ids = {id(p) for p in model.model.parameters()}
    for buffer in model.model.buffers():
        parameter_ids.add(id(buffer))

    baseline = accelerator_reading(device)
    if baseline:
        print(f"after load: allocated {baseline[0]:.2f} GiB, "
              f"driver {baseline[1]:.2f} GiB", flush=True)

    if args.path == "generate":
        runner = run_generate_path
    else:
        def runner(model, prompt, max_tokens):
            return run_raw_path(model, prompt, max_tokens,
                                force_full=args.force_full_output)
    previous = None
    for turn in range(1, args.turns + 1):
        step = turn - 1
        if args.reset_at is not None and turn >= args.reset_at:
            step = turn - args.reset_at
        target = args.prompt_start + step * args.prompt_growth
        prompt = build_prompt(model.tokenizer, target)
        actual_tokens = len(model.tokenizer(prompt, add_special_tokens=False)["input_ids"])

        started = time.monotonic()
        text = runner(model, prompt, args.max_tokens)
        elapsed = time.monotonic() - started

        # Measured across the trim, not just after it: whether
        # ``empty_cache()`` returns anything is the whole question about the
        # fix already shipped in the panel runner.
        reclaimed = None
        if args.trim and device.startswith("mps"):
            before = accelerator_reading(device)
            torch.mps.empty_cache()
            after = accelerator_reading(device)
            if before and after:
                reclaimed = before[1] - after[1]

        reading = accelerator_reading(device)
        line = (f"turn {turn:>3}  prompt {actual_tokens:>6} tok  "
                f"out {len(text):>5} ch  {elapsed:>6.1f}s")
        if reclaimed is not None:
            line += f"  trim {reclaimed:+.2f}"
        if reading:
            delta = "" if previous is None else \
                f"  Δdriver {reading[1] - previous[1]:+.2f}"
            line += (f"  allocated {reading[0]:>6.2f} GiB  "
                     f"driver {reading[1]:>6.2f} GiB{delta}")
            previous = reading
        footprint = process_footprint_gib()
        if footprint is not None:
            line += f"  footprint {footprint:>6.2f} GiB"
        print(line, flush=True)

        if args.census:
            held, biggest = tensor_census(parameter_ids)
            print(f"          live tensors {held:.2f} GiB", flush=True)
            for entry in biggest:
                print(f"            {entry}", flush=True)

    print("\nreading the result: driver climbing while allocated stays flat is "
          "the allocator not returning blocks; both climbing together is live "
          "retention — and the census names what holds it.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
