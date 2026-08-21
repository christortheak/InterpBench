"""Test-only subprocess runner: drives the REAL checkpoint/resume helpers
(:mod:`steerlab_server.experiment.resume`) with a fake "generation" function,
so the signal path can be exercised end-to-end without a model.

    python checkpoint_harness.py <run_dir> <total_records> <seconds_per_record>

Behavior mirrors the headless CLI run path exactly:
- installs SIGUSR1/SIGTERM checkpoint handlers,
- resumes when the run directory carries resume-state.json,
- exits 0 on completion (report.json written, resume-state cleared),
- exits ``resume.CHECKPOINT_EXIT_CODE`` (85) when a signal parked the run.
"""

import json
import os
import sys
import time

from steerlab_server.experiment import resume


def fake_generation(index: int, delay: float) -> dict:
    time.sleep(delay)  # stands in for the forward pass
    return {
        "condition": "cond",
        "promptIndex": index,
        "promptID": f"prompt-{index}",
        "sampleIndex": 0,
        "seed": index,
        "output": f"deterministic text {index}",
        "wordCount": 3,
        "distinct2": 1.0,
    }


def main() -> int:
    run_dir, total, delay = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    os.makedirs(run_dir, exist_ok=True)
    flag = resume.CheckpointFlag().install()
    resuming = resume.is_resumable(run_dir)
    if resume.is_complete(run_dir):
        raise resume.ResumeError(f"{run_dir} is complete — refusing to touch it")
    writer = resume.GenerationWriter(run_dir, verb="run", checkpoint=flag,
                                     resume=resuming)
    try:
        for index in range(total):
            if writer.skip("cond", index, f"prompt-{index}", 0,
                           resume.KIND_SAMPLED):
                continue
            writer.emit(fake_generation(index, delay))
    except resume.CheckpointRequested:
        return resume.CHECKPOINT_EXIT_CODE
    finally:
        writer.close()
    with open(os.path.join(run_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump({"complete": True, "records": len(writer.records)}, fh)
    resume.clear_state(run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
