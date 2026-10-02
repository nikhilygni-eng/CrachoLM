"""Temporary full-size checkpoint integration check; synthetic training is discarded."""
import gc
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path("/home/escanor/Downloads/CrachoLM")
sys.path.insert(0, str(ROOT))


def worker(arguments):
    from train_250m import LowMemoryTrainer, main
    original = LowMemoryTrainer._at_update_boundary
    def stop_after_one_update(self, did_step):
        if did_step:
            os.kill(os.getpid(), signal.SIGINT)
        return original(self, did_step)
    LowMemoryTrainer._at_update_boundary = stop_after_one_update
    sys.argv = ["train_250m.py", *arguments]
    main()


def parent(report_path):
    import torch
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="250m_stop_check_", dir=report_path.parent) as tmp:
        root = Path(tmp)
        for split, filename in [("train", "train.txt"), ("validation", "validation.txt")]:
            text = (ROOT / "data_general/fluency_stories" / filename).read_text()
            (root / filename).write_text("\n".join(text.splitlines()[:250]) + "\n")
        first = root / "first"
        resumed = root / "resumed"
        commands = [
            ["train", "--train-text", str(root / "train.txt"),
             "--validation-text", str(root / "validation.txt"),
             "--output", str(first), "--grad-accum", "16", "--seq-len", "512"],
            ["resume", "--checkpoint", str(first / "last.pt"), "--output", str(resumed)],
        ]
        summaries = []
        for directory, arguments in zip([first, resumed], commands):
            subprocess.run([sys.executable, "-u", __file__, "worker", *arguments], cwd=ROOT,
                           check=True, timeout=150)
            checkpoint = directory / "last.pt"
            state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
            summaries.append(dict(
                successful_updates=state["global_step"], tokens_processed=state["tokens_processed"],
                progress=state["training_progress"], checkpoint_bytes=checkpoint.stat().st_size,
                optimizer_state_entries=len(state["optimizer_state_dict"]["state"]),
                has_scaler="scaler_state_dict" in state,
                has_cuda_rng="cuda_rng_state" in state,
            ))
            del state
            gc.collect()
        assert summaries[0]["successful_updates"] == 1
        assert summaries[1]["successful_updates"] == 2
        assert summaries[0]["progress"]["next_batch"] == 16
        assert summaries[1]["progress"]["next_batch"] == 32
        assert summaries[1]["tokens_processed"] == 2 * summaries[0]["tokens_processed"]
        assert all(s["has_scaler"] and s["has_cuda_rng"] and s["optimizer_state_entries"] > 0 for s in summaries)
        result = dict(passed=True, parameters=250431488, sequence_length=512,
                      gradient_accumulation=16, runs=summaries,
                      note="Real SIGINT, full-size GPU checkpoint save and CLI resume tested. Temporary probe weights removed. Original checkpoints unchanged.")
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        worker(sys.argv[2:])
    else:
        parent(Path(sys.argv[1]).resolve())
