#!/usr/bin/env python3
"""Initialize, hardware-test, or train the separate CrachoLM 250M model.

Initialization and the synthetic memory probe do not train a useful assistant.
Real training requires explicit train and validation files. Existing output
directories are never reused; resumed training writes into a fresh directory.
"""
import argparse
import gc
import hashlib
import json
import math
import random
import shutil
import time
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config_250m import make_250m_config
from src.bpe_tokenizer import CrachoBPETokenizer
from src.checkpoint import get_model_config_from_checkpoint, load_checkpoint_file
from src.data_general import CrachoGeneralDataset
from src.inference import load_model_for_inference
from src.model import CrachoLM
from src.resumable_training import ResumableTrainerMixin, atomic_torch_save, training_loader
from src.trainer import CrachoTrainer

ROOT = Path(__file__).resolve().parent
INITIAL = ROOT / "checkpoints_250m_initial_20260928"
TOKENIZER = ROOT / "checkpoints_capability_mixed_20260928/tokenizer_bpe.json"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fresh_directory(path):
    path = Path(path).resolve()
    # Even an empty existing directory is protected, including old model dirs.
    path.mkdir(parents=True, exist_ok=False)
    return path


def check_config(model):
    expected = asdict(make_250m_config().model)
    if asdict(model.config) != expected:
        raise ValueError("This command only accepts the separate 250M architecture.")
    return model.get_num_params()


def optimizer_for(model, lr=3e-4, weight_decay=0.1):
    groups = [
        {"params": [p for p in model.parameters() if p.ndim >= 2], "weight_decay": weight_decay},
        {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
    ]
    # foreach's tensor-list intermediates would consume roughly another 1GB.
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), foreach=False)


class LowMemoryTrainer(ResumableTrainerMixin, CrachoTrainer):
    def _create_optimizer(self):
        return optimizer_for(self.model, self.config.training.learning_rate,
                             self.config.training.weight_decay)

    def load_checkpoint(self, filepath):
        # Loading a full optimizer checkpoint onto CUDA before restoring it
        # would temporarily duplicate multiple gigabytes of GPU tensors.
        state = load_checkpoint_file(str(filepath), torch.device("cpu"))
        if asdict(get_model_config_from_checkpoint(state)) != asdict(self.model.config):
            raise ValueError("Resume architecture differs from the 250M preset.")
        if not state.get("optimizer_state_dict"):
            raise ValueError("Use an actual training checkpoint for --resume.")
        self.model.load_state_dict(state["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        if self.scaler is not None and state.get("scaler_state_dict"):
            self.scaler.load_state_dict(state["scaler_state_dict"])
        self.start_epoch = state["epoch"]
        self.global_step = state["global_step"]
        self.tokens_processed = state["tokens_processed"]
        self.best_val_loss = state["best_val_loss"]
        self.restore_progress(state)
        if "rng_state" in state:
            torch.set_rng_state(state["rng_state"])
        if self.device.type == "cuda" and "cuda_rng_state" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        del state
        gc.collect()
        print(f"Resumed epoch {self.current_epoch}, next batch {self.next_batch}, "
              f"successful updates {self.global_step}", flush=True)

    def save_checkpoint(self, filepath, epoch, val_loss, is_best=False):
        from src.checkpoint import build_checkpoint
        state = build_checkpoint(self.model, self.optimizer, self.scheduler,
                                 self.config, epoch, self.global_step,
                                 self.tokens_processed, val_loss, self.best_val_loss)
        state.update(checkpoint_kind="training", rng_state=torch.get_rng_state(),
                     tokenizer_sha256=sha256(Path(filepath).parent / "tokenizer_bpe.json"))
        progress = self.progress_state()
        if progress is not None:
            state["training_progress"] = progress
        if self.scaler is not None:
            state["scaler_state_dict"] = self.scaler.state_dict()
        if self.device.type == "cuda":
            state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        destination = Path(filepath)
        atomic_torch_save(state, destination)
        validation = "not evaluated yet" if val_loss is None else f"{val_loss:.4f}"
        print(f"Saved {destination}: update {self.global_step}; validation {validation}", flush=True)


def initialize(args):
    tokenizer = CrachoBPETokenizer.load(str(TOKENIZER))
    config = make_250m_config(tokenizer.vocab_size)
    output = fresh_directory(args.output)
    config.system.checkpoints_dir = str(output)
    model = CrachoLM(config.model)
    count = check_config(model)
    shutil.copy2(TOKENIZER, output / "tokenizer_bpe.json")
    state = {
        "model_state_dict": model.state_dict(), "config_dict": asdict(config),
        "epoch": 0, "global_step": 0, "tokens_processed": 0,
        "checkpoint_kind": "random_initialization_untrained",
        "tokenizer_sha256": sha256(output / "tokenizer_bpe.json"),
        "parameters": count,
    }
    path = output / "initial_model.pt"
    temporary = path.with_suffix(".pt.partial")
    torch.save(state, temporary)
    temporary.replace(path)
    metadata = {k: v for k, v in state.items() if k != "model_state_dict"}
    metadata.update(torch_version=torch.__version__, seed=args.seed,
                    checkpoint=str(path), sha256=sha256(path))
    (output / "model_info.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


def probe(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is needed for the requested memory probe.")
    if args.steps < 2:
        raise ValueError("At least two successful updates are needed to test optimizer-state memory.")
    report = Path(args.report).resolve()
    if report.exists():
        raise FileExistsError(report)
    report.parent.mkdir(parents=True, exist_ok=True)
    model, tokenizer, info = load_model_for_inference(args.checkpoint, device="cpu")
    count = check_config(model)
    if not 1 <= args.seq_len <= model.config.max_seq_len:
        raise ValueError("Invalid probe sequence length.")
    model = model.to("cuda").train()
    optimizer = optimizer_for(model)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    torch.cuda.reset_peak_memory_stats()
    losses, attempts = [], 0
    torch.cuda.synchronize()
    started = time.monotonic()
    while len(losses) < args.steps:
        attempts += 1
        if attempts > args.steps + 20:
            raise RuntimeError("Too many AMP-overflow skips; probe failed.")
        optimizer.zero_grad(set_to_none=True)
        # Random tokens only: this is a full forward/backward/AdamW memory test.
        ids = torch.randint(4, tokenizer.vocab_size, (1, args.seq_len + 1), device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            _, loss = model(ids[:, :-1].contiguous(), ids[:, 1:].contiguous())
        if not math.isfinite(loss.item()):
            raise RuntimeError("Non-finite probe loss.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= old_scale:
            losses.append(loss.item())
            print(f"GPU probe update {len(losses)}/{args.steps}: loss={loss.item():.4f}, gradient_norm={norm.item():.4f}", flush=True)
    torch.cuda.synchronize()
    result = dict(parameters=count, gpu=torch.cuda.get_device_name(),
                  torch_version=torch.__version__, cuda_version=torch.version.cuda,
                  micro_batch=1, sequence_length=args.seq_len, precision="FP16 autocast; FP32 weights/AdamW states",
                  successful_optimizer_updates=len(losses), attempts=attempts,
                  seconds=time.monotonic()-started,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                  losses=losses, saved_probe_weights=False, checkpoint=info["checkpoint"],
                  note="Synthetic memory test only. Updated weights discarded; initial checkpoint unchanged.")
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("This preset is configured for CUDA training.")
    if min(args.epochs, args.grad_accum, args.seq_len) < 1 or args.seq_len > 512:
        raise ValueError("Epochs/accumulation must be positive; sequence length must be 1..512.")
    if args.checkpoint_every < 1:
        raise ValueError("Checkpoint interval must be positive.")
    paths = [Path(args.train_text).resolve(), Path(args.validation_text).resolve()]
    hashes = [sha256(p) for p in paths]
    if paths[0] == paths[1] or hashes[0] == hashes[1]:
        raise ValueError("Training and held-out validation text must be different.")
    source = Path(args.resume or args.checkpoint).resolve()
    model, tokenizer, info = load_model_for_inference(source, device="cpu")
    check_config(model)
    config = make_250m_config(tokenizer.vocab_size)
    config.training.max_epochs = args.epochs
    config.training.grad_accum_steps = args.grad_accum
    tokenizer._tokenize_word = lru_cache(maxsize=65536)(tokenizer._tokenize_word)
    datasets, stats = [], []
    for path, fingerprint in zip(paths, hashes):
        tokens = tokenizer.encode(path.read_text(encoding="utf-8"), add_special_tokens=False)
        unknown = tokens.count(tokenizer.unk_id)
        if unknown / max(1, len(tokens)) > 0.005:
            raise ValueError(f"Too many unknown tokens in {path}.")
        datasets.append(CrachoGeneralDataset(tokens, args.seq_len))
        stats.append(dict(path=str(path), sha256=fingerprint, tokens=len(tokens), unknown=unknown))
    if args.resume:
        old_manifest = source.parent / "training_manifest.json"
        old = json.loads(old_manifest.read_text())
        if old["data"] != stats or old["sequence_length"] != args.seq_len or old["grad_accum"] != args.grad_accum:
            raise ValueError("Resume requires the original datasets, sequence length and accumulation.")
        if args.epochs != old["epochs"]:
            raise ValueError("Resume requires the original planned total --epochs to preserve the LR schedule.")
        if args.seed != old["seed"]:
            raise ValueError("Resume requires the original seed to preserve the data order.")
        state = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
        if state["epoch"] >= args.epochs:
            raise ValueError("This run already completed all planned epochs.")
        del state
    output = fresh_directory(args.output)
    config.system.checkpoints_dir = str(output)
    shutil.copy2(info["tokenizer"], output / "tokenizer_bpe.json")
    manifest = dict(source_checkpoint=str(source), data=stats, epochs=args.epochs,
                    sequence_length=args.seq_len, grad_accum=args.grad_accum,
                    seed=args.seed, model_config=asdict(config.model), torch_version=torch.__version__,
                    checkpoint_every=args.checkpoint_every,
                    note="Next-token pretraining; this is not instruction tuning or an intelligence benchmark.")
    (output / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    loaders = [training_loader(datasets[0], batch_size=1, seed=args.seed),
               DataLoader(datasets[1], batch_size=1, shuffle=False, num_workers=0,
                          generator=torch.Generator().manual_seed(args.seed))]
    trainer = LowMemoryTrainer(model, config, *loaders, tokenizer, torch.device("cuda"),
                               resume_checkpoint_path=str(source) if args.resume else None,
                               checkpoint_every=args.checkpoint_every)
    trainer.train()


def resume(args):
    """Reuse the saved run's data paths and settings; only output is new."""
    source = Path(args.checkpoint).resolve()
    old = json.loads((source.parent / "training_manifest.json").read_text())
    args.resume = source
    args.train_text, args.validation_text = [item["path"] for item in old["data"]]
    args.epochs, args.grad_accum = old["epochs"], old["grad_accum"]
    args.seq_len, args.seed = old["sequence_length"], old["seed"]
    args.checkpoint_every = old.get("checkpoint_every", 100)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Save fresh, untrained 250M weights separately.")
    init.add_argument("--output", type=Path, default=INITIAL)
    check = commands.add_parser("probe", help="Discardable CUDA forward/backward/AdamW test.")
    check.add_argument("--checkpoint", type=Path, default=INITIAL / "initial_model.pt")
    check.add_argument("--steps", type=int, default=2)
    check.add_argument("--seq-len", type=int, default=512)
    check.add_argument("--report", type=Path, required=True)
    fit = commands.add_parser("train", help="Pretrain on explicitly supplied separate data files.")
    fit.add_argument("--checkpoint", type=Path, default=INITIAL / "initial_model.pt")
    fit.add_argument("--resume", type=Path)
    fit.add_argument("--train-text", required=True)
    fit.add_argument("--validation-text", required=True)
    fit.add_argument("--output", type=Path, required=True)
    fit.add_argument("--epochs", type=int, default=1, help="Total planned epochs, including resumed epochs.")
    fit.add_argument("--grad-accum", type=int, default=16)
    fit.add_argument("--seq-len", type=int, default=512)
    fit.add_argument("--checkpoint-every", type=int, default=100,
                     help="Refresh last.pt every N successful updates, plus on Ctrl+C and epoch end.")
    continuation = commands.add_parser("resume", help="Continue a saved run using its original settings.")
    continuation.add_argument("--checkpoint", type=Path, required=True)
    continuation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    {"init": initialize, "probe": probe, "train": train, "resume": resume}[args.command](args)


if __name__ == "__main__":
    main()
