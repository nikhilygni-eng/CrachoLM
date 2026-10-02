#!/usr/bin/env python3
"""Bounded English-fluency experiment using CrachoLM's own trained weights.

Prepare a small public TinyStories sample:
  .venv/bin/python fluency_experiment.py prepare --train-stories 5000 --val-stories 500

Run a measured continuation experiment:
  .venv/bin/python fluency_experiment.py train --steps 200

The original checkpoints are read only. This is a story-fluency experiment,
not a promise of general question-answering accuracy.
"""
import argparse
import hashlib
import json
import math
import random
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET = "https://huggingface.co/datasets/roneneldan/TinyStories"
END = b"<|endoftext|>"
TRANSLATE = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u00a0": " ", "\u2026": "...",
})


def clean_story(text):
    return re.sub(r"\s+", " ", text.translate(TRANSLATE)).strip()


def story_stream(stream, max_bytes):
    """Yield complete UTF-8 stories and never consume beyond the byte limit."""
    buffer = b""
    remaining = max_bytes
    while remaining:
        block = stream.read(min(65536, remaining))
        if not block:
            break
        remaining -= len(block)
        buffer += block
        while END in buffer:
            document, buffer = buffer.split(END, 1)
            yield document.decode("utf-8", errors="strict")
        if len(buffer) > 1024 * 1024:
            raise ValueError("Missing story delimiter; unexpected dataset response.")


def unique_stories(stream, count, seen, max_bytes):
    written = 0
    for document in story_stream(stream, max_bytes):
        text = clean_story(document)
        if len(text) < 100 or len(text) > 10000:
            continue
        digest = hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        yield text
        written += 1
        if written == count:
            return
    if written < count:
        raise ValueError(f"Only found {written}/{count} complete unique stories within the download limit.")


def prepare(args):
    if args.train_stories < 1 or args.val_stories < 1:
        raise ValueError("Story counts must be positive.")
    output = Path(args.data_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to replace existing data: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seen = set()
    manifest = {
        "source": DATASET,
        "authors": ["Ronen Eldan", "Yuanzhi Li"],
        "paper": "https://arxiv.org/abs/2305.07759",
        "dataset_license": "CDLA-Sharing-1.0 (as listed on the source dataset card)",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "First N eligible unique stories from each official split; not a random corpus sample.",
        "normalization": "Unicode punctuation to ASCII equivalents; whitespace collapsed.",
        "splits": {},
    }
    for split, count in [("train", args.train_stories), ("valid", args.val_stories)]:
        url = f"{DATASET}/resolve/main/TinyStoriesV2-GPT4-{split}.txt"
        request = urllib.request.Request(url, headers={"User-Agent": "CrachoLM-fluency-experiment/1.0"})
        path = output / ("train.txt" if split == "train" else "validation.txt")
        temporary = path.with_suffix(".txt.partial")
        try:
            with urllib.request.urlopen(request, timeout=30) as response, temporary.open("w", encoding="utf-8") as handle:
                for number, text in enumerate(unique_stories(response, count, seen, args.max_download_mib * 1024**2), 1):
                    handle.write(text + "\n<eos>\n")
                    if number % 1000 == 0:
                        print(f"{split}: {number} stories prepared", flush=True)
            temporary.replace(path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        manifest["splits"][split] = {
            "url": url, "stories": count, "file": path.name,
            "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(f"{split}: saved {count} stories to {path}", flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Prepared disjoint training and validation stories. Original project data is unchanged.", flush=True)


def train(args):
    import gc
    import shutil
    from dataclasses import asdict
    from functools import lru_cache
    import torch
    from torch.utils.data import DataLoader, RandomSampler, Subset
    from config import Config, TrainingConfig
    from src.data_general import CrachoGeneralDataset
    from src.generator import generate_text
    from src.inference import load_model_for_inference
    from src.trainer import CrachoTrainer

    if args.steps < 1 or args.batch_size < 1 or args.grad_accum < 1:
        raise ValueError("Steps, batch size and gradient accumulation must be positive.")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.output).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing experiment: {run_dir}")
    data_dir = Path(args.data_dir).resolve()
    for name in ["train.txt", "validation.txt", "manifest.json"]:
        if not (data_dir / name).is_file():
            raise FileNotFoundError(f"Prepare the story corpus first: missing {data_dir / name}")
    manifest = json.loads((data_dir / "manifest.json").read_text())
    for record in manifest["splits"].values():
        path = data_dir / record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Prepared data changed: {path}. Create a fresh data directory.")

    model, tokenizer, source_info = load_model_for_inference(args.checkpoint, args.tokenizer, device)
    if not hasattr(tokenizer, "_tokenize_word"):
        raise ValueError("This experiment needs the model's paired BPE tokenizer.")
    tokenizer._tokenize_word = lru_cache(maxsize=65536)(tokenizer._tokenize_word)
    token_stats = {}
    datasets = {}
    for split, filename in [("train", "train.txt"), ("validation", "validation.txt")]:
        text = (data_dir / filename).read_text(encoding="utf-8")
        ids = tokenizer.encode(text, add_special_tokens=False)
        unknown = ids.count(tokenizer.unk_id)
        token_stats[split] = {"tokens": len(ids), "unknown": unknown,
                              "unknown_percent": 100 * unknown / max(1, len(ids))}
        if token_stats[split]["unknown_percent"] > 0.5:
            raise ValueError(f"{split} unknown-token rate is too high: {token_stats[split]}")
        datasets[split] = CrachoGeneralDataset(ids, model.config.max_seq_len)
        print(f"{split}: {token_stats[split]}", flush=True)
    sample_generator = torch.Generator().manual_seed(args.seed)
    sampler = RandomSampler(datasets["train"], replacement=True,
                            num_samples=args.steps * args.batch_size * args.grad_accum,
                            generator=sample_generator)
    train_loader = DataLoader(datasets["train"], batch_size=args.batch_size, sampler=sampler)
    validation_indices = random.Random(args.seed).sample(
        range(len(datasets["validation"])),
        min(args.val_chunks, len(datasets["validation"])),
    )
    val_loader = DataLoader(Subset(datasets["validation"], validation_indices),
                            batch_size=args.batch_size, shuffle=False)
    config = Config(model=model.config)
    config.training = TrainingConfig(
        batch_size=args.batch_size, grad_accum_steps=args.grad_accum,
        learning_rate=args.learning_rate, min_lr=args.learning_rate * 0.2,
        max_epochs=1, warmup_steps=min(10, max(1, args.steps // 10)),
        weight_decay=0.01, log_interval=args.grad_accum * 10,
    )
    config.system.checkpoints_dir = str(run_dir)
    trainer = CrachoTrainer(model, config, train_loader, val_loader, tokenizer, device)

    prompts = [
        "One day, a little girl went to the park. She",
        "Tom was hungry, so he",
        "The dog saw a red ball and",
        "User: What is machine learning?\nAssistant:",
    ]
    def samples():
        outputs = []
        for prompt in prompts:
            torch.manual_seed(args.seed)
            output = generate_text(model, tokenizer, prompt, max_new_tokens=72,
                                   temperature=0.7, top_k=40, device=device)
            outputs.append({"prompt": prompt, "output": output})
        return outputs

    run_dir.mkdir(parents=True, exist_ok=True)
    before_loss = trainer.evaluate()
    before_samples = samples()
    started = time.monotonic()
    print(f"Baseline validation loss: {before_loss:.6f}; bounded run: {args.steps} optimizer updates", flush=True)
    train_loss = trainer.train_epoch(1)
    after_loss = trainer.evaluate()
    after_samples = samples()
    duration = time.monotonic() - started
    report = {
        "source_model": source_info,
        "data_manifest": manifest,
        "token_counts": token_stats,
        "validation_chunks": validation_indices,
        "validation_target_tokens": len(validation_indices) * model.config.max_seq_len,
        "seed": args.seed, "optimizer_steps": trainer.global_step,
        "tokens_processed_in_experiment": trainer.tokens_processed,
        "seconds": duration, "training_loss": train_loss,
        "validation_loss_before": before_loss, "validation_loss_after": after_loss,
        "perplexity_before": math.exp(min(before_loss, 20)),
        "perplexity_after": math.exp(min(after_loss, 20)),
        "samples_before": before_samples, "samples_after": after_samples,
        "quality_note": "Validation measures next-token prediction on simple stories, not factuality or general chat quality. Generated examples require human review.",
        "saved_candidate": False,
    }
    # Save only an improved candidate. Never switch the app's default model here.
    if math.isfinite(after_loss) and after_loss < before_loss:
        original = torch.load(source_info["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
        inherited_tokens = original.get("tokens_processed", 0)
        del original
        gc.collect()
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config_dict": asdict(config),
            "epoch": 1, "global_step": trainer.global_step,
            "tokens_processed": inherited_tokens + trainer.tokens_processed,
            "val_loss": after_loss, "best_val_loss": after_loss,
            "checkpoint_kind": "inference_weights_only",
            "source_checkpoint": source_info["checkpoint"],
            "tokenizer_sha256": hashlib.sha256(Path(source_info["tokenizer"]).read_bytes()).hexdigest(),
        }
        temporary = run_dir / "best_model.pt.partial"
        torch.save(checkpoint, temporary)
        temporary.replace(run_dir / "best_model.pt")
        shutil.copy2(source_info["tokenizer"], run_dir / "tokenizer_bpe.json")
        report["saved_candidate"] = True
    report_path = run_dir / "experiment_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in [
        "validation_loss_before", "validation_loss_after", "perplexity_before",
        "perplexity_after", "optimizer_steps", "seconds", "saved_candidate",
    ]}, indent=2), flush=True)
    print(f"Report: {report_path}", flush=True)



def compare(args):
    """Compare two checkpoints on every prepared validation chunk."""
    import gc
    from functools import lru_cache
    import torch
    from torch.utils.data import DataLoader
    from src.data_general import CrachoGeneralDataset
    from src.generator import generate_text
    from src.inference import load_model_for_inference

    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir).resolve()
    manifest = json.loads((data_dir / "manifest.json").read_text())
    record = manifest["splits"]["valid"]
    validation_path = data_dir / record["file"]
    if hashlib.sha256(validation_path.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("Validation data no longer matches the preparation manifest.")
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace an existing comparison: {output}")
    prompts = [
        "One day, a little girl went to the park. She",
        "Tom was hungry, so he",
        "The dog saw a red ball and",
        "After the rain stopped, Mira",
        "User: What is machine learning?\nAssistant:",
    ]
    report = {"data_manifest": manifest, "seed": 42, "max_new_tokens": 128,
              "temperature": 0.7, "top_k": 40, "checkpoints": [],
              "note": "Loss and perplexity measure story prediction. Examples are unedited model output; they are not a grammar or factual-accuracy score."}
    expected_tokenizer = None
    for label, checkpoint in [("original_v3", args.baseline), ("fluency_candidate", args.candidate)]:
        model, tokenizer, info = load_model_for_inference(checkpoint, device=device)
        fingerprint = hashlib.sha256(Path(info["tokenizer"]).read_bytes()).hexdigest()
        if expected_tokenizer is not None and fingerprint != expected_tokenizer:
            raise ValueError("A loss comparison requires the same tokenizer in both models.")
        expected_tokenizer = fingerprint
        tokenizer._tokenize_word = lru_cache(maxsize=65536)(tokenizer._tokenize_word)
        tokens = tokenizer.encode(validation_path.read_text(encoding="utf-8"), add_special_tokens=False)
        dataset = CrachoGeneralDataset(tokens, model.config.max_seq_len)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        weighted_loss = 0.0
        target_tokens = 0
        with torch.inference_mode():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    _, loss = model(x, y)
                count = int((y != 0).sum().item())
                weighted_loss += loss.item() * count
                target_tokens += count
        loss = weighted_loss / target_tokens
        samples = []
        for prompt in prompts:
            torch.manual_seed(42)
            result = generate_text(model, tokenizer, prompt, max_new_tokens=128,
                                   temperature=0.7, top_k=40, device=device)
            samples.append({"prompt": prompt, "output": result})
        item = {"label": label, "model": info, "loss": loss,
                "perplexity": math.exp(loss), "target_tokens": target_tokens,
                "samples": samples}
        report["checkpoints"].append(item)
        print(json.dumps({key: item[key] for key in ["label", "loss", "perplexity", "target_tokens"]}), flush=True)
        del model, tokenizer, loader, dataset, tokens
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Full comparison saved: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Download a limited, attributed story corpus.")
    prep.add_argument("--data-dir", default=str(ROOT / "data_general" / "fluency_stories"))
    prep.add_argument("--train-stories", type=int, default=5000)
    prep.add_argument("--val-stories", type=int, default=500)
    prep.add_argument("--max-download-mib", type=int, default=32, help="Maximum bytes consumed per split.")
    fit = sub.add_parser("train", help="Run a bounded continuation experiment.")
    fit.add_argument("--data-dir", default=str(ROOT / "data_general" / "fluency_stories"))
    fit.add_argument("--checkpoint", default=str(ROOT / "checkpoints_general_v3" / "best_model.pt"))
    fit.add_argument("--tokenizer", default=None)
    fit.add_argument("--output", default=str(ROOT / ("checkpoints_fluency_" + datetime.now().strftime("%Y%m%d_%H%M%S"))))
    fit.add_argument("--steps", type=int, default=200)
    fit.add_argument("--batch-size", type=int, default=4)
    fit.add_argument("--grad-accum", type=int, default=4)
    fit.add_argument("--learning-rate", type=float, default=0.0001)
    fit.add_argument("--val-chunks", type=int, default=64)
    fit.add_argument("--seed", type=int, default=42)
    comparison = sub.add_parser("compare", help="Compare two checkpoints on all held-out story chunks.")
    comparison.add_argument("--data-dir", default=str(ROOT / "data_general" / "fluency_stories"))
    comparison.add_argument("--baseline", default=str(ROOT / "checkpoints_general_v3" / "best_model.pt"))
    comparison.add_argument("--candidate", required=True)
    comparison.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "compare":
        compare(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
