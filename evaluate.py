#!/usr/bin/env python3
"""
CrachoLM Phase 8: Model Evaluation CLI Script
===============================================
Loads a trained checkpoint, runs the evaluation question bank against it,
prints per-question results and overall metrics, and saves output to
logs/eval_results.json and logs/eval_results.csv.

Usage:
  python3 evaluate.py
  python3 evaluate.py --checkpoint checkpoints/best_model.pt
  python3 evaluate.py --checkpoint checkpoints/best_model.pt --temperature 0.1

IMPORTANT:
  This script does NOT modify model weights.
  All evaluation data lives exclusively in data/eval/ — never in data/raw/.
"""

import argparse
import json
import os
import sys
import math
import torch
from datetime import datetime

from src.device import get_device
from src.tokenizer import CrachoTokenizer
from src.model import CrachoLM
from src.evaluator import (
    run_evaluation,
    compute_metrics,
    save_results_json,
    save_results_csv,
    verify_no_data_leakage,
)
from src.checkpoint import load_checkpoint_file, get_model_config_from_checkpoint

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
EVAL_QUESTIONS_PATH = os.path.join(PROJECT_ROOT, "data", "eval", "eval_questions.json")
TRAINING_DATA_PATH  = os.path.join(PROJECT_ROOT, "data", "raw", "tinyshakespeare.txt")
DEFAULT_CHECKPOINT  = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")
DEFAULT_TOKENIZER   = os.path.join(PROJECT_ROOT, "checkpoints", "tokenizer.json")


def parse_args():
    parser = argparse.ArgumentParser(description="CrachoLM Evaluation CLI")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER,
                        help="Path to tokenizer JSON file")
    parser.add_argument("--eval-data", type=str, default=EVAL_QUESTIONS_PATH,
                        help="Path to evaluation questions JSON")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature during eval (default 0.1 for near-deterministic)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top-K filtering during eval")
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "logs"),
                        help="Directory to save JSON and CSV result files")
    return parser.parse_args()


def print_banner(title: str):
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


def main():
    args = parse_args()
    device = get_device()

    print_banner("CrachoLM Phase 8 — Model Evaluation")

    # ── 1. Data Leakage Guard ─────────────────────────────────────────────
    print("\n[STEP 1] Verifying evaluation data is separate from training data...")
    leakage_ok = verify_no_data_leakage(args.eval_data, TRAINING_DATA_PATH)
    if not leakage_ok:
        sys.exit(1)

    # ── 2. Load Evaluation Questions ─────────────────────────────────────
    print(f"\n[STEP 2] Loading evaluation questions from: {args.eval_data}")
    if not os.path.exists(args.eval_data):
        print(f"[X] Evaluation data file not found at: {args.eval_data}")
        sys.exit(1)
    with open(args.eval_data, "r", encoding="utf-8") as f:
        questions = json.load(f)
    print(f"    [✓] Loaded {len(questions)} evaluation questions.")

    # ── 3. Load Tokenizer ─────────────────────────────────────────────────
    print(f"\n[STEP 3] Loading tokenizer from: {args.tokenizer}")
    if not os.path.exists(args.tokenizer):
        print(f"[X] Tokenizer file not found at: {args.tokenizer}")
        print("    Run: python3 train.py  to train the model and save the tokenizer first.")
        sys.exit(1)
    tokenizer = CrachoTokenizer.load(args.tokenizer)

    # ── 4. Load Checkpoint ────────────────────────────────────────────────
    print(f"\n[STEP 4] Loading model checkpoint from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"[X] Checkpoint not found at: {args.checkpoint}")
        print("    Run: python3 train.py --epochs 10  to produce a checkpoint first.")
        sys.exit(1)

    try:
        # load_checkpoint_file uses weights_only=False (required for optimizer state).
        # Safe here because the file is produced locally by train.py.
        checkpoint = load_checkpoint_file(args.checkpoint, device)
    except Exception as e:
        print(f"[X] Failed to load checkpoint: {e}")
        sys.exit(1)

    model_config = get_model_config_from_checkpoint(checkpoint)

    # Vocab mismatch guard
    if model_config.vocab_size != tokenizer.vocab_size:
        print(f"[X] Vocabulary mismatch: checkpoint has {model_config.vocab_size}, "
              f"tokenizer has {tokenizer.vocab_size}.")
        sys.exit(1)

    model = CrachoLM(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    n_params = model.get_num_params()

    val_loss_at_ckpt = checkpoint.get("val_loss", None)
    epoch_at_ckpt    = checkpoint.get("epoch", "unknown")

    print(f"    [✓] Model loaded. Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    val_loss_str = f"{val_loss_at_ckpt:.4f}" if val_loss_at_ckpt is not None else "N/A"
    print(f"    [✓] Checkpoint epoch: {epoch_at_ckpt} | Saved val loss: {val_loss_str}")

    # ── 5. Run Evaluation ─────────────────────────────────────────────────
    print(f"\n[STEP 5] Running evaluation ({len(questions)} questions, "
          f"temperature={args.temperature}, top_k={args.top_k})...")
    results = run_evaluation(
        model=model,
        tokenizer=tokenizer,
        questions=questions,
        device=device,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    # ── 6. Compute Metrics ────────────────────────────────────────────────
    metrics = compute_metrics(results)

    # ── 7. Print Per-Question Results Table ───────────────────────────────
    print_banner("Per-Question Results")
    FMT = "{:<10} {:<22} {:<20} {:<20} {:<6}"
    print(FMT.format("ID", "Category", "Expected", "Model Answer", "Result"))
    print("-" * 68)
    correct_examples   = []
    incorrect_examples = []
    for r in results:
        status = "✓ PASS" if r["correct"] else "✗ FAIL"
        exp_display = str(r["expected"])[:18]
        ans_display = str(r["model_answer"])[:18]
        print(FMT.format(r["id"], r["category"], exp_display, ans_display, status))
        if r["correct"]:
            correct_examples.append(r)
        else:
            incorrect_examples.append(r)

    # ── 8. Print Metrics Summary ──────────────────────────────────────────
    print_banner("Overall Metrics")
    print(f"  Model Parameters     : {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"  Vocabulary Size      : {tokenizer.vocab_size} tokens (character-level)")
    print(f"  Checkpoint Used      : {os.path.basename(args.checkpoint)}")
    print(f"  Checkpoint Epoch     : {epoch_at_ckpt}")
    if val_loss_at_ckpt:
        perplexity = math.exp(min(val_loss_at_ckpt, 20))
        print(f"  Saved Val Loss       : {val_loss_at_ckpt:.4f}  (Perplexity: {perplexity:.1f})")
    print(f"  Total Questions      : {metrics['total_questions']}")
    print(f"  Correct Answers      : {metrics['correct_answers']}")
    print(f"  Incorrect Answers    : {metrics['incorrect_answers']}")
    print(f"  Exact-Match Accuracy : {metrics['exact_match_accuracy'] * 100:.1f}%")
    print()
    print("  Per-Category Breakdown:")
    for cat, stats in metrics["per_category"].items():
        print(f"    {cat:<28}: {stats['correct']}/{stats['total']}  "
              f"({stats['accuracy']*100:.0f}%)")

    # ── 9. Example Highlights ─────────────────────────────────────────────
    if correct_examples:
        print_banner("Example Correct Answers")
        for r in correct_examples[:3]:
            print(f"  [{r['id']}] Prompt   : {r['prompt']!r}")
            print(f"         Expected : {r['expected']!r}")
            print(f"         Got      : {r['model_answer'][:40]!r}")
            print()

    if incorrect_examples:
        print_banner("Example Incorrect Answers")
        for r in incorrect_examples[:3]:
            print(f"  [{r['id']}] Prompt   : {r['prompt']!r}")
            print(f"         Expected : {r['expected']!r}")
            print(f"         Got      : {r['model_answer'][:40]!r}")
            print()

    # ── 10. Honest Limitations Notice ─────────────────────────────────────
    print_banner("⚠  IMPORTANT: Limitations of This Evaluation")
    print("""
  CrachoLM-0.1 is a ~4.87M parameter character-level model trained from scratch.

  What these results mean:
    - A CORRECT answer shows pattern completion from training text.
    - It does NOT indicate understanding, reasoning, or factual knowledge.
    - A WRONG answer is expected on most factual or arithmetic questions.

  What these results do NOT mean:
    - High accuracy does NOT equal general intelligence.
    - Correct answers may be coincidental pattern matches.
    - The model cannot verify facts, it only predicts likely characters.

  How to improve accuracy (future phases):
    1. Train on a larger, more diverse text corpus.
    2. Upgrade to a subword (BPE) tokenizer for semantic tokens.
    3. Scale model parameters to 10M+ after verifying loss curves.
""")

    # ── 11. Save Results ──────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.output_dir, f"eval_results_{timestamp}.json")
    csv_path  = os.path.join(args.output_dir, f"eval_results_{timestamp}.csv")

    meta = {
        "timestamp": timestamp,
        "checkpoint": args.checkpoint,
        "tokenizer": args.tokenizer,
        "n_params": n_params,
        "vocab_size": tokenizer.vocab_size,
        "checkpoint_epoch": epoch_at_ckpt,
        "val_loss_at_checkpoint": val_loss_at_ckpt,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }
    save_results_json(results, metrics, meta, json_path)
    save_results_csv(results, csv_path)

    print(f"\n  Results saved to logs/  (JSON + CSV with timestamp {timestamp})")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
