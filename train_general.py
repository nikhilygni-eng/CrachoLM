#!/usr/bin/env python3
"""
CrachoLM-General-v2 Training Script
====================================
Trains CrachoLM-General-v2 (70M parameters) using a Subword BPE Tokenizer,
multi-domain corpus from data_general/raw/, Automatic Mixed Precision (AMP),
Gradient Accumulation, and checkpointing in checkpoints_general/.

Usage:
  python3 train_general.py
  python3 train_general.py --epochs 10 --batch-size 4 --grad-accum 16
  python3 train_general.py --resume checkpoints_general/best_model.pt
"""

import math
import argparse
import os
import sys

# Ensure root folder is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import default_config
from src.device import get_device, print_device_info
from src.bpe_tokenizer import CrachoBPETokenizer
from src.model import CrachoLM
from src.data_general import load_and_combine_general_corpus, prepare_general_dataloaders
from src.trainer import CrachoTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train CrachoLM-General-v2 from scratch")
    parser.add_argument("--data-dir", type=str, default="data_general/raw", help="Path to raw general text corpus directory")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints_general", help="Path to save checkpoints & BPE tokenizer")
    parser.add_argument("--target-vocab", type=int, default=1024, help="Target Subword BPE Vocabulary Size")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Micro-batch size per GPU iteration")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence context length")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume training from")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Override warmup steps for LR scheduler (default: config.training.warmup_steps=100)")
    return parser.parse_args()


def main():
    args = parse_args()

    print_device_info()
    device = get_device()

    # 1. System & Architecture Config (70M Model Scale)
    config = default_config
    config.model.d_model = 768
    config.model.n_layers = 10
    config.model.n_heads = 12
    config.model.d_ff = 3072
    config.model.max_seq_len = args.seq_len if args.seq_len else 256
    config.training.batch_size = args.batch_size if args.batch_size else 4
    config.training.grad_accum_steps = args.grad_accum if args.grad_accum else 16
    if args.epochs:
        config.training.max_epochs = args.epochs
    if args.warmup_steps is not None:
        config.training.warmup_steps = args.warmup_steps
        print(f"[*] Warmup Steps Override   : {config.training.warmup_steps} (via --warmup-steps CLI)")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # 2. Multi-Domain General Dataset Setup
    print(f"[*] Loading multi-domain general text corpus from: {args.data_dir}")
    raw_text, corpus_meta = load_and_combine_general_corpus(args.data_dir)
    print(f"[✓] Loaded {corpus_meta['num_files']} text file(s) ({corpus_meta['total_raw_chars']:,} characters).")

    # 3. Load the verified existing Subword BPE Tokenizer
    tokenizer_path = "checkpoints_general_v2/tokenizer_bpe.json"

    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Verified tokenizer not found: {tokenizer_path}"
        )

    print(f"[*] Loading verified BPE tokenizer from: {tokenizer_path}")
    tokenizer = CrachoBPETokenizer.load(tokenizer_path)

    # Automatically set model vocab_size to match tokenizer vocab_size
    config.model.vocab_size = tokenizer.vocab_size

    # 4. Data Loaders Setup
    print(f"[*] Building DataLoaders (seq_len={config.model.max_seq_len}, micro_batch={config.training.batch_size}, grad_accum={config.training.grad_accum_steps})...")
    train_loader, val_loader, stats = prepare_general_dataloaders(
        text_data=raw_text,
        tokenizer=tokenizer,
        seq_len=config.model.max_seq_len,
        batch_size=config.training.batch_size,
        val_split=0.10,
        raw_dir=args.data_dir
    )

    print("\n" + "=" * 68)
    print(" 📊 DATASET VALIDATION REPORT (CrachoLM-General-v2)")
    print("=" * 68)
    print(f" Raw Character Count  : {stats['raw_char_count']:,}")
    print(f" Files Included       : {corpus_meta['num_files']}")
    print(f" Tokenizer Vocab Size : {stats['vocab_size']} tokens (Subword BPE)")
    print(f" Total Subword Tokens : {stats['total_tokens']:,}")
    print(f" Train Token Count    : {stats['train_tokens']:,} ({stats['train_samples']} non-overlapping samples)")
    print(f" Val Token Count      : {stats['val_tokens']:,} ({stats['val_samples']} non-overlapping samples)")
    print(f" Unknown Token Count  : {stats['unk_count']} (Unk Rate: {stats['unk_rate_percent']}%)")
    print("=" * 68 + "\n")

    # 5. Model Initialization
    print(f"[*] Initializing CrachoLM-General-v2 model with random weights...")
    model = CrachoLM(config.model)
    n_params = model.get_num_params()
    print(f"[✓] Total Trainable Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # Set checkpoint output directory in config
    config.system.checkpoints_dir = args.ckpt_dir

    # 6. Initialize Trainer & Execute Training Loop
    trainer = CrachoTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        resume_checkpoint_path=args.resume
    )

    trainer.train()

    print("\n" + "=" * 68)
    print(" 🚀 CrachoLM-General-v2 Training Complete!")
    print(f" Best Model Checkpoint: {os.path.join(args.ckpt_dir, 'best_model.pt')}")
    print(f" BPE Tokenizer File   : {tokenizer_path}")
    print(" Run `python3 generate_general.py` to test generation!")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
