#!/usr/bin/env python3
"""
CrachoLM Main Model Training Script (Phase 7 Enhanced)
======================================================
Trains CrachoLM-0.1 from scratch using Gradient Accumulation, Warmup + Cosine Decay,
AMP Mixed Precision, and automatic open dataset fetching.

Usage:
  python3 train.py
  python3 train.py --epochs 10 --batch-size 16 --grad-accum 4
  python3 train.py --resume checkpoints/best_model.pt
"""

import math
import argparse
import os
import sys

from config import default_config
from src.device import get_device, print_device_info
from src.tokenizer import CrachoTokenizer
from src.model import CrachoLM
from src.data import load_raw_text, prepare_dataloaders
from src.trainer import CrachoTrainer
from data.prepare_tinyshakespeare import download_tinyshakespeare


def parse_args():
    parser = argparse.ArgumentParser(description="Train CrachoLM-0.1 from scratch")
    parser.add_argument("--data-path", type=str, default=None, help="Path to raw text dataset file")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override micro-batch size")
    parser.add_argument("--grad-accum", type=int, default=None, help="Override gradient accumulation steps")
    parser.add_argument("--seq-len", type=int, default=None, help="Override sequence context length")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint file to resume training from")
    return parser.parse_args()


def main():
    args = parse_args()

    print_device_info()
    device = get_device()

    # 1. Configuration Setup
    config = default_config
    if args.epochs:
        config.training.max_epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.grad_accum:
        config.training.grad_accum_steps = args.grad_accum
    if args.seq_len:
        config.model.max_seq_len = args.seq_len

    # 2. Data File Setup (Download TinyShakespeare if default dataset missing)
    data_path = args.data_path
    if not data_path:
        data_path = os.path.join(config.data.raw_data_dir, "tinyshakespeare.txt")

    if not os.path.exists(data_path):
        print(f"[*] Raw text dataset not found at '{data_path}'. Fetching TinyShakespeare...")
        data_path = download_tinyshakespeare(config.data.raw_data_dir)

    print(f"[*] Loading raw dataset text from: {data_path}")
    raw_text = load_raw_text(data_path)

    # 3. Tokenizer Setup
    tokenizer_path = os.path.join(config.system.checkpoints_dir, "tokenizer.json")
    if os.path.exists(tokenizer_path):
        print(f"[*] Loading existing tokenizer from: {tokenizer_path}")
        tokenizer = CrachoTokenizer.load(tokenizer_path)
    else:
        print(f"[*] Training new tokenizer on dataset corpus...")
        tokenizer = CrachoTokenizer()
        tokenizer.train_from_text(raw_text)
        tokenizer.save(tokenizer_path)

    # Synchronize vocabulary size in config
    config.model.vocab_size = tokenizer.vocab_size

    # 4. Data Loaders Setup
    print(f"[*] Building DataLoaders (seq_len={config.model.max_seq_len}, micro_batch={config.training.batch_size}, grad_accum={config.training.grad_accum_steps})...")
    train_loader, val_loader, stats = prepare_dataloaders(
        text_data=raw_text,
        tokenizer=tokenizer,
        seq_len=config.model.max_seq_len,
        batch_size=config.training.batch_size,
        val_split=0.10
    )

    # 5. Model Initialization (From Scratch)
    print(f"[*] Initializing CrachoLM model with random weights...")
    model = CrachoLM(config.model)
    print(f"[*] Total Trainable Parameters: {model.get_num_params():,} ({model.get_num_params()/1e6:.2f}M)")

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

    # Model Evaluation Summary
    print("\n" + "=" * 68)
    print("      HOW TO EVALUATE IF TRAINED MODEL IS TRULY IMPROVING")
    print("=" * 68)
    print(" 1. Initial Random Baseline Loss: ~" + f"{math.log(tokenizer.vocab_size):.2f}")
    print(" 2. Healthy Training Curve: Both Train Loss and Val Loss drop together.")
    print(" 3. Text Generation Test: Run `python3 generate.py --prompt \"First Citizen:\"` to inspect generated text.")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
