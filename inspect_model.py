#!/usr/bin/env python3
"""
CrachoLM Phase 7: Baseline System & Performance Inspection
==========================================================
Inspects model parameters, dataset size, VRAM footprint, throughput,
and generates sample outputs for diagnostic analysis.
"""

import time
import os
import torch

from config import default_config
from src.device import get_device, get_device_info, print_device_info
from src.tokenizer import CrachoTokenizer
from src.model import CrachoLM
from src.data import load_raw_text, prepare_dataloaders
from src.trainer import CrachoTrainer
from src.generator import generate_text


def run_baseline_inspection():
    print("=" * 68)
    print("           CrachoLM Baseline Inspection & Analysis")
    print("=" * 68)

    # 1. Hardware & GPU Memory
    print_device_info()
    device = get_device()

    # 2. Config & Parameter Count
    config = default_config
    data_path = os.path.join(config.data.raw_data_dir, "sample_corpus.txt")
    raw_text = load_raw_text(data_path)

    tokenizer = CrachoTokenizer()
    tokenizer.train_from_text(raw_text)
    config.model.vocab_size = tokenizer.vocab_size

    model = CrachoLM(config.model).to(device)
    params = model.get_num_params()

    print("\n" + "-" * 68)
    print(" 📊 METRIC INSPECTION REPORT")
    print("-" * 68)
    print(f" 1. Model Parameters      : {params:,} ({params/1e6:.2f} Million)")
    print(f" 2. Vocabulary Size        : {tokenizer.vocab_size} tokens (Character-level)")
    print(f" 3. Dataset Character Count: {len(raw_text):,} chars")
    print(f" 4. Model Context Length   : {config.model.max_seq_len} tokens")
    print(f" 5. Hidden Dim (d_model)   : {config.model.d_model} (Layers: {config.model.n_layers}, Heads: {config.model.n_heads})")

    # 3. Benchmark Training Speed & VRAM
    train_loader, val_loader, stats = prepare_dataloaders(
        text_data=raw_text,
        tokenizer=tokenizer,
        seq_len=64,
        batch_size=8,
        val_split=0.1
    )

    trainer = CrachoTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device
    )

    print("\n[*] Running 1 Diagnostic Training Epoch for Speed & VRAM Measurement...")
    t0 = time.time()
    train_loss = trainer.train_epoch(1)
    t1 = time.time()
    val_loss = trainer.evaluate()

    tokens_epoch = stats["train_tokens"]
    elapsed = t1 - t0
    tok_per_sec = tokens_epoch / max(elapsed, 1e-5)

    vram_used = "N/A"
    if device.type == "cuda":
        vram_used = f"{torch.cuda.memory_allocated(0)/(1024**3):.2f} GB allocated / {torch.cuda.memory_reserved(0)/(1024**3):.2f} GB reserved"

    print("-" * 68)
    print(f" 6. Diagnostic Train Loss  : {train_loss:.4f}")
    print(f" 7. Diagnostic Val Loss    : {val_loss:.4f}")
    print(f" 8. Training Speed         : {tok_per_sec:.0f} tokens/sec ({elapsed:.2f}s per epoch)")
    print(f" 9. GPU VRAM Usage         : {vram_used}")
    print("-" * 68)

    # 4. Sample Generation
    print("\n[*] Sample Generation (Untrained/Initial State):")
    sample_gen = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt="CrachoLM",
        max_new_tokens=50,
        temperature=0.8,
        top_k=20,
        device=device
    )
    print(f"   Output: '{sample_gen.strip()}'")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    run_baseline_inspection()
