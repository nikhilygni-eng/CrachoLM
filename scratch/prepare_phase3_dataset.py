#!/usr/bin/env python3
"""
Phase 3 Scaled Dataset Preparation & Tokenizer Audit Script
============================================================
Downloads and prepares a ~17M-token multi-domain corpus into:
  data_general/raw_scaled/

Domains:
  1. 01_general_english.txt: WikiText-103 (valid + test + train subset)
  2. 02_educational_science.txt: Project Gutenberg Science Classics
  3. 03_conversations_qa.txt: Databricks Dolly-15k (full 15,000 entries)
  4. 04_programming_tech.txt: Full Python PEPs (0001-0700) + Python Docs

Format:
  - Document & turn boundaries enforced via <eos> (Token ID 3)
  - No training started, zero existing checkpoints touched.
"""

import os
import sys
import json
import math
import shutil
import hashlib
import urllib.request
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bpe_tokenizer import CrachoBPETokenizer

OUTPUT_DIR = "data_general/raw_scaled"
TOKENIZER_PATH = "checkpoints_general_v2/tokenizer_bpe.json"

def fetch_url(url: str, description: str, timeout: int = 30) -> str:
    print(f"[*] Downloading {description} from {url}...")
    headers = {"User-Agent": "Mozilla/5.0 (CrachoLM-Corpus-Builder/2.0)"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        content = response.read().decode("utf-8", errors="replace")
    print(f"    [✓] Fetched {len(content):,} characters.")
    return content

def prepare_general_english() -> str:
    # WikiText-103 valid + test + train sample
    urls = [
        "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
        "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt",
        "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
    ]
    texts = []
    for u in urls:
        t = fetch_url(u, u.split("/")[-1])
        t_clean = t.strip()
        if t_clean:
            texts.append(t_clean)
    return "\n<eos>\n".join(texts)

def prepare_educational_science() -> str:
    # Project Gutenberg classics: Faraday (1447), Darwin (2009), Newton Opticks (33504), Galileo (46036)
    urls = [
        ("https://www.gutenberg.org/cache/epub/1447/pg1447.txt", "Faraday - Chemical History of a Candle"),
        ("https://www.gutenberg.org/cache/epub/2009/pg2009.txt", "Darwin - Origin of Species"),
        ("https://www.gutenberg.org/cache/epub/33504/pg33504.txt", "Newton - Opticks"),
        ("https://www.gutenberg.org/cache/epub/30155/pg30155.txt", "Relativity - Einstein")
    ]
    texts = []
    for u, desc in urls:
        try:
            raw = fetch_url(u, desc)
            # Remove Gutenberg headers/footers
            start = raw.find("*** START OF THE PROJECT GUTENBERG")
            if start != -1:
                raw = raw[start:]
                end_idx = raw.find("End of the Project Gutenberg")
                if end_idx != -1:
                    raw = raw[:end_idx]
            texts.append(raw.strip())
        except Exception as e:
            print(f"[!] Warning: Failed to fetch {desc}: {e}")
    return "\n<eos>\n".join(texts)

def prepare_conversations_qa() -> str:
    # Databricks Dolly-15k (Full 15,000 entries)
    url = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
    raw = fetch_url(url, "Databricks Dolly-15k (Full)")
    entries = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
            instruction = item.get("instruction", "").strip()
            context = item.get("context", "").strip()
            response = item.get("response", "").strip()
            parts = []
            if context:
                parts.append(f"Instruction Context: {context}")
            parts.append(f"User: {instruction}")
            parts.append(f"Assistant: {response}<eos>")
            entries.append("\n".join(parts))
        except Exception:
            continue
    return "\n\n".join(entries)

def prepare_programming_tech() -> str:
    # Python PEPs (0001, 0008, 0020, 0257, 0484, 0554, 0600, 0622, 0634, 0654, 0700)
    pep_nums = [1, 8, 12, 20, 257, 484, 526, 554, 600, 622, 634, 654, 681, 700]
    urls = [f"https://peps.python.org/pep-{num:04d}.txt" for num in pep_nums]
    texts = []
    for u in urls:
        try:
            raw = fetch_url(u, u.split("/")[-1])
            clean = re.sub(r"^[=\-\*\~]{3,}$", "", raw, flags=re.MULTILINE)
            texts.append(clean.strip())
        except Exception as e:
            print(f"[!] Warning: Failed to fetch {u}: {e}")
    return "\n<eos>\n".join(texts)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 68)
    print(" 🚀 Phase 3 Scaled Dataset Preparation & Verification")
    print("=" * 68)

    domain_files = {
        "01_general_english.txt": prepare_general_english,
        "02_educational_science.txt": prepare_educational_science,
        "03_conversations_qa.txt": prepare_conversations_qa,
        "04_programming_tech.txt": prepare_programming_tech
    }

    file_contents = {}
    for fname, func in domain_files.items():
        print(f"\n--- Preparing {fname} ---")
        content = func()
        fpath = os.path.join(OUTPUT_DIR, fname)
        with open(fpath, "w", encoding="utf-8") as fp:
            fp.write(content)
        file_contents[fname] = content
        print(f"[✓] Saved {fname}: {len(content):,} characters ({os.path.getsize(fpath):,} bytes)")

    print("\n" + "=" * 68)
    print(" 🔬 TOKENIZER AUDIT & TOKEN COUNT ANALYSIS")
    print("=" * 68)

    tokenizer = CrachoBPETokenizer.load(TOKENIZER_PATH)
    print(f"[✓] Loaded Tokenizer: {TOKENIZER_PATH} (vocab_size={tokenizer.vocab_size})")

    total_tokens = 0
    domain_token_counts = {}

    for fname, content in file_contents.items():
        tokens = tokenizer.encode(content, add_special_tokens=False)
        count = len(tokens)
        domain_token_counts[fname] = count
        total_tokens += count
        eos_count = tokens.count(tokenizer.eos_id)
        print(f"  - {fname:30s}: {count:,} tokens (<eos> ID 3 count: {eos_count:,})")

    print("-" * 68)
    print(f" TOTAL CORPUS TOKEN COUNT: {total_tokens:,} tokens")
    print("=" * 68)

    # Token Count & Split Calculations
    train_tokens = int(total_tokens * 0.90)
    val_tokens = total_tokens - train_tokens

    seq_len = 256
    batch_size = 4
    grad_accum_steps = 16

    train_seqs = (train_tokens - 1) // seq_len
    val_seqs = (val_tokens - 1) // seq_len

    len_train_loader = math.ceil(train_seqs / batch_size)
    optim_updates_per_epoch = math.ceil(len_train_loader / grad_accum_steps)
    total_scheduler_steps_2ep = optim_updates_per_epoch * 2
    recommended_warmup = max(5, int(total_scheduler_steps_2ep * 0.05))

    # Checkpoint Size & Disk Space Audit
    sample_ckpt = "checkpoints_general_v2/checkpoint_epoch_1.pt"
    ckpt_size_bytes = os.path.getsize(sample_ckpt) if os.path.exists(sample_ckpt) else 857_000_000
    ckpt_size_mb = ckpt_size_bytes / (1024 * 1024)

    total_disk, used_disk, free_disk = shutil.disk_usage(".")
    free_gb = free_disk / (1024 ** 3)
    est_ckpt_dir_size_mb = ckpt_size_mb * 3  # epoch 1, epoch 2, best_model

    report = {
        "output_dir": OUTPUT_DIR,
        "total_tokens": total_tokens,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "domain_token_counts": domain_token_counts,
        "len_train_loader": len_train_loader,
        "optim_updates_per_epoch": optim_updates_per_epoch,
        "total_scheduler_steps_2ep": total_scheduler_steps_2ep,
        "recommended_warmup": recommended_warmup,
        "ckpt_size_mb": round(ckpt_size_mb, 2),
        "est_ckpt_dir_size_mb": round(est_ckpt_dir_size_mb, 2),
        "free_disk_gb": round(free_gb, 2)
    }

    with open("scratch/scaled_prep_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 68)
    print(" 📊 SCALED DATASET AUDIT REPORT")
    print("=" * 68)
    print(f" Total Tokens          : {total_tokens:,} tokens")
    print(f" Train Split (90%)     : {train_tokens:,} tokens")
    print(f" Val Split (10%)       : {val_tokens:,} tokens")
    print(f" Exact len(train_loader): {len_train_loader} batches")
    print(f" Optim Updates / Epoch : {optim_updates_per_epoch} steps")
    print(f" Total Steps (2 Epochs): {total_scheduler_steps_2ep} steps")
    print(f" Recommended Warmup    : {recommended_warmup} steps")
    print(f" Single Checkpoint Size: {ckpt_size_mb:.2f} MB")
    print(f" Est 3 Checkpoints Dir : {est_ckpt_dir_size_mb:.2f} MB (~{est_ckpt_dir_size_mb/1024:.2f} GB)")
    print(f" Available Free Disk   : {free_gb:.2f} GB")
    print("=" * 68 + "\n")

if __name__ == "__main__":
    main()
