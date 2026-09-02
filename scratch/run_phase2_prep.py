#!/usr/bin/env python3
"""
Phase 2 Full Benchmark Corpus Preparation & Validation
=======================================================
Downloads the full multi-domain corpus into data_general/benchmark_raw/,
replacing Phase 1 smoke-test files with their larger verified versions.

Steps:
  1. Record Phase 1 file stats for reference
  2. Force-download full versions of all 4 domains
  3. Deduplicate check (hash-based, per-file)
  4. Train a temporary BPE tokenizer → exact subword token count
  5. Print full source/license manifest and final report

No training started. No checkpoints written.
Protected paths NOT touched: checkpoints_general/, checkpoints_general_benchmark/, backup_70m/
"""

import hashlib
import json
import math
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bpe_tokenizer import CrachoBPETokenizer

OUTPUT_DIR = "data_general/benchmark_raw"

SOURCES = {
    "01_general_english.txt": {
        "name": "WikiText-2 Raw (full validation split)",
        "url": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
        "type": "plain_text",
        "license": "CC BY-SA 3.0",
        "license_url": "https://blog.salesforce.com/wikitext-2-and-wikitext-103/",
        "attribution_required": True,
        "non_commercial": False,
        "share_alike": True,
    },
    "02_educational_science.txt": {
        "name": "Project Gutenberg — The Chemical History of a Candle (Faraday, 1861)",
        "url": "https://www.gutenberg.org/cache/epub/1447/pg1447.txt",
        "type": "gutenberg",
        "license": "Public Domain (US, pre-1928)",
        "license_url": "https://www.gutenberg.org/license",
        "attribution_required": False,
        "non_commercial": False,
        "share_alike": False,
    },
    "03_conversations_qa.txt": {
        "name": "Databricks Dolly-15k (first 3000 entries)",
        "url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl",
        "type": "jsonl",
        "license": "CC BY-SA 4.0",
        "license_url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k",
        "attribution_required": True,
        "non_commercial": False,
        "share_alike": True,
    },
    "04_programming_tech.txt": {
        "name": "Python PEP Specifications (PEP 8, PEP 20, PEP 484) — Public Domain",
        "urls": [
            "https://raw.githubusercontent.com/python/peps/main/peps/pep-0008.rst",
            "https://raw.githubusercontent.com/python/peps/main/peps/pep-0020.rst",
            "https://raw.githubusercontent.com/python/peps/main/peps/pep-0484.rst",
        ],
        "type": "multi_rst",
        "license": "Public Domain",
        "license_url": "https://github.com/python/peps",
        "attribution_required": False,
        "non_commercial": False,
        "share_alike": False,
    },
}


def fetch(url: str, desc: str) -> str:
    print(f"  [↓] Fetching: {desc} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (CrachoLM Phase2 Prep)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def process_plain_text(raw: str) -> str:
    lines = [l.strip() for l in raw.splitlines() if l.strip() and not l.startswith("=")]
    return "\n\n".join(lines)


def process_gutenberg(raw: str) -> str:
    start = raw.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    if start != -1:
        raw = raw[raw.find("\n", start) + 1:]
    end = raw.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    if end != -1:
        raw = raw[:end]
    return raw.strip()


def process_jsonl(raw: str, max_items: int = 3000) -> str:
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
            if len(entries) >= max_items:
                break
        except Exception:
            continue
    return "\n\n".join(entries)


def process_rst_multi(urls: list) -> str:
    texts = []
    for url in urls:
        doc = url.split("/")[-1]
        raw = fetch(url, f"PEP ({doc})")
        clean = re.sub(r"^[=\-\*\~]{3,}$", "", raw, flags=re.MULTILINE)
        texts.append(clean.strip())
    return "\n\n".join(texts)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 68)
    print(" 🚀 CrachoLM-General-v2 Phase 2 Full Corpus Preparation")
    print("=" * 68)

    # ── Step 1: Record Phase 1 stats ────────────────────────────────────────
    print("\n[STEP 1] Phase 1 file inventory (for reference):")
    phase1_stats = {}
    for fname in sorted(SOURCES.keys()):
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fpath):
            sz = os.path.getsize(fpath)
            phase1_stats[fname] = sz
            print(f"  {fname}: {sz:,} bytes")
        else:
            print(f"  {fname}: NOT FOUND")

    # ── Step 2: Download & process all domains ───────────────────────────────
    print("\n[STEP 2] Downloading full Phase 2 corpus...")
    results = {}
    errors = []

    for fname, info in SOURCES.items():
        fpath = os.path.join(OUTPUT_DIR, fname)
        print(f"\n  Domain: {info['name']}")
        try:
            src_type = info["type"]
            if src_type == "plain_text":
                raw = fetch(info["url"], info["name"])
                text = process_plain_text(raw)
            elif src_type == "gutenberg":
                raw = fetch(info["url"], info["name"])
                text = process_gutenberg(raw)
            elif src_type == "jsonl":
                raw = fetch(info["url"], info["name"])
                text = process_jsonl(raw, max_items=3000)
            elif src_type == "multi_rst":
                text = process_rst_multi(info["urls"])
            else:
                raise ValueError(f"Unknown type: {src_type}")

            # Write file (overwrites Phase 1 version)
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(text)

            results[fname] = {
                "chars": len(text),
                "hash": sha256(text),
                "info": info,
            }
            p1_sz = phase1_stats.get(fname, 0)
            print(f"  [✓] Written: {len(text):,} chars  (Phase 1 was {p1_sz:,} bytes)")

        except Exception as e:
            print(f"  [X] ERROR: {e}")
            errors.append(f"{fname}: {e}")

    # ── Step 3: Duplicate-data check (cross-file hash comparison) ────────────
    print("\n[STEP 3] Duplicate-data check (SHA-256 cross-file comparison):")
    hashes = {fname: d["hash"] for fname, d in results.items()}
    seen = {}
    dup_found = False
    for fname, h in hashes.items():
        if h in seen:
            print(f"  [WARN] {fname} is IDENTICAL to {seen[h]}!")
            dup_found = True
        else:
            seen[h] = fname
    if not dup_found:
        print("  [✓] No duplicate files detected.")

    # ── Step 4: Train temporary BPE tokenizer → exact token count ────────────
    print("\n[STEP 4] Training temporary BPE tokenizer for exact token count...")
    combined_text = ""
    for fname in sorted(results.keys()):
        fpath = os.path.join(OUTPUT_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            combined_text += f.read() + "\n\n"

    tok = CrachoBPETokenizer(target_vocab_size=2048)
    tok.train_from_text(combined_text)
    token_ids = tok.encode(combined_text, add_special_tokens=False)
    total_tokens = len(token_ids)
    unk_count = token_ids.count(tok.unk_id)
    unk_rate = (unk_count / total_tokens * 100) if total_tokens > 0 else 0

    train_tokens = int(total_tokens * 0.90)
    val_tokens = total_tokens - train_tokens
    seqs_train = train_tokens // 256
    batches_per_epoch = seqs_train // 4
    optim_steps_per_epoch = batches_per_epoch // 16
    recommended_warmup = max(1, round(optim_steps_per_epoch * 5 * 0.05))  # 5% of 5-epoch total

    print(f"  [✓] Vocab Size      : {tok.vocab_size}")
    print(f"  [✓] Total Tokens    : {total_tokens:,}")
    print(f"  [✓] Unk Token Count : {unk_count:,} ({unk_rate:.2f}% unk rate)")
    print(f"  [✓] Train Tokens    : {train_tokens:,}")
    print(f"  [✓] Val Tokens      : {val_tokens:,}")
    print(f"  [✓] Train Sequences : {seqs_train:,}  (non-overlapping, seq_len=256)")
    print(f"  [✓] Batches/epoch   : {batches_per_epoch:,}")
    print(f"  [✓] Optim steps/ep  : {optim_steps_per_epoch}")
    print(f"  [✓] Recommended warmup (5% of 5ep): {recommended_warmup} steps")

    # ── Step 5: Update SOURCES_AND_LICENSES.md ───────────────────────────────
    total_chars = sum(d["chars"] for d in results.values())
    meta_path = os.path.join(OUTPUT_DIR, "SOURCES_AND_LICENSES.md")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("# CrachoLM-General-v2 Phase 2 Benchmark Corpus — Data Sources & Licenses\n\n")
        f.write(f"**Preparation Mode**: Phase 2 (Full Benchmark ~800k+ tokens)\n")
        f.write(f"**Total Character Count**: {total_chars:,} characters\n")
        f.write(f"**Exact Subword Tokens** (vocab_size=2048): {total_tokens:,} tokens\n\n")
        f.write("## Manifest\n\n")
        f.write("| File | Source | License | Attribution | Non-Commercial | ShareAlike | Chars |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for fname, d in results.items():
            info = d["info"]
            f.write(f"| `{fname}` | {info['name']} | {info['license']} | "
                    f"{'Yes' if info['attribution_required'] else 'No'} | "
                    f"No | "
                    f"{'Yes' if info['share_alike'] else 'No'} | "
                    f"{d['chars']:,} |\n")
        f.write("\n## Detailed Sources\n\n")
        for fname, d in results.items():
            info = d["info"]
            f.write(f"### `{fname}`\n")
            f.write(f"- **Source**: {info['name']}\n")
            if "url" in info:
                f.write(f"- **URL**: `{info['url']}`\n")
            else:
                for u in info["urls"]:
                    f.write(f"- **URL**: `{u}`\n")
            f.write(f"- **License**: {info['license']}\n")
            f.write(f"- **License Source**: {info['license_url']}\n")
            f.write(f"- **Attribution Required**: {'Yes' if info['attribution_required'] else 'No'}\n")
            f.write(f"- **Non-Commercial Restriction**: No\n")
            f.write(f"- **SHA-256**: `{d['hash'][:16]}...`\n")
            f.write(f"- **Characters**: {d['chars']:,}\n\n")

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(" 📊 PHASE 2 CORPUS PREPARATION FINAL REPORT")
    print("=" * 68)
    print(f" Output Directory    : {OUTPUT_DIR}")
    print(f" Total .txt Files    : {len(results)}")
    print(f" Total Characters    : {total_chars:,}")
    print(f" Exact Token Count   : {total_tokens:,} subword tokens (vocab=2048)")
    print(f" Unk Rate            : {unk_rate:.2f}%")
    print(f" Train / Val Split   : {train_tokens:,} / {val_tokens:,} tokens")
    print(f" Download Errors     : {len(errors)}")
    if errors:
        for e in errors:
            print(f"   [X] {e}")
    print(f" Duplicate Files     : {'YES — check warnings above!' if dup_found else 'None'}")
    print()
    print(" Per-File Summary:")
    for fname, d in results.items():
        est_tok = d["chars"] // 4
        print(f"   {fname}: {d['chars']:,} chars")
    print("=" * 68)
    print(" STOPPED — No training started. Awaiting approval.")
    print("=" * 68)


if __name__ == "__main__":
    main()
