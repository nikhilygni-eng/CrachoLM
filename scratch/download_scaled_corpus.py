#!/usr/bin/env python3
"""
Physical Corpus Download & Disk File Writer
=============================================
Physically fetches raw text datasets from open sources and writes them directly
to disk at:
  /home/escanor/Downloads/CrachoLM/data_general/raw_scaled/

Files created:
  1. 01_general_english.txt
  2. 02_educational_science.txt
  3. 03_conversations_qa.txt
  4. 04_programming_tech.txt
"""

import os
import sys
import json
import re
import urllib.request
import importlib.util

BASE_DIR = "/home/escanor/Downloads/CrachoLM"
OUTPUT_DIR = os.path.join(BASE_DIR, "data_general", "raw_scaled")
TOKENIZER_PATH = os.path.join(BASE_DIR, "checkpoints_general_v2", "tokenizer_bpe.json")

# Import BPE Tokenizer without PyTorch dependency
bpe_spec = importlib.util.spec_from_file_location("bpe_tokenizer", os.path.join(BASE_DIR, "src", "bpe_tokenizer.py"))
bpe_module = importlib.util.module_from_spec(bpe_spec)
bpe_spec.loader.exec_module(bpe_module)
CrachoBPETokenizer = bpe_module.CrachoBPETokenizer


def fetch_url(url: str, description: str, timeout: int = 60) -> str:
    print(f"[*] Downloading {description} from {url}...")
    headers = {"User-Agent": "Mozilla/5.0 (CrachoLM-Corpus-Fetcher/2.0)"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        content = response.read().decode("utf-8", errors="replace")
    print(f"    [✓] Received {len(content):,} characters ({len(content.encode('utf-8')):,} bytes).")
    return content


def build_general_english() -> str:
    urls = [
        "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
        "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt",
        "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    ]
    parts = []
    for u in urls:
        try:
            t = fetch_url(u, u.split("/")[-1])
            if t.strip():
                parts.append(t.strip())
        except Exception as e:
            print(f"[!] Failed {u}: {e}")
    return "\n<eos>\n".join(parts)


def build_educational_science() -> str:
    urls = [
        ("https://www.gutenberg.org/cache/epub/1447/pg1447.txt", "Faraday - Chemical History of a Candle"),
        ("https://www.gutenberg.org/cache/epub/2009/pg2009.txt", "Darwin - Origin of Species"),
        ("https://www.gutenberg.org/cache/epub/33504/pg33504.txt", "Newton - Opticks"),
        ("https://www.gutenberg.org/cache/epub/30155/pg30155.txt", "Relativity - Einstein"),
        ("https://www.gutenberg.org/cache/epub/1259/pg1259.txt", "Twenty Thousand Leagues Under the Sea"),
        ("https://www.gutenberg.org/cache/epub/84/pg84.txt", "Frankenstein - Mary Shelley")
    ]
    parts = []
    for u, desc in urls:
        try:
            raw = fetch_url(u, desc)
            start = raw.find("*** START OF THE PROJECT GUTENBERG")
            if start != -1:
                raw = raw[start:]
                end_idx = raw.find("End of the Project Gutenberg")
                if end_idx != -1:
                    raw = raw[:end_idx]
            parts.append(raw.strip())
        except Exception as e:
            print(f"[!] Failed {desc}: {e}")
    return "\n<eos>\n".join(parts)


def build_conversations_qa() -> str:
    url = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
    raw = fetch_url(url, "Databricks Dolly-15k (Full 15,000 entries)")
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


def build_programming_tech() -> str:
    pep_nums = [1, 8, 12, 20, 257, 484, 526, 554, 600, 622, 634, 654, 681, 700]
    parts = []
    for num in pep_nums:
        downloaded = False
        for ext in ["rst", "md"]:
            u = f"https://raw.githubusercontent.com/python/peps/main/peps/pep-{num:04d}.{ext}"
            try:
                raw = fetch_url(u, f"PEP {num:04d}.{ext}")
                clean = re.sub(r"^[=\-\*\~]{3,}$", "", raw, flags=re.MULTILINE)
                parts.append(clean.strip())
                downloaded = True
                break
            except Exception:
                continue
        if not downloaded:
            print(f"[!] Warning: PEP {num:04d} could not be fetched.")

    tut_pages = ["appetite", "interpreter", "introduction", "controlflow", "datastructures", "modules", "inputoutput", "errors", "classes", "stdlib", "stdlib2", "venv"]
    for page in tut_pages:
        u = f"https://raw.githubusercontent.com/python/cpython/main/Doc/tutorial/{page}.rst"
        try:
            raw = fetch_url(u, f"Doc/tutorial/{page}.rst")
            clean = re.sub(r"^[=\-\*\~]{3,}$", "", raw, flags=re.MULTILINE)
            parts.append(clean.strip())
        except Exception as e:
            print(f"[!] Failed {u}: {e}")

    return "\n<eos>\n".join(parts)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 68)
    print(f" 🚀 Downloading Scaled Corpus to Disk: {OUTPUT_DIR}")
    print("=" * 68)

    generators = {
        "01_general_english.txt": build_general_english,
        "02_educational_science.txt": build_educational_science,
        "03_conversations_qa.txt": build_conversations_qa,
        "04_programming_tech.txt": build_programming_tech
    }

    for filename, gen_func in generators.items():
        filepath = os.path.join(OUTPUT_DIR, filename)
        print(f"\n[+] Processing {filename}...")
        content = gen_func()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        size_bytes = os.path.getsize(filepath)
        print(f"    [✓] PHYSICALLY WRITTEN TO DISK: {filepath}")
        print(f"        Bytes: {size_bytes:,} | Chars: {len(content):,}")

    print("\n" + "=" * 68)
    print(" 🔬 PHYSICAL DISK & TOKENIZER VERIFICATION")
    print("=" * 68)

    tokenizer = CrachoBPETokenizer.load(TOKENIZER_PATH)
    total_tokens = 0
    validation_passed = True

    required_files = list(generators.keys())
    print("\nDirectory Listing & Strict Domain Validation:")

    total_disk_bytes = 0
    for fname in required_files:
        fpath = os.path.join(OUTPUT_DIR, fname)
        
        # 1. Existence check
        if not os.path.exists(fpath):
            print(f"  [❌] {fname:30s} | FILE DOES NOT EXIST!")
            validation_passed = False
            continue
            
        sz = os.path.getsize(fpath)
        total_disk_bytes += sz
        
        # 2. Non-zero byte size check
        if sz == 0:
            print(f"  [❌] {fname:30s} | Size: {sz} bytes (FILE IS EMPTY!)")
            validation_passed = False
            continue
            
        with open(fpath, "r", encoding="utf-8") as fp:
            txt = fp.read()
            
        tokens = tokenizer.encode(txt, add_special_tokens=False)
        tok_count = len(tokens)
        total_tokens += tok_count
        eos_count = tokens.count(tokenizer.eos_id)
        
        # 3. Non-zero token count check
        if tok_count == 0:
            print(f"  [❌] {fname:30s} | Size: {sz:10,} bytes | Tokens: 0 (NO TOKENS PRODUCED!)")
            validation_passed = False
        else:
            print(f"  [✓] {fname:30s} | Size: {sz/1024/1024:6.2f} MB ({sz:10,} bytes) | Tokens: {tok_count:9,} | <eos> ID 3: {eos_count:5,}")

    print("-" * 68)
    print(f" Total Disk Usage (du -sh) : {total_disk_bytes / 1024 / 1024:.2f} MB ({total_disk_bytes:,} bytes)")
    print(f" Total Exact Token Count    : {total_tokens:,} tokens")
    print("=" * 68 + "\n")

    if not validation_passed:
        print("=" * 68)
        print(" ❌ CORPUS NOT READY FOR TRAINING")
        print("    One or more required domain files failed validation.")
        print("=" * 68)
        sys.exit(1)
    else:
        print("=" * 68)
        print(" SUCCESS: ALL DOMAIN FILES VALIDATED AND READY FOR PREPARATION")
        print("=" * 68)


if __name__ == "__main__":
    main()

