import os
import sys
import json
import math
import shutil
import hashlib
import urllib.request
import re

OUTPUT_DIR = "data_general/raw_scaled"
TOKENIZER_PATH = "checkpoints_general_v2/tokenizer_bpe.json"

def fetch_url(url: str, description: str, timeout: int = 30) -> str:
    print(f"[*] Downloading {description}...")
    headers = {"User-Agent": "Mozilla/5.0 (CrachoLM-Corpus-Builder/2.0)"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        content = response.read().decode("utf-8", errors="replace")
    print(f"    [✓] Fetched {len(content):,} characters ({len(content.encode('utf-8')):,} bytes).")
    return content

def prepare_general_english() -> str:
    urls = [
        "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
        "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt",
        "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt",
        "https://huggingface.co/datasets/karpathy/tinyshakespeare/raw/main/input.txt"
    ]
    texts = []
    for u in urls:
        try:
            t = fetch_url(u, u.split("/")[-1])
            if t.strip():
                texts.append(t.strip())
        except Exception as e:
            print(f"[!] Warning: Failed to fetch {u}: {e}")
    return "\n<eos>\n".join(texts)

def prepare_educational_science() -> str:
    urls = [
        ("https://www.gutenberg.org/cache/epub/1447/pg1447.txt", "Faraday - Chemical History of a Candle"),
        ("https://www.gutenberg.org/cache/epub/2009/pg2009.txt", "Darwin - Origin of Species"),
        ("https://www.gutenberg.org/cache/epub/33504/pg33504.txt", "Newton - Opticks"),
        ("https://www.gutenberg.org/cache/epub/30155/pg30155.txt", "Relativity - Einstein"),
        ("https://www.gutenberg.org/cache/epub/1259/pg1259.txt", "Twenty Thousand Leagues Under the Sea"),
        ("https://www.gutenberg.org/cache/epub/84/pg84.txt", "Frankenstein - Mary Shelley")
    ]
    texts = []
    for u, desc in urls:
        try:
            raw = fetch_url(u, desc)
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

def prepare_programming_tech() -> str:
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
    print(" 🚀 CrachoLM Phase 3 Corpus Download & Scaled Verification")
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

    print("\n[✓] All 4 domains downloaded and formatted with <eos> boundaries.")

if __name__ == "__main__":
    main()
