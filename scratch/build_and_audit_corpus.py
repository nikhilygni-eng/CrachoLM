import os
import sys
import json
import math
import shutil
import hashlib
import urllib.request
import re

# Direct import of CrachoBPETokenizer without importing torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import tokenizer class directly from module file to bypass torch imports in src/__init__.py
import importlib.util
bpe_spec = importlib.util.spec_from_file_location("bpe_tokenizer", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "bpe_tokenizer.py"))
bpe_module = importlib.util.module_from_spec(bpe_spec)
bpe_spec.loader.exec_module(bpe_module)
CrachoBPETokenizer = bpe_module.CrachoBPETokenizer

OUTPUT_DIR = "data_general/raw_scaled"
TOKENIZER_PATH = "checkpoints_general_v2/tokenizer_bpe.json"

def fetch_url(url: str, description: str, timeout: int = 30) -> str:
    print(f"[*] Downloading {description}...")
    headers = {"User-Agent": "Mozilla/5.0 (CrachoLM-Corpus-Builder/2.0)"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        content = response.read().decode("utf-8", errors="replace")
    print(f"    [✓] Fetched {len(content):,} characters.")
    return content

def prepare_general_english() -> str:
    urls = [
        "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
        "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/test.txt",
        "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/train.txt"
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
        ("https://www.gutenberg.org/cache/epub/30155/pg30155.txt", "Relativity - Einstein")
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
    print(" 🚀 Building Scaled Corpus & Running Forensic Audit")
    print("=" * 68)

    domain_funcs = {
        "01_general_english.txt": prepare_general_english,
        "02_educational_science.txt": prepare_educational_science,
        "03_conversations_qa.txt": prepare_conversations_qa,
        "04_programming_tech.txt": prepare_programming_tech
    }

    file_contents = {}
    for fname, func in domain_funcs.items():
        print(f"\n[+] Preparing {fname}...")
        txt = func()
        fpath = os.path.join(OUTPUT_DIR, fname)
        with open(fpath, "w", encoding="utf-8") as fp:
            fp.write(txt)
        file_contents[fname] = txt
        print(f"    Saved: {fpath} ({len(txt):,} chars, {os.path.getsize(fpath):,} bytes)")

    tokenizer = CrachoBPETokenizer.load(TOKENIZER_PATH)
    print(f"\n[✓] Tokenizer loaded: {TOKENIZER_PATH} (eos_id = {tokenizer.eos_id})")

    total_exact_tokens = 0
    results = {}

    for fname, content in file_contents.items():
        fpath = os.path.join(OUTPUT_DIR, fname)
        file_size_bytes = os.path.getsize(fpath)
        
        # Tokenize with BPE tokenizer
        tokens = tokenizer.encode(content, add_special_tokens=False)
        exact_token_count = len(tokens)
        total_exact_tokens += exact_token_count
        
        # Count literal <eos> tags vs token ID 3 occurrences
        literal_eos_count = content.count("<eos>")
        token_id_3_count = tokens.count(tokenizer.eos_id)
        
        results[fname] = {
            "bytes": file_size_bytes,
            "chars": len(content),
            "exact_tokens": exact_token_count,
            "literal_eos_tags": literal_eos_count,
            "token_id_3_count": token_id_3_count
        }

    with open("scratch/forensic_corpus_audit.json", "w") as fp:
        json.dump({"total_exact_tokens": total_exact_tokens, "files": results}, fp, indent=2)

    print("\n" + "=" * 68)
    print(" 📊 FORENSIC TOKEN AUDIT RESULTS")
    print("=" * 68)
    for fname, r in results.items():
        print(f" - {fname:30s}: Bytes={r['bytes']:,} | Chars={r['chars']:,} | Exact Tokens={r['exact_tokens']:,} | Literal <eos>={r['literal_eos_tags']:,} | Token ID 3={r['token_id_3_count']:,}")
    print("-" * 68)
    print(f" TOTAL EXACT TOKENS ACROSS ALL FILES: {total_exact_tokens:,}")
    print("=" * 68)

if __name__ == "__main__":
    main()
