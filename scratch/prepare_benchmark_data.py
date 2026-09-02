#!/usr/bin/env python3
"""
CrachoLM-General-v2 Benchmark Data Preparation Script (Verified & Resumable)
=============================================================================
Downloads and extracts clean, permissively-licensed text from verified,
official open sources to build a multi-domain benchmark corpus.

License Verification:
- General English: WikiText-2 Raw (CC BY-SA 3.0)
- Educational Science: Project Gutenberg Public Domain Science (Public Domain)
- Conversations & Q&A: Databricks Dolly-15k (CC BY-SA 4.0 via Hugging Face Resolve LFS)
- Programming & Tech: Python PEP 8 & PEP 20 (Public Domain / PSF License)
"""

import argparse
import json
import os
import sys
import urllib.request
import re
from typing import Dict, List, Tuple

DATASET_SOURCES = {
    "general_english": {
        "name": "WikiText-2 Raw",
        "url": "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt",
        "license": "Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)",
        "license_source": "Salesforce Research WikiText Repository (https://blog.salesforce.com/wikitext-2-and-wikitext-103/)",
        "attribution_required": True,
        "restrictions": "ShareAlike (SA). Non-Commercial restriction: NO.",
        "file_name": "01_general_english.txt"
    },
    "educational_science": {
        "name": "Project Gutenberg — The Chemical History of a Candle (M. Faraday)",
        "url": "https://www.gutenberg.org/cache/epub/1447/pg1447.txt",
        "license": "Public Domain (US Public Domain, published prior to 1928)",
        "license_source": "Project Gutenberg License Terms (https://www.gutenberg.org/license)",
        "attribution_required": False,
        "restrictions": "None (Free commercial & non-commercial use).",
        "file_name": "02_educational_science.txt"
    },
    "conversations_qa": {
        "name": "Databricks Dolly-15k",
        "url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl",
        "license": "Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)",
        "license_source": "Databricks Labs Official Repository LICENSE (https://huggingface.co/datasets/databricks/databricks-dolly-15k)",
        "attribution_required": True,
        "restrictions": "ShareAlike (SA). Non-Commercial restriction: NO.",
        "file_name": "03_conversations_qa.txt"
    },
    "programming_tech": {
        "name": "Python PEP Specifications (PEP 8 & PEP 20)",
        "urls": [
            "https://raw.githubusercontent.com/python/peps/main/peps/pep-0008.rst",
            "https://raw.githubusercontent.com/python/peps/main/peps/pep-0020.rst"
        ],
        "license": "Public Domain (Explicit header statement in PEP 8 and PEP 20)",
        "license_source": "Python PEPs Official Repository (https://github.com/python/peps/blob/main/peps/pep-0008.rst)",
        "attribution_required": False,
        "restrictions": "None.",
        "file_name": "04_programming_tech.txt"
    }
}


def download_url(url: str, description: str) -> str:
    """Downloads raw text content from a verified HTTP URL."""
    print(f"[*] Fetching: {description}...")
    headers = {"User-Agent": "Mozilla/5.0 (CrachoLM Benchmark Data Prep)"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read().decode("utf-8", errors="replace")
    return content


def process_general_english(output_path: str, smoke_test: bool) -> str:
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        print(f"[✓] Reusing existing file: {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            return f.read()

    info = DATASET_SOURCES["general_english"]
    raw = download_url(info["url"], info["name"])
    lines = [line.strip() for line in raw.splitlines() if line.strip() and not line.startswith("=")]
    text = "\n\n".join(lines)
    if smoke_test:
        text = text[:80000]
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def process_educational_science(output_path: str, smoke_test: bool) -> str:
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        print(f"[✓] Reusing existing file: {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            return f.read()

    info = DATASET_SOURCES["educational_science"]
    raw = download_url(info["url"], info["name"])
    start_idx = raw.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    if start_idx != -1:
        raw = raw[raw.find("\n", start_idx) + 1:]
    end_idx = raw.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    if end_idx != -1:
        raw = raw[:end_idx]
    
    clean_text = raw.strip()
    if smoke_test:
        clean_text = clean_text[:80000]
        
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(clean_text)
    return clean_text


def process_conversations_qa(output_path: str, smoke_test: bool) -> str:
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        print(f"[✓] Reusing existing file: {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            return f.read()

    info = DATASET_SOURCES["conversations_qa"]
    raw_content = download_url(info["url"], info["name"])
    
    entries = []
    max_items = 250 if smoke_test else 2500

    for line_str in raw_content.splitlines():
        line_str = line_str.strip()
        if not line_str:
            continue
        try:
            item = json.loads(line_str)
            instruction = item.get("instruction", "").strip()
            context = item.get("context", "").strip()
            response = item.get("response", "").strip()

            dialogue = []
            if context:
                dialogue.append(f"Instruction Context: {context}")
            dialogue.append(f"User: {instruction}")
            dialogue.append(f"Assistant: {response}")
            
            entries.append("\n".join(dialogue))
            if len(entries) >= max_items:
                break
        except Exception:
            continue

    text = "\n\n".join(entries)
    if smoke_test and len(text) > 80000:
        text = text[:80000]
        
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def process_programming_tech(output_path: str, smoke_test: bool) -> str:
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        print(f"[✓] Reusing existing file: {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            return f.read()

    info = DATASET_SOURCES["programming_tech"]
    texts = []
    for url in info["urls"]:
        doc_name = url.split("/")[-1]
        raw = download_url(url, f"Python PEP ({doc_name})")
        clean_text = re.sub(r"^[=\-\*\~]{3,}$", "", raw, flags=re.MULTILINE)
        texts.append(clean_text.strip())
    
    combined = "\n\n".join(texts)
    if smoke_test and len(combined) > 75000:
        combined = combined[:75000]
        
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(combined)
    return combined


def main():
    parser = argparse.ArgumentParser(description="Prepare CrachoLM-General-v2 Benchmark Dataset")
    parser.add_argument("--output-dir", type=str, default="data_general/benchmark_raw", help="Directory to save benchmark corpus")
    parser.add_argument("--smoke-test", action="store_true", help="Run Phase 1 smoke-test preparation (~50k-100k tokens)")
    args = parser.parse_args()

    mode_label = "Phase 1 (Smoke-Test ~50k-100k tokens)" if args.smoke_test else "Phase 2 (Full ~800k tokens)"
    print("=" * 68)
    print(f" 🚀 CrachoLM-General-v2 Benchmark Data Preparation (Verified & Resumable)")
    print(f" Mode       : {mode_label}")
    print(f" Output Dir : {args.output_dir}")
    print("=" * 68 + "\n")

    os.makedirs(args.output_dir, exist_ok=True)

    file_results = []
    total_chars = 0
    download_errors = []

    # 1. General English
    try:
        path1 = os.path.join(args.output_dir, DATASET_SOURCES["general_english"]["file_name"])
        txt1 = process_general_english(path1, args.smoke_test)
        est_tokens = len(txt1) // 4
        total_chars += len(txt1)
        file_results.append((DATASET_SOURCES["general_english"]["file_name"], len(txt1), est_tokens, DATASET_SOURCES["general_english"]))
        print(f"[✓] Prepared {DATASET_SOURCES['general_english']['file_name']} ({len(txt1):,} chars, ~{est_tokens:,} tokens)")
    except Exception as e:
        print(f"[X] Error processing General English: {e}")
        download_errors.append(f"General English: {e}")

    # 2. Educational Science
    try:
        path2 = os.path.join(args.output_dir, DATASET_SOURCES["educational_science"]["file_name"])
        txt2 = process_educational_science(path2, args.smoke_test)
        est_tokens = len(txt2) // 4
        total_chars += len(txt2)
        file_results.append((DATASET_SOURCES["educational_science"]["file_name"], len(txt2), est_tokens, DATASET_SOURCES["educational_science"]))
        print(f"[✓] Prepared {DATASET_SOURCES['educational_science']['file_name']} ({len(txt2):,} chars, ~{est_tokens:,} tokens)")
    except Exception as e:
        print(f"[X] Error processing Educational Science: {e}")
        download_errors.append(f"Educational Science: {e}")

    # 3. Conversations Q&A
    try:
        path3 = os.path.join(args.output_dir, DATASET_SOURCES["conversations_qa"]["file_name"])
        txt3 = process_conversations_qa(path3, args.smoke_test)
        est_tokens = len(txt3) // 4
        total_chars += len(txt3)
        file_results.append((DATASET_SOURCES["conversations_qa"]["file_name"], len(txt3), est_tokens, DATASET_SOURCES["conversations_qa"]))
        print(f"[✓] Prepared {DATASET_SOURCES['conversations_qa']['file_name']} ({len(txt3):,} chars, ~{est_tokens:,} tokens)")
    except Exception as e:
        print(f"[X] Error processing Conversations Q&A: {e}")
        download_errors.append(f"Conversations Q&A: {e}")

    # 4. Programming & Technical
    try:
        path4 = os.path.join(args.output_dir, DATASET_SOURCES["programming_tech"]["file_name"])
        txt4 = process_programming_tech(path4, args.smoke_test)
        est_tokens = len(txt4) // 4
        total_chars += len(txt4)
        file_results.append((DATASET_SOURCES["programming_tech"]["file_name"], len(txt4), est_tokens, DATASET_SOURCES["programming_tech"]))
        print(f"[✓] Prepared {DATASET_SOURCES['programming_tech']['file_name']} ({len(txt4):,} chars, ~{est_tokens:,} tokens)")
    except Exception as e:
        print(f"[X] Error processing Programming & Tech: {e}")
        download_errors.append(f"Programming Tech: {e}")

    # Write SOURCES_AND_LICENSES.md metadata file
    meta_path = os.path.join(args.output_dir, "SOURCES_AND_LICENSES.md")
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("# CrachoLM-General-v2 Benchmark Corpus Data Sources & Licenses (Verified)\n\n")
        f.write(f"**Preparation Mode**: {mode_label}\n")
        f.write(f"**Total Character Count**: {total_chars:,} characters\n")
        f.write(f"**Estimated Total Tokens**: ~{total_chars // 4:,} subword tokens\n\n")
        f.write("## Dataset Manifest\n\n")
        f.write("| File Name | Dataset Source | License | Attribution Required? | Non-Commercial? | ShareAlike? |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for fname, chars, tokens, info in file_results:
            attr_str = "Yes" if info["attribution_required"] else "No"
            f.write(f"| `{fname}` | {info['name']} | {info['license']} | {attr_str} | No | {'Yes' if 'ShareAlike' in info['license'] else 'No'} |\n")
        
        f.write("\n## Detailed Source & License Documentation\n\n")
        for fname, chars, tokens, info in file_results:
            f.write(f"### `{fname}`\n")
            f.write(f"- **Source Name**: {info['name']}\n")
            if "url" in info:
                f.write(f"- **Exact Download URL**: `{info['url']}`\n")
            else:
                f.write(f"- **Exact Download URLs**: `{', '.join(info['urls'])}`\n")
            f.write(f"- **Exact License**: {info['license']}\n")
            f.write(f"- **Source of License Info**: {info['license_source']}\n")
            f.write(f"- **Attribution Required**: {'Yes' if info['attribution_required'] else 'No'}\n")
            f.write(f"- **License Restrictions**: {info['restrictions']}\n")
            f.write(f"- **Character Count**: {chars:,}\n")
            f.write(f"- **Estimated Tokens**: ~{tokens:,}\n\n")

    print(f"\n[✓] Created SOURCES_AND_LICENSES.md at: {meta_path}")

    total_est_tokens = total_chars // 4
    print("\n" + "=" * 68)
    print(" 📊 PHASE 1 BENCHMARK DATASET PREPARATION SUMMARY")
    print("=" * 68)
    print(f" Output Directory      : {args.output_dir}")
    print(f" Total Files Created   : {len(file_results)} text files + 1 metadata file")
    print(f" Total Character Count : {total_chars:,} characters")
    print(f" Estimated Subword Tok : ~{total_est_tokens:,} tokens")
    print(f" Download Errors       : {len(download_errors)} errors")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
