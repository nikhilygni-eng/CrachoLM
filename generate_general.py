#!/usr/bin/env python3
"""
CrachoLM-General-v2 Inference & Text Generation CLI
===================================================
Generates text using the trained CrachoLM-General-v2 model and Subword BPE Tokenizer.
Includes strict checkpoint-tokenizer vocabulary validation to prevent broken outputs.

Usage:
  python3 generate_general.py --prompt "Question: What is machine learning?"
  python3 generate_general.py --prompt "def fibonacci(" --temperature 0.7 --top-k 40
"""

import argparse
import os
import sys
import torch

# Ensure root folder is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.device import get_device
from src.bpe_tokenizer import CrachoBPETokenizer
from src.model import CrachoLM
from src.generator import generate_text
from src.checkpoint import load_checkpoint_file, get_model_config_from_checkpoint
from src.inference import load_model_for_inference, DEFAULT_CHECKPOINT


def parse_args():
    parser = argparse.ArgumentParser(description="CrachoLM-General-v2 Text Generation CLI")
    parser.add_argument("--prompt", type=str, default="User: What is artificial intelligence?\nAssistant:", help="Prompt text")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT), help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default=None, help="Training tokenizer JSON (default: beside checkpoint)")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-K sampling cutoff")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding (ArgMax)")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Penalty factor for repeated tokens (> 1.0 reduces repetition)")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0, help="Size of n-grams that cannot be repeated (> 0 disables duplicate n-grams)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    print(f"[*] Compute Device : {device}")
    print(f"[*] BPE Tokenizer  : {args.tokenizer}")
    print(f"[*] Model Checkpoint: {args.checkpoint}")

    model, tokenizer, info = load_model_for_inference(
        args.checkpoint, args.tokenizer, device
    )
    print(f"[✓] Loaded {info['model_name']} ({info['parameters_m']}M parameters)")
    print(f"[*] Paired tokenizer: {info['tokenizer']}")

    print("-" * 68)
    print("Generating output...")
    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        greedy=args.greedy,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        device=device
    )

    print("\n====================================================================")
    print("                    CrachoLM-General-v2 Output")
    print("====================================================================")
    print(output)
    print("====================================================================\n")


if __name__ == "__main__":
    main()
