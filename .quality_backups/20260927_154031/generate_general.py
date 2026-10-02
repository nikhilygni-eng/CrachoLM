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


def parse_args():
    parser = argparse.ArgumentParser(description="CrachoLM-General-v2 Text Generation CLI")
    parser.add_argument("--prompt", type=str, default="Question: What is artificial intelligence?", help="Prompt text")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_general/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="checkpoints_general/tokenizer_bpe.json", help="Path to BPE tokenizer JSON")
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

    if not os.path.exists(args.tokenizer):
        print(f"[X] ERROR: BPE Tokenizer file not found at '{args.tokenizer}'. Train model first.")
        sys.exit(1)

    if not os.path.exists(args.checkpoint):
        print(f"[X] ERROR: Checkpoint file not found at '{args.checkpoint}'. Train model first.")
        sys.exit(1)

    # 1. Load BPE Tokenizer
    tokenizer = CrachoBPETokenizer.load(args.tokenizer)

    # 2. Load Checkpoint
    checkpoint = load_checkpoint_file(args.checkpoint, device)
    model_config = get_model_config_from_checkpoint(checkpoint)

    # 3. STRICT VOCABULARY VALIDATION CHECK
    ckpt_vocab_size = model_config.vocab_size
    tok_vocab_size = tokenizer.vocab_size

    if ckpt_vocab_size != tok_vocab_size:
        print("\n" + "!" * 68)
        print(" [X] VOCABULARY SIZE MISMATCH ERROR!")
        print("!" * 68)
        print(f"  Model Checkpoint Vocab Size : {ckpt_vocab_size}")
        print(f"  BPE Tokenizer Vocab Size    : {tok_vocab_size}")
        print("  Reason: The loaded checkpoint was trained with a different tokenizer vocabulary!")
        print("  Action: Aborting text generation to prevent corrupted outputs.")
        print("!" * 68 + "\n")
        sys.exit(1)

    print(f"[✓] Vocabulary Validation PASSED: Model & Tokenizer match ({tok_vocab_size} tokens).")

    # 4. Instantiate Model & Load Weights
    model = CrachoLM(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[✓] Model Loaded Successfully! ({n_params:,} / {n_params/1e6:.2f}M parameters)")

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
