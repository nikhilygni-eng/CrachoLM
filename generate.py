#!/usr/bin/env python3
"""
CrachoLM Text Generation CLI Script
====================================
Generates text autoregressively using a trained CrachoLM model checkpoint.

Examples:
  python3 generate.py --prompt "CrachoLM"
  python3 generate.py --prompt "Transformer" --temperature 0.7 --top-k 30
  python3 generate.py --prompt "Language models" --greedy
"""

import argparse
import os
import sys
import torch

from src.device import get_device
from src.tokenizer import CrachoTokenizer
from src.model import CrachoLM
from src.generator import generate_text
from src.checkpoint import load_checkpoint_file, get_model_config_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="CrachoLM Text Generation CLI")
    parser.add_argument(
        "--prompt",
        type=str,
        default="CrachoLM",
        help="Text prompt string to start generation"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to trained model checkpoint file (.pt)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="checkpoints/tokenizer.json",
        help="Path to tokenizer file (.json)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (higher = more creative, 0.0 = greedy)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-K sampling filter (retains only top-K probable tokens)"
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="Enable deterministic greedy decoding (argmax)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Setup Compute Device
    device = get_device()
    print(f"[*] Target Compute Device: {device}")

    # 2. Safety Check A: Tokenizer File Existence
    tokenizer_path = os.path.abspath(args.tokenizer)
    if not os.path.exists(tokenizer_path):
        print(f"[X] ERROR: Tokenizer file not found at '{tokenizer_path}'.")
        print("    Please train the tokenizer first or run Phase 5 training.")
        sys.exit(1)

    print(f"[*] Loading Tokenizer from: {tokenizer_path}")
    tokenizer = CrachoTokenizer.load(tokenizer_path)

    # 3. Safety Check B: Checkpoint File Existence
    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.exists(checkpoint_path):
        print(f"[X] ERROR: Model checkpoint file not found at '{checkpoint_path}'.")
        print("    Please run training first (python3 train.py) to generate checkpoints/best_model.pt.")
        sys.exit(1)

    # 4. Load Checkpoint Safely
    # load_checkpoint_file uses weights_only=False — required for optimizer state;
    # safe here because the file is produced locally by train.py.
    print(f"[*] Loading Model Checkpoint from: {checkpoint_path}")
    try:
        checkpoint = load_checkpoint_file(checkpoint_path, device)
    except Exception as e:
        print(f"[X] ERROR: Failed to load model checkpoint file: {e}")
        sys.exit(1)

    # Restore Model Architecture Configuration from plain dict (no Config object)
    model_config = get_model_config_from_checkpoint(checkpoint)

    # 5. Safety Check C: Tokenizer & Checkpoint Vocab Mismatch Check
    if model_config.vocab_size != tokenizer.vocab_size:
        print(f"[X] ERROR: Vocabulary Size Mismatch!")
        print(f"    - Checkpoint Vocab Size: {model_config.vocab_size}")
        print(f"    - Tokenizer Vocab Size : {tokenizer.vocab_size}")
        print("    Please ensure you are using the exact tokenizer trained with this model checkpoint.")
        sys.exit(1)

    # Instantiate Model & Restore State Dict
    model = CrachoLM(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"[✓] Model & Tokenizer loaded successfully!")
    print(f"[*] Generation Hyperparameters:")
    print(f"    - Prompt         : '{args.prompt}'")
    print(f"    - Max New Tokens : {args.max_new_tokens}")
    print(f"    - Decoding Mode  : {'Greedy (ArgMax)' if (args.greedy or args.temperature == 0) else 'Stochastic Sampling'}")
    if not (args.greedy or args.temperature == 0):
        print(f"    - Temperature    : {args.temperature}")
        print(f"    - Top-K          : {args.top_k}")
    print("-" * 68)

    # 6. Execute Text Generation
    print("Generating output...\n")
    output_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        greedy=args.greedy,
        device=device
    )

    print("=" * 68)
    print("                    CrachoLM Output")
    print("=" * 68)
    print(output_text)
    print("=" * 68)


if __name__ == "__main__":
    main()
