#!/usr/bin/env python3
"""
CrachoLM Interactive Chat CLI
=============================
Type any prompt or question in terminal to test model responses interactively.
"""

import os
import sys
import torch

from src.device import get_device
from src.tokenizer import CrachoTokenizer
from src.model import CrachoLM
from src.generator import generate_text
from src.checkpoint import load_checkpoint_file, get_model_config_from_checkpoint


def main():
    device = get_device()
    tokenizer_path = os.path.abspath("checkpoints/tokenizer.json")
    checkpoint_path = os.path.abspath("checkpoints/best_model.pt")

    if not os.path.exists(tokenizer_path) or not os.path.exists(checkpoint_path):
        print("[X] ERROR: Missing checkpoint or tokenizer. Please run train.py first.")
        return

    tokenizer = CrachoTokenizer.load(tokenizer_path)
    checkpoint = load_checkpoint_file(checkpoint_path, device)
    model_config = get_model_config_from_checkpoint(checkpoint)

    model = CrachoLM(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("\n" + "=" * 60)
    print("      CrachoLM Interactive Chat (Type 'exit' to quit)")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input("You > ").strip()
            if prompt.lower() in ("exit", "quit"):
                print("Exiting chat. Goodbye!")
                break
            if not prompt:
                continue

            print("\nCrachoLM is thinking...\n")
            output = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=100,
                temperature=0.7,
                top_k=40,
                device=device
            )
            print("=" * 60)
            print("CrachoLM Response:")
            print("=" * 60)
            print(output)
            print("=" * 60 + "\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Goodbye!")
            break


if __name__ == "__main__":
    main()
