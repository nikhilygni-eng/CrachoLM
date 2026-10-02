#!/usr/bin/env python3
"""
CrachoLM Interactive Chat CLI
=============================
Type any prompt or question in terminal to test model responses interactively.
"""

import argparse
import os
import sys
import torch

from src.device import get_device
from src.tokenizer import CrachoTokenizer
from src.model import CrachoLM
from src.generator import generate_text
from src.chat_response import generate_chat_reply
from src.checkpoint import load_checkpoint_file, get_model_config_from_checkpoint
from src.inference import load_model_for_inference


def main():
    parser = argparse.ArgumentParser(description="Chat with CrachoLM")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tokenizer", default=None)
    args = parser.parse_args()
    device = get_device()
    model, tokenizer, info = load_model_for_inference(
        args.checkpoint, args.tokenizer, device
    )
    print(f"Loaded {info['model_name']} ({info['parameters_m']}M parameters)")

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
            output, _ = generate_chat_reply(
                model, tokenizer, prompt, max_new_tokens=80,
                temperature=0.7, top_k=40, device=device,
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
