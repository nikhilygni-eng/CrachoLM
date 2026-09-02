#!/usr/bin/env python3
"""
Quick generation test for CrachoLM-General-v2 post-fix validation.
Runs two prompts and prints exact output.
"""
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.device import get_device
from src.bpe_tokenizer import CrachoBPETokenizer
from src.model import CrachoLM
from src.generator import generate_text
from src.checkpoint import load_checkpoint_file, get_model_config_from_checkpoint

CHECKPOINT = "checkpoints_general_benchmark/best_model.pt"
TOKENIZER  = "checkpoints_general_benchmark/tokenizer_bpe.json"

PROMPTS = [
    "The purpose of science is",
    "User: What is Python?\nAssistant:",
]

device = get_device()
tokenizer = CrachoBPETokenizer.load(TOKENIZER)
checkpoint = load_checkpoint_file(CHECKPOINT, device)
model_config = get_model_config_from_checkpoint(checkpoint)
model = CrachoLM(model_config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"[✓] Model loaded — {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"[✓] Tokenizer vocab size: {tokenizer.vocab_size}")
print(f"[✓] Special token IDs — PAD:{tokenizer.pad_id} UNK:{tokenizer.unk_id} BOS:{tokenizer.bos_id} EOS:{tokenizer.eos_id}")
print()

for i, prompt in enumerate(PROMPTS, 1):
    output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=80,
        temperature=0.7,
        top_k=40,
        greedy=False,
        device=device
    )
    print(f"{'=' * 68}")
    print(f" TEST {i} — Prompt: {repr(prompt)}")
    print(f"{'=' * 68}")
    print(output)
    print()

    # Check for special token strings in output
    bad_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]
    found = [t for t in bad_tokens if t in output]
    if found:
        print(f"[FAIL] Special token strings found in output: {found}")
    else:
        print(f"[PASS] No special token strings in output.")
    print()
