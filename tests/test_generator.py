"""
CrachoLM Generation Module Diagnostic Test
==========================================
Tests text generation with temperature sampling, top-k filtering, and greedy decoding.
"""

import os
import sys
import torch

# Add project root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import ModelConfig
from src.device import get_device
from src.tokenizer import CrachoTokenizer
from src.model import CrachoLM
from src.generator import generate_text


def test_generation_pipeline():
    print("=" * 68)
    print("            CrachoLM Phase 6: Generator Test Suite")
    print("=" * 68)

    device = get_device()
    print(f"[1] Target Compute Device: {device}")

    # Build small tokenizer & model
    sample_text = "CrachoLM is an educational language model built from scratch in PyTorch."
    tokenizer = CrachoTokenizer()
    tokenizer.train_from_text(sample_text)

    config = ModelConfig(vocab_size=tokenizer.vocab_size, d_model=128, n_layers=2, n_heads=4, max_seq_len=64)
    model = CrachoLM(config).to(device)

    prompt = "CrachoLM"
    print(f"[2] Testing Greedy Decoding (temperature=0.0)...")
    greedy_output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=20,
        greedy=True,
        device=device
    )
    print(f"    Greedy Output  : '{greedy_output}'")
    assert greedy_output.startswith(prompt), "Error: Output does not start with prompt!"
    print(f"    [✓] Greedy Decoding Passed!")

    print(f"[3] Testing Stochastic Sampling (temperature=0.8, top_k=20)...")
    sample_output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=20,
        temperature=0.8,
        top_k=20,
        greedy=False,
        device=device
    )
    print(f"    Sampled Output : '{sample_output}'")
    assert sample_output.startswith(prompt), "Error: Output does not start with prompt!"
    print(f"    [✓] Stochastic Sampling Passed!")

    print("=" * 68)
    print(" Generator Test Suite Completed PASSED (100% SUCCESS)!")
    print("=" * 68)


if __name__ == "__main__":
    test_generation_pipeline()
