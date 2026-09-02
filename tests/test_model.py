"""
CrachoLM Model Architecture Test Suite
======================================
Tests and validates:
1. Model initialization from ModelConfig
2. Parameter count calculation
3. Forward pass with dummy input tensors
4. Output logit shape verification (Batch, SeqLen, VocabSize)
5. Cross-entropy loss calculation with targets
6. Hardware device execution (CUDA / CPU auto-selection)
7. Causal masking integrity test
"""

import os
import sys
import torch

# Add project root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import ModelConfig
from src.device import get_device, get_device_info
from src.model import CrachoLM


def test_model_architecture():
    print("=" * 68)
    print("            CrachoLM Phase 3: Model Architecture Test Suite")
    print("=" * 68)

    # 1. Setup Compute Device
    device = get_device()
    device_info = get_device_info()
    print(f"[1] Target Device selected: {device} ({device_info['device_name']})")
    print("-" * 68)

    # 2. Model Configuration
    vocab_size = 120
    d_model = 256
    n_layers = 6
    n_heads = 8
    max_seq_len = 512
    
    config = ModelConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=max_seq_len,
        dropout=0.1
    )

    print(f"[2] Initializing CrachoLM Model Architecture...")
    print(f"    - Vocab Size    : {config.vocab_size}")
    print(f"    - d_model       : {config.d_model}")
    print(f"    - Layers        : {config.n_layers}")
    print(f"    - Attention Heads: {config.n_heads}")
    print(f"    - Max Seq Len   : {config.max_seq_len}")

    model = CrachoLM(config).to(device)
    total_params = model.get_num_params()
    print(f"    [✓] Model initialized from scratch!")
    print(f"    [✓] Trainable Parameters: {total_params:,} ({total_params / 1e6:.2f} Million)")
    print("-" * 68)

    # 3. Dummy Forward Pass Test
    batch_size = 4
    seq_len = 32
    print(f"[3] Testing Dummy Tensor Forward Pass...")
    print(f"    Dummy Input Shape : Batch={batch_size}, SeqLen={seq_len}")

    # Generate random integer token IDs in range [0, vocab_size)
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # Forward pass without targets (inference mode)
    with torch.no_grad():
        logits_infer, loss_infer = model(dummy_input)

    print(f"    [✓] Inference Logits Shape: {tuple(logits_infer.shape)} -> Expected: ({batch_size}, 1, {vocab_size})")
    print(f"    [✓] Loss (Inference mode) : {loss_infer} (Expected: None)")

    # Forward pass with targets (training mode)
    dummy_targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    logits_train, loss_train = model(dummy_input, targets=dummy_targets)

    print(f"    [✓] Training Logits Shape : {tuple(logits_train.shape)} -> Expected: ({batch_size}, {seq_len}, {vocab_size})")
    print(f"    [✓] Cross-Entropy Loss     : {loss_train.item():.4f}")

    assert logits_train.shape == (batch_size, seq_len, vocab_size), "Error: Training logits shape mismatch!"
    print(f"    [✓] Output shape verification PASSED!")
    print("-" * 68)

    # 4. Causal Mask Integrity Test
    print(f"[4] Testing Causal Mask Integrity (No future information leakage)...")
    # Change the last token in sequence and check that earlier token logits remain identical
    input_seq_1 = torch.tensor([[10, 20, 30, 40, 50]], device=device)
    input_seq_2 = torch.tensor([[10, 20, 30, 40, 99]], device=device) # token 4 changed from 50 to 99

    model.eval()
    with torch.no_grad():
        logits_1, _ = model(input_seq_1, targets=input_seq_1)
        logits_2, _ = model(input_seq_2, targets=input_seq_2)

    # Compare logits for position 0, 1, 2, 3 (should be exact match)
    diff_first_four = torch.max(torch.abs(logits_1[0, :4, :] - logits_2[0, :4, :])).item()
    print(f"    Max Logit Diff for position 0..3: {diff_first_four:.8f}")
    assert diff_first_four < 1e-5, "Error: Causal masking failed! Future tokens leaked to earlier steps."
    print(f"    [✓] Causal Mask Integrity Verified! (No future token leakage)")

    print("=" * 68)
    print(" Model Architecture Test Suite Completed PASSED (100% SUCCESS)!")
    print("=" * 68)


if __name__ == "__main__":
    test_model_architecture()
