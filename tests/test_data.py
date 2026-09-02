"""
CrachoLM Data Pipeline Test Suite
==================================
Tests and validates:
1. Loading raw text from data/raw/
2. Text cleaning and normalization
3. Custom tokenizer integration
4. Next-Token Prediction sequence target shifting (Target = Input shifted right by 1)
5. PyTorch DataLoader batching (Input shape & Target shape)
6. Safety validation checks (empty file, insufficient tokens, vocab mismatch)
"""

import os
import sys
import tempfile
import torch

# Add project root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tokenizer import CrachoTokenizer
from src.data import (
    clean_text,
    load_raw_text,
    CrachoDataset,
    prepare_dataloaders,
)


def test_data_pipeline():
    print("=" * 68)
    print("            CrachoLM Phase 4: Data Pipeline Test Suite")
    print("=" * 68)

    # 1. Load Raw Text Sample
    sample_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "sample_corpus.txt"))
    print(f"[1] Loading Raw Text File: {sample_file}")
    raw_text = load_raw_text(sample_file)
    print(f"    [✓] Raw Character Count: {len(raw_text):,} chars")

    # 2. Test Text Cleaning
    print(f"[2] Cleaning & Normalizing Raw Text...")
    cleaned = clean_text(raw_text)
    print(f"    [✓] Cleaned Character Count: {len(cleaned):,} chars")
    assert len(cleaned) > 0, "Error: Cleaned text is empty!"
    print("-" * 68)

    # 3. Train Tokenizer & Convert Text to Token IDs
    print(f"[3] Tokenizing Text with CrachoTokenizer...")
    tokenizer = CrachoTokenizer()
    tokenizer.train_from_text(cleaned)
    print(f"    [✓] Tokenizer Vocab Size: {tokenizer.vocab_size} tokens")

    # 4. DataLoader Construction & Configurable Sequence Length
    seq_len = 64
    batch_size = 4
    print(f"[4] Constructing DataLoaders (seq_len={seq_len}, batch_size={batch_size})...")

    train_loader, val_loader, stats = prepare_dataloaders(
        text_data=cleaned,
        tokenizer=tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
        val_split=0.15
    )

    print("    Dataset Statistics:")
    print(f"      - Total Corpus Tokens : {stats['total_tokens']:,}")
    print(f"      - Train Sequence Samples: {stats['train_samples']:,}")
    print(f"      - Val Sequence Samples  : {stats['val_samples']:,}")
    print(f"      - Train Batches / Epoch : {len(train_loader)}")
    print(f"      - Val Batches / Epoch   : {len(val_loader)}")
    print("-" * 68)

    # 5. Target Shift & Next-Token Prediction Verification
    print(f"[5] Verifying Autoregressive Target Shift (Input vs Target)...")
    for x_batch, y_batch in train_loader:
        print(f"    Batch Input Tensor Shape  (x): {tuple(x_batch.shape)}")
        print(f"    Batch Target Tensor Shape (y): {tuple(y_batch.shape)}")
        
        # Verify shapes
        assert x_batch.shape == (batch_size, seq_len), "Error: Input batch shape mismatch!"
        assert y_batch.shape == (batch_size, seq_len), "Error: Target batch shape mismatch!"
        
        # Verify target shift: y[0][t] MUST equal x[0][t+1]
        input_seq = x_batch[0]
        target_seq = y_batch[0]

        # Shifted slice match
        shift_match = (input_seq[1:] == target_seq[:-1]).all().item()
        assert shift_match, "Error: Target is not correctly shifted right by 1 position!"
        
        # Decode first sequence example for human review
        sample_input_text = tokenizer.decode(input_seq[:10].tolist())
        sample_target_text = tokenizer.decode(target_seq[:10].tolist())
        print(f"    Sample Input  Text (first 10 tokens): '{sample_input_text}'")
        print(f"    Sample Target Text (first 10 tokens): '{sample_target_text}'")
        print(f"    [✓] Target Shift Verified! (Input token i+1 == Target token i)")
        break

    print("-" * 68)

    # 6. Safety Checks & Exception Handling Tests
    print(f"[6] Testing Safety Validation Checks...")
    
    # Test A: Empty Dataset Check
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as empty_file:
        empty_file.write("   \n   \n")
        empty_path = empty_file.name
    try:
        load_raw_text(empty_path)
        assert False, "Safety Check Failed: Empty dataset did not raise ValueError!"
    except ValueError as e:
        print(f"    [✓] Safety Check A Passed (Empty File): Caught -> {e}")
    finally:
        if os.path.exists(empty_path):
            os.remove(empty_path)

    # Test B: Insufficient Text Check
    try:
        short_text = "Hi" # 2 tokens
        CrachoDataset(tokenizer.encode(short_text), seq_len=512)
        assert False, "Safety Check Failed: Insufficient text did not raise ValueError!"
    except ValueError as e:
        print(f"    [✓] Safety Check B Passed (Insufficient Text): Caught -> {e}")

    # Test C: Vocabulary Mismatch Check
    try:
        invalid_tokens = [99999] # Invalid token ID far exceeding vocab size
        small_tok = CrachoTokenizer()
        small_tok.train_from_text("abc") # Vocab size ~7
        # Pass raw text containing unmapped token IDs exceeding vocab
        class MismatchTokenizer(CrachoTokenizer):
            def encode(self, text, add_special_tokens=False):
                return [99999] * 100
        prepare_dataloaders("abc" * 50, MismatchTokenizer(), seq_len=10)
        assert False, "Safety Check Failed: Vocab mismatch did not raise ValueError!"
    except ValueError as e:
        print(f"    [✓] Safety Check C Passed (Vocab Mismatch): Caught -> {e}")

    print("=" * 68)
    print(" Data Pipeline Test Suite Completed PASSED (100% SUCCESS)!")
    print("=" * 68)


if __name__ == "__main__":
    test_data_pipeline()
