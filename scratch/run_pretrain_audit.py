import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bpe_tokenizer import CrachoBPETokenizer
from src.data_general import prepare_general_dataloaders, load_and_combine_general_corpus

RAW_DIR = "data_general/raw_scaled"
TOKENIZER_PATH = "checkpoints_general_v2/tokenizer_bpe.json"

def main():
    print("=" * 68)
    print(" 🔬 Real DataLoader & Pre-Training Audit")
    print("=" * 68)

    tokenizer = CrachoBPETokenizer.load(TOKENIZER_PATH)
    raw_text, corpus_meta = load_and_combine_general_corpus(RAW_DIR)

    train_loader, val_loader, stats = prepare_general_dataloaders(
        text_data=raw_text,
        tokenizer=tokenizer,
        seq_len=256,
        batch_size=4,
        val_split=0.10,
        shuffle_train=True,
        raw_dir=RAW_DIR
    )

    print("\n[STATISTICS RETURNED BY REAL CODE]:")
    print(f"  - len(train_dataset) : {stats['train_samples']}")
    print(f"  - len(val_dataset)   : {stats['val_samples']}")
    print(f"  - len(train_loader)  : {len(train_loader)}")
    print(f"  - len(val_loader)    : {len(val_loader)}")
    print(f"  - total_tokens       : {stats['total_tokens']:,}")
    print(f"  - train_tokens       : {stats['train_tokens']:,}")
    print(f"  - val_tokens         : {stats['val_tokens']:,}")

    # Inspect first few batches
    print("\n[FIRST FEW BATCHES TENSOR SHAPES & BOUND CHECKS]:")
    for i, (x, y) in enumerate(train_loader):
        print(f"  - Batch {i+1} x shape: {x.shape}, y shape: {y.shape}")
        assert x.shape == (4, 256), f"Expected shape (4, 256), got {x.shape}"
        assert y.shape == (4, 256), f"Expected shape (4, 256), got {y.shape}"
        
        # Token ID bounds check [0, 2047]
        min_x, max_x = x.min().item(), x.max().item()
        min_y, max_y = y.min().item(), y.max().item()
        assert 0 <= min_x <= max_x < tokenizer.vocab_size, f"Token ID out of bounds in x: [{min_x}, {max_x}]"
        assert 0 <= min_y <= max_y < tokenizer.vocab_size, f"Token ID out of bounds in y: [{min_y}, {max_y}]"
        print(f"    [✓] Token IDs bounded within [0, {tokenizer.vocab_size - 1}]: x=[{min_x}, {max_x}], y=[{min_y}, {max_y}]")
        
        if i >= 2:
            break

    print("\n" + "=" * 68)
    print(" 🎉 REAL DATALOADER AUDIT PASSED 100%!")
    print("=" * 68)

if __name__ == "__main__":
    main()
