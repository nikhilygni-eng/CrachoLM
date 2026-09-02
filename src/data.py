"""
CrachoLM Data Preparation & Pipeline Module
============================================
Handles raw text loading, text cleaning, tokenization, sequence dataset creation,
train/validation splitting, and PyTorch DataLoader generation.
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional

from src.tokenizer import CrachoTokenizer


def clean_text(raw_text: str) -> str:
    """
    Cleans raw text data without destroying linguistic structure.
    
    Processing Steps:
    - Normalizes Windows carriage returns (\r\n -> \n)
    - Strips non-printable ASCII/Unicode control codes while preserving newlines & tabs
    - Collapses 3+ consecutive newlines down to 2 (paragraph boundary)
    
    Args:
        raw_text (str): Raw input string.
        
    Returns:
        str: Cleaned text string.
    """
    if not raw_text:
        return ""

    # Normalize carriage returns
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    # Filter non-printable characters (preserve \n, \t, and standard printable chars)
    cleaned_chars = [ch for ch in text if ch.isprintable() or ch in ("\n", "\t")]
    text = "".join(cleaned_chars)

    # Collapse excessive empty lines (more than 2 consecutive newlines)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    return text.strip()


class CrachoDataset(Dataset):
    """
    PyTorch Dataset for Next-Token Prediction Autoregressive Training.
    
    Each sample index `i` yields:
        Input (x)  : [token_i,   token_i+1, ..., token_i+seq_len-1]
        Target (y) : [token_i+1, token_i+2, ..., token_i+seq_len  ]
    """

    def __init__(self, token_ids: List[int], seq_len: int):
        """
        Args:
            token_ids (List[int]): Complete list of token IDs in corpus.
            seq_len (int): Length of each sequence example (context window).
        """
        self.seq_len = seq_len
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)

        # Safety Check: Insufficient text check
        if len(self.token_ids) <= self.seq_len:
            raise ValueError(
                f"Insufficient text tokens ({len(self.token_ids)}) for requested sequence length ({self.seq_len}). "
                f"Corpus token count must be strictly greater than seq_len."
            )

    def __len__(self) -> int:
        return (len(self.token_ids) - 1) // self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len

        x = self.token_ids[start : start + self.seq_len]
        y = self.token_ids[start + 1 : start + self.seq_len + 1]

        return x, y


def load_raw_text(file_path: str) -> str:
    """
    Loads raw text file with safety validations.
    
    Args:
        file_path (str): Path to raw text file.
        
    Returns:
        str: Raw text content.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw text file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Safety Check: Empty dataset check
    if not text or len(text.strip()) == 0:
        raise ValueError(f"Dataset file at '{file_path}' is completely empty!")

    return text


def prepare_dataloaders(
    text_data: str,
    tokenizer: CrachoTokenizer,
    seq_len: int = 512,
    batch_size: int = 16,
    val_split: float = 0.1,
    shuffle_train: bool = True,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, Dict[str, int]]:
    """
    Full pipeline to clean, tokenize, split, and construct DataLoaders.
    
    Args:
        text_data (str): Raw text dataset string.
        tokenizer (CrachoTokenizer): Trained tokenizer instance.
        seq_len (int): Sequence context length.
        batch_size (int): Micro-batch size.
        val_split (float): Fraction of tokens reserved for validation (e.g. 0.1 = 10%).
        shuffle_train (bool): Whether to shuffle training sequence samples.
        num_workers (int): DataLoader parallel worker processes.
        
    Returns:
        Tuple[DataLoader, DataLoader, Dict[str, int]]:
            - train_loader
            - val_loader
            - dataset_stats metadata dictionary
    """
    # 1. Clean Text
    cleaned_text = clean_text(text_data)
    
    # 2. Tokenize Text
    token_ids = tokenizer.encode(cleaned_text, add_special_tokens=False)

    # 3. Safety Check: Vocabulary mismatch check
    max_token_id = max(token_ids) if token_ids else 0
    if max_token_id >= tokenizer.vocab_size:
        raise ValueError(
            f"Tokenizer Vocabulary Mismatch! Max token ID in corpus ({max_token_id}) "
            f"exceeds tokenizer vocab size ({tokenizer.vocab_size})."
        )

    # 4. Safety Check: Insufficient text check
    min_required = seq_len + 1
    if len(token_ids) < min_required:
        raise ValueError(
            f"Insufficient dataset text! Total tokens = {len(token_ids)}, but minimum required "
            f"for sequence length {seq_len} is {min_required} tokens."
        )

    # 5. Train / Validation Split
    val_size = int(len(token_ids) * val_split)
    train_size = len(token_ids) - val_size

    train_tokens = token_ids[:train_size]
    val_tokens = token_ids[train_size:]

    # Verify validation set has enough tokens, adjust split if needed
    if len(val_tokens) <= seq_len and val_split > 0:
        # Fallback: copy last segment for validation if corpus is small
        val_tokens = token_ids[-min_required:]

    # 6. Create PyTorch Datasets
    train_dataset = CrachoDataset(train_tokens, seq_len=seq_len)
    val_dataset = CrachoDataset(val_tokens, seq_len=seq_len)

    # 7. Create PyTorch DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    stats = {
        "raw_char_count": len(text_data),
        "clean_char_count": len(cleaned_text),
        "total_tokens": len(token_ids),
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "vocab_size": tokenizer.vocab_size,
    }

    return train_loader, val_loader, stats


if __name__ == "__main__":
    # Self-test code
    sample_text = "CrachoLM educational dataset. " * 100
    tokenizer = CrachoTokenizer()
    tokenizer.train_from_text(sample_text)
    
    train_loader, val_loader, stats = prepare_dataloaders(
        text_data=sample_text,
        tokenizer=tokenizer,
        seq_len=32,
        batch_size=4,
        val_split=0.1
    )
    
    print("Data Pipeline Self-Test Summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    for x_batch, y_batch in train_loader:
        print(f"\nBatch Shapes -> Input x: {x_batch.shape}, Target y: {y_batch.shape}")
        print(f"Sample x[0][:5]: {x_batch[0][:5].tolist()}")
        print(f"Sample y[0][:5]: {y_batch[0][:5].tolist()}")
        # Verify target is shifted right by 1 position
        assert (x_batch[0][1:5] == y_batch[0][0:4]).all(), "Target shift validation failed!"
        print("Target Shift Verification PASSED!")
        break
