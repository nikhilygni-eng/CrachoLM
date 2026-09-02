"""
CrachoLM-General-v2 Data Preparation Pipeline Module
=====================================================
Handles multi-domain raw text loading, dataset validation statistics,
subword BPE tokenization, non-overlapping sequence dataset creation,
and PyTorch DataLoader generation.
"""

import os
import sys
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional

# Ensure project root is in sys.path when module is executed directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bpe_tokenizer import CrachoBPETokenizer


def load_and_combine_general_corpus(raw_dir: str) -> Tuple[str, Dict[str, int]]:
    """
    Safely loads and combines all .txt files found in raw_dir into a single corpus string.
    
    Args:
        raw_dir (str): Directory containing raw .txt corpus files.
        
    Returns:
        Tuple[str, Dict[str, int]]: Combined text and file metadata dictionary.
    """
    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir, exist_ok=True)
        print(f"[!] Warning: Created raw corpus directory at '{raw_dir}'. Place .txt files there.")

    file_paths = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
    if not file_paths:
        raise FileNotFoundError(f"No .txt corpus files found in directory: {raw_dir}")

    combined_texts = []
    file_stats = {}

    for path in file_paths:
        file_name = os.path.basename(path)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read().strip()
        if content:
            combined_texts.append(content)
            file_stats[file_name] = len(content)

    full_corpus = "\n<eos>\n".join(combined_texts)
    if not full_corpus.strip():
        raise ValueError(f"All dataset files in '{raw_dir}' are completely empty!")

    metadata = {
        "num_files": len(file_paths),
        "total_raw_chars": len(full_corpus),
        "file_details": file_stats
    }
    return full_corpus, metadata


class CrachoGeneralDataset(Dataset):
    """
    PyTorch Dataset for Next-Token Prediction with Non-Overlapping Chunking.
    
    Guarantees every token is trained on exactly once per epoch without
    wasteful window overlap.
    """

    def __init__(self, token_ids: List[int], seq_len: int):
        self.seq_len = seq_len
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)

        if len(self.token_ids) <= self.seq_len:
            raise ValueError(
                f"Insufficient tokens ({len(self.token_ids)}) for sequence length ({self.seq_len})."
            )

    def __len__(self) -> int:
        return (len(self.token_ids) - 1) // self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        x = self.token_ids[start : start + self.seq_len]
        y = self.token_ids[start + 1 : start + self.seq_len + 1]
        return x, y


def prepare_general_dataloaders(
    text_data: str,
    tokenizer: CrachoBPETokenizer,
    seq_len: int = 256,
    batch_size: int = 4,
    val_split: float = 0.10,
    shuffle_train: bool = True,
    num_workers: int = 0,
    raw_dir: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, Dict[str, any]]:
    """
    Constructs train/val DataLoaders with stratified per-domain 90/10 splitting.
    Splits each domain file independently before combining to prevent domain bias in validation.
    Preserves <eos> boundaries between domains.
    """
    domain_stats = {}
    train_tokens = []
    val_tokens = []

    # 1. Stratified domain splitting
    if raw_dir and os.path.exists(raw_dir) and glob.glob(os.path.join(raw_dir, "*.txt")):
        file_paths = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
        for path in file_paths:
            fname = os.path.basename(path)
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                d_content = f.read().strip()
            if not d_content:
                continue
            d_toks = tokenizer.encode(d_content, add_special_tokens=False)
            d_val_size = max(1, int(len(d_toks) * val_split))
            d_train_size = len(d_toks) - d_val_size

            d_tr = d_toks[:d_train_size]
            d_va = d_toks[d_train_size:]

            # Ensure boundary <eos> tag exists at boundary of split domain stream
            if d_tr and d_tr[-1] != tokenizer.eos_id:
                d_tr.append(tokenizer.eos_id)
            if d_va and d_va[-1] != tokenizer.eos_id:
                d_va.append(tokenizer.eos_id)

            train_tokens.extend(d_tr)
            val_tokens.extend(d_va)

            domain_stats[fname] = {
                "total_tokens": len(d_toks),
                "train_tokens": len(d_tr),
                "val_tokens": len(d_va)
            }
    else:
        # Fallback if raw_dir not passed: encode single text_data
        token_ids = tokenizer.encode(text_data, add_special_tokens=False)
        val_size = max(seq_len + 1, int(len(token_ids) * val_split))
        train_size = len(token_ids) - val_size
        train_tokens = token_ids[:train_size]
        val_tokens = token_ids[train_size:]

    total_tokens = len(train_tokens) + len(val_tokens)
    unk_count = train_tokens.count(tokenizer.unk_id) + val_tokens.count(tokenizer.unk_id)
    unk_rate = (unk_count / total_tokens * 100.0) if total_tokens > 0 else 0.0

    # 2. Create Datasets & Loaders
    train_dataset = CrachoGeneralDataset(train_tokens, seq_len=seq_len)
    val_dataset = CrachoGeneralDataset(val_tokens, seq_len=seq_len)

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
        "total_tokens": total_tokens,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "vocab_size": tokenizer.vocab_size,
        "unk_count": unk_count,
        "unk_rate_percent": round(unk_rate, 4),
        "domain_stats": domain_stats
    }

    return train_loader, val_loader, stats


if __name__ == "__main__":
    raw_dir = "data_general/raw"
    text, meta = load_and_combine_general_corpus(raw_dir)
    print(f"Loaded Corpus Meta: {meta}")

    tok = CrachoBPETokenizer(target_vocab_size=256)
    tok.train_from_text(text)

    tr_loader, va_loader, stats = prepare_general_dataloaders(text, tok, seq_len=64, batch_size=2)
    print("\nDataset Validation Report:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
