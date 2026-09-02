"""
CrachoLM Custom Tokenizer Module
=================================
A character-level tokenizer built from scratch for CrachoLM-0.1.
Handles vocabulary building, encoding, decoding, special token management,
and local persistence (save/load).
"""

import json
import os
from typing import List, Dict, Union, Optional


class CrachoTokenizer:
    """
    Educational Character-Level Tokenizer built 100% from scratch.
    
    Special Tokens:
        <pad>: Padding token (ID 0)
        <unk>: Unknown token (ID 1)
        <bos>: Beginning of sequence token (ID 2)
        <eos>: End of sequence token (ID 3)
    """
    
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    def __init__(self, special_tokens: Optional[List[str]] = None):
        """
        Initializes the tokenizer with standard special tokens.
        """
        if special_tokens is None:
            self.special_tokens = [
                self.PAD_TOKEN,
                self.UNK_TOKEN,
                self.BOS_TOKEN,
                self.EOS_TOKEN,
            ]
        else:
            self.special_tokens = special_tokens

        # ID Mappings
        self.char2idx: Dict[str, int] = {}
        self.idx2char: Dict[int, str] = {}
        
        # Populate special tokens into vocabulary first
        self._init_special_tokens()

    def _init_special_tokens(self):
        """Pre-populates vocabulary with special tokens."""
        for idx, token in enumerate(self.special_tokens):
            self.char2idx[token] = idx
            self.idx2char[idx] = token

    @property
    def pad_id(self) -> int:
        return self.char2idx[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.char2idx[self.UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.char2idx[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.char2idx[self.EOS_TOKEN]

    @property
    def vocab_size(self) -> int:
        """Returns total size of vocabulary including special tokens."""
        return len(self.char2idx)

    def train_from_text(self, text: str):
        """
        Builds vocabulary from a raw text dataset string.
        
        Args:
            text (str): Training dataset corpus.
        """
        # Collect unique characters present in the training corpus
        unique_chars = sorted(list(set(text)))
        
        # Assign IDs starting after special tokens
        start_idx = len(self.special_tokens)
        for i, ch in enumerate(unique_chars):
            if ch not in self.char2idx:
                idx = start_idx + i
                self.char2idx[ch] = idx
                self.idx2char[idx] = ch

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encodes input string into a list of integer token IDs.
        
        Args:
            text (str): Text string to tokenize.
            add_special_tokens (bool): If True, prepends <bos> and appends <eos>.
            
        Returns:
            List[int]: List of token IDs.
        """
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_id)

        for ch in text:
            # Map character to token ID or unk_id if unseen
            token_id = self.char2idx.get(ch, self.unk_id)
            tokens.append(token_id)

        if add_special_tokens:
            tokens.append(self.eos_id)

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decodes a list of token IDs back into a human-readable text string.
        
        Args:
            token_ids (List[int]): List of integer token IDs.
            skip_special_tokens (bool): If True, removes special tokens (<pad>, <bos>, <eos>).
            
        Returns:
            str: Decoded text string.
        """
        chars = []
        special_ids = {self.pad_id, self.bos_id, self.eos_id}
        
        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            
            # Retrieve character or <unk> representation
            char = self.idx2char.get(tid, self.UNK_TOKEN)
            chars.append(char)

        return "".join(chars)

    def save(self, filepath: str):
        """
        Saves the tokenizer vocabulary and mappings to a local JSON file.
        
        Args:
            filepath (str): Target file path (e.g. 'checkpoints/tokenizer.json').
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "special_tokens": self.special_tokens,
            "char2idx": self.char2idx,
            "idx2char": {str(k): v for k, v in self.idx2char.items()},
            "vocab_size": self.vocab_size
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[CrachoTokenizer] Tokenizer saved successfully to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "CrachoTokenizer":
        """
        Loads a tokenizer from a saved local JSON file.
        
        Args:
            filepath (str): Path to saved tokenizer JSON file.
            
        Returns:
            CrachoTokenizer: Restored tokenizer instance.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        instance = cls(special_tokens=data.get("special_tokens"))
        instance.char2idx = data["char2idx"]
        instance.idx2char = {int(k): v for k, v in data["idx2char"].items()}
        print(f"[CrachoTokenizer] Tokenizer loaded successfully from: {filepath} (Vocab Size: {instance.vocab_size})")
        return instance


if __name__ == "__main__":
    # Simple self-test demo
    sample_text = "Hello, CrachoLM! Building a decoder-only language model from scratch in PyTorch."
    tokenizer = CrachoTokenizer()
    tokenizer.train_from_text(sample_text)

    print(f"Vocab Size: {tokenizer.vocab_size}")
    encoded = tokenizer.encode(sample_text, add_special_tokens=True)
    print(f"Encoded Token IDs ({len(encoded)} tokens): {encoded[:15]}...")
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    print(f"Decoded Text: '{decoded}'")
    assert decoded == sample_text, "Decoding failed to match original text!"
    print("Self-test PASSED!")
