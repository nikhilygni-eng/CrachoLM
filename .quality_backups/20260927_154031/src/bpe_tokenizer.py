"""
CrachoLM-General-v2 Subword BPE Tokenizer Module
=================================================
A Byte-Pair Encoding (BPE) subword tokenizer for general language modeling.
Includes special token management (<pad>, <unk>, <bos>, <eos>), BPE merge rules,
subword vocabulary construction, and local JSON persistence.
"""

import json
import os
import sys
import re
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Set, Union

# Ensure project root is in sys.path when module is executed directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CrachoBPETokenizer:
    """
    Subword Byte-Pair Encoding (BPE) Tokenizer for CrachoLM-General-v2.
    
    Special Tokens:
        <pad>: Padding token (ID 0)
        <unk>: Unknown subword token (ID 1)
        <bos>: Beginning of sequence token (ID 2)
        <eos>: End of sequence token (ID 3)
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"

    def __init__(
        self,
        special_tokens: Optional[List[str]] = None,
        target_vocab_size: int = 4096
    ):
        if special_tokens is None:
            self.special_tokens = [
                self.PAD_TOKEN,
                self.UNK_TOKEN,
                self.BOS_TOKEN,
                self.EOS_TOKEN,
            ]
        else:
            self.special_tokens = special_tokens

        self.target_vocab_size = target_vocab_size

        # Mappings
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []  # Ranked list of merge rules (pair_a, pair_b)

        # Initialize special tokens in vocabulary
        self._init_special_tokens()

    def _init_special_tokens(self):
        """Initializes special tokens at fixed leading indices 0..3."""
        for idx, token in enumerate(self.special_tokens):
            self.token2idx[token] = idx
            self.idx2token[idx] = token

    @property
    def pad_id(self) -> int:
        return self.token2idx[self.PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token2idx[self.UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token2idx[self.BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token2idx[self.EOS_TOKEN]

    @property
    def vocab_size(self) -> int:
        return len(self.token2idx)

    def _get_stats(self, vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, str], int]:
        """Counts frequency of adjacent symbol pairs in word vocabulary."""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair: Tuple[str, str], vocab: Dict[Tuple[str, ...], int]) -> Dict[Tuple[str, ...], int]:
        """Replaces all occurrences of pair in vocabulary with merged symbol."""
        v_out = {}
        bigram = pair
        replacement = "".join(pair)

        for word, freq in vocab.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == bigram[0] and word[i + 1] == bigram[1]:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            v_out[tuple(new_word)] = freq
        return v_out

    def train_from_text(self, text: str, target_vocab_size: Optional[int] = None):
        """
        Trains BPE subword merges and builds vocabulary from raw text corpus.
        
        Args:
            text (str): Training raw text.
            target_vocab_size (int, optional): Desired vocabulary size.
        """
        if target_vocab_size is not None:
            self.target_vocab_size = target_vocab_size

        print(f"[*] Training BPE Tokenizer (Target Vocab Size: {self.target_vocab_size})...")

        # 1. Pre-tokenize text into words using basic whitespace/punctuation regex
        words = re.findall(r"\w+|\S", text)
        
        # Represent words as tuple of characters ending with word boundary symbol '</w>'
        word_freqs = Counter(words)
        vocab = {}
        for word, freq in word_freqs.items():
            # Add character list + word boundary marker
            symbols = tuple(list(word) + ["</w>"])
            vocab[symbols] = freq

        # 2. Add base characters into vocabulary
        unique_chars = set()
        for symbols in vocab.keys():
            for sym in symbols:
                unique_chars.add(sym)

        start_idx = len(self.special_tokens)
        for i, char_sym in enumerate(sorted(list(unique_chars))):
            if char_sym not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[char_sym] = idx
                self.idx2token[idx] = char_sym

        # 3. Iterative BPE pair merging
        num_merges = self.target_vocab_size - len(self.token2idx)
        print(f"[*] Initial Base Vocab Size: {len(self.token2idx)} | Running up to {max(0, num_merges)} merges...")

        self.merges = []
        for i in range(max(0, num_merges)):
            pairs = self._get_stats(vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < 1:
                break

            vocab = self._merge_vocab(best_pair, vocab)
            self.merges.append(best_pair)

            new_token = "".join(best_pair)
            if new_token not in self.token2idx:
                idx = len(self.token2idx)
                self.token2idx[new_token] = idx
                self.idx2token[idx] = new_token

        print(f"[✓] BPE Training complete! Final Vocab Size: {self.vocab_size} ({len(self.merges)} merge rules)")

    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenizes a single word using learned BPE merge rules."""
        symbols = list(word) + ["</w>"]
        if len(symbols) == 1:
            return symbols

        for pair in self.merges:
            bigram = pair
            i = 0
            new_symbols = []
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == bigram[0] and symbols[i + 1] == bigram[1]:
                    new_symbols.append("".join(pair))
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
            if len(symbols) == 1:
                break
        return symbols

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encodes text string into a list of integer subword token IDs.
        
        Args:
            text (str): Input text string.
            add_special_tokens (bool): If True, prepends <bos> and appends <eos>.
            
        Returns:
            List[int]: Integer token IDs.
        """
        tokens = []
        if add_special_tokens:
            tokens.append(self.bos_id)

        # Preserve special tokens (<pad>, <unk>, <bos>, <eos>) if explicitly present in text
        special_pattern = r"(<pad>|<unk>|<bos>|<eos>)"
        parts = re.split(special_pattern, text)
        for part in parts:
            if not part:
                continue
            if part in self.token2idx:
                tokens.append(self.token2idx[part])
            else:
                words = re.findall(r"\w+|\S", part)
                for w in words:
                    subwords = self._tokenize_word(w)
                    for sub in subwords:
                        token_id = self.token2idx.get(sub, self.unk_id)
                        tokens.append(token_id)

        if add_special_tokens:
            tokens.append(self.eos_id)

        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decodes a list of subword token IDs back into a text string.
        
        Args:
            token_ids (List[int]): Token ID sequence.
            skip_special_tokens (bool): If True, removes special tokens (<pad>, <bos>, <eos>).
            
        Returns:
            str: Decoded text string.
        """
        subwords = []
        special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}

        for tid in token_ids:
            if skip_special_tokens and tid in special_ids:
                continue
            tok = self.idx2token.get(tid, self.UNK_TOKEN)
            subwords.append(tok)

        raw_text = "".join(subwords)
        # Restore space boundaries represented by </w>
        cleaned_text = raw_text.replace("</w>", " ")
        return cleaned_text.strip()

    def save(self, filepath: str):
        """Saves tokenizer state dict to local JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = {
            "special_tokens": self.special_tokens,
            "target_vocab_size": self.target_vocab_size,
            "vocab_size": self.vocab_size,
            "token2idx": self.token2idx,
            "idx2token": {str(k): v for k, v in self.idx2token.items()},
            "merges": self.merges,
            "tokenizer_type": "BPE"
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[CrachoBPETokenizer] Saved BPE tokenizer to: {filepath} (Vocab Size: {self.vocab_size})")

    @classmethod
    def load(cls, filepath: str) -> "CrachoBPETokenizer":
        """Loads BPE tokenizer from local JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        instance = cls(
            special_tokens=data.get("special_tokens"),
            target_vocab_size=data.get("target_vocab_size", 4096)
        )
        instance.token2idx = data["token2idx"]
        instance.idx2token = {int(k): v for k, v in data["idx2token"].items()}
        instance.merges = [tuple(m) for m in data.get("merges", [])]
        print(f"[CrachoBPETokenizer] Loaded BPE tokenizer from: {filepath} (Vocab Size: {instance.vocab_size})")
        return instance


if __name__ == "__main__":
    sample_text = (
        "CrachoLM-General-v2 is an advanced language model built from scratch. "
        "It supports multi-domain learning including programming, science, conversation, and general knowledge. "
        "Functions: def calculate_sum(a, b): return a + b."
    )
    tok = CrachoBPETokenizer(target_vocab_size=120)
    tok.train_from_text(sample_text)

    enc = tok.encode(sample_text, add_special_tokens=True)
    dec = tok.decode(enc, skip_special_tokens=True)

    print(f"\nSelf-Test Verification:")
    print(f" Special Token IDs -> PAD:{tok.pad_id}, UNK:{tok.unk_id}, BOS:{tok.bos_id}, EOS:{tok.eos_id}")
    print(f" Vocab Size       -> {tok.vocab_size}")
    print(f" Encoded IDs ({len(enc)}) -> {enc[:10]}...")
    print(f" Decoded Text     -> '{dec}'")

    test_path = "/tmp/test_bpe_tok.json"
    tok.save(test_path)
    restored = CrachoBPETokenizer.load(test_path)
    assert restored.vocab_size == tok.vocab_size, "Saved/Loaded vocab size mismatch!"
    print("[✓] Self-Test PASSED!")
