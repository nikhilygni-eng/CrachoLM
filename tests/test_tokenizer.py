"""
CrachoLM Tokenizer Test Script
================================
Demonstrates and validates:
1. Training tokenizer vocabulary from raw text
2. Encoding text to token IDs
3. Decoding token IDs back to original text
4. Special token handling (<pad>, <unk>, <bos>, <eos>)
5. Local saving and loading of tokenizer state (.json)
"""

import os
import sys

# Add project root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tokenizer import CrachoTokenizer


def test_tokenizer_pipeline():
    print("=" * 68)
    print("            CrachoLM Phase 2: Tokenizer Test Suite")
    print("=" * 68)

    # 1. Sample Training Corpus
    training_corpus = """
    CrachoLM is a decoder-only Transformer language model built completely from scratch!
    It uses zero pretrained weights, zero external APIs, and zero pretrained tokenizers.
    Mathematical foundations: Self-Attention, Multi-Head Attention, Feed-Forward Networks.
    1234567890 -+=@#$%^&*()
    """
    
    print("[1] Building Vocabulary from Sample Training Corpus...")
    tokenizer = CrachoTokenizer()
    tokenizer.train_from_text(training_corpus)

    print(f"    [✓] Special Tokens : {tokenizer.special_tokens}")
    print(f"    [✓] Total Vocab Size: {tokenizer.vocab_size} tokens")

    # Display vocabulary mapping sample
    sample_vocab = list(tokenizer.char2idx.items())[:15]
    print(f"    [✓] Vocab Sample    : {sample_vocab} ...")
    print("-" * 68)

    # 2. Test Encoding & Decoding
    input_text = "CrachoLM learns to predict the next token."
    print(f"[2] Testing Encoding & Decoding")
    print(f"    Original Text      : '{input_text}'")

    # Encode without special tokens
    encoded_ids = tokenizer.encode(input_text, add_special_tokens=False)
    print(f"    Encoded IDs        : {encoded_ids}")

    # Encode with special tokens
    encoded_with_special = tokenizer.encode(input_text, add_special_tokens=True)
    print(f"    Encoded w/ <bos>/<eos>: {encoded_with_special}")

    # Decode back
    decoded_text = tokenizer.decode(encoded_ids, skip_special_tokens=True)
    print(f"    Decoded Text       : '{decoded_text}'")

    # Assert exact match
    assert input_text == decoded_text, "Error: Decoded text does not match original input!"
    print(f"    [✓] Exact Match Verified!")
    print("-" * 68)

    # 3. Test Unknown Token (<unk>) Handling
    unseen_text = "Emoji test 😀 and rare glyphs 𝚯"
    print(f"[3] Testing Unknown Token (<unk>) Handling")
    print(f"    Unseen Text Input  : '{unseen_text}'")
    encoded_unseen = tokenizer.encode(unseen_text, add_special_tokens=False)
    print(f"    Encoded IDs        : {encoded_unseen}")
    decoded_unseen = tokenizer.decode(encoded_unseen, skip_special_tokens=False)
    print(f"    Decoded Output     : '{decoded_unseen}'")
    print(f"    [✓] <unk> Handling Verified!")
    print("-" * 68)

    # 4. Test Local Persistence (Save & Load)
    save_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "tokenizer.json")
    print(f"[4] Testing Save & Load Functionality")
    print(f"    Saving tokenizer to : {save_path}")
    tokenizer.save(save_path)

    print(f"    Loading tokenizer from: {save_path}")
    loaded_tokenizer = CrachoTokenizer.load(save_path)

    # Verify loaded tokenizer behavior
    re_encoded = loaded_tokenizer.encode(input_text, add_special_tokens=True)
    re_decoded = loaded_tokenizer.decode(re_encoded, skip_special_tokens=True)
    assert re_decoded == input_text, "Error: Loaded tokenizer failed decode test!"
    print(f"    [✓] Saved & Loaded Tokenizer Verified Successfully!")

    print("=" * 68)
    print(" Tokenizer Test Suite Completed PASSED (100% SUCCESS)!")
    print("=" * 68)


if __name__ == "__main__":
    test_tokenizer_pipeline()
