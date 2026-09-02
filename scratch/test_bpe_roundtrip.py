import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bpe_tokenizer import CrachoBPETokenizer

tok = CrachoBPETokenizer.load("checkpoints_general/tokenizer_bpe.json")

test_phrases = [
    "Question: What is artificial intelligence?",
    "def fibonacci(n):\n    return n",
    "First Citizen:\nCome on: we have some agai"
]

print("==========================================================")
print("     BPE Tokenizer Encode/Decode Reversibility Test")
print("==========================================================")

for p in test_phrases:
    ids = tok.encode(p, add_special_tokens=False)
    dec = tok.decode(ids, skip_special_tokens=True)
    print(f"Original : {repr(p)}")
    print(f"Token IDs: {ids[:15]}... ({len(ids)} tokens)")
    print(f"Decoded  : {repr(dec)}")
    print("-" * 50)
