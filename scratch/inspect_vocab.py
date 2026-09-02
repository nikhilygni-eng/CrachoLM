import json
import os

vocab_file = "checkpoints_general_v2/tokenizer_bpe.json"
if os.path.exists(vocab_file):
    with open(vocab_file, "r") as f:
        data = json.load(f)

    vocab = data.get("vocab", {})
    merges = data.get("merges", [])
    print(f"Vocab size: {len(vocab)}")
    
    # Reverse mapping id -> token
    id2token = {v: k for k, v in vocab.items()}
    
    print("\nFirst 30 tokens in vocab:")
    for i in range(min(30, len(vocab))):
        print(f"  {i}: {repr(id2token.get(i))}")

    print("\nTokens matching 'bs' or containing 'b'/'s':")
    for tok_str, idx in vocab.items():
        if "bs" in tok_str or tok_str == "bs" or tok_str == " bs":
            print(f"  ID {idx}: {repr(tok_str)}")
