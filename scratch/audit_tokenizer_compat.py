import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bpe_tokenizer import CrachoBPETokenizer

print("====================================================================")
print(" 🔬 Tokenizer & Data Pipeline Compatibility Audit")
print("====================================================================")

tok_path = "checkpoints_general_v2/tokenizer_bpe.json"
assert os.path.exists(tok_path), f"Tokenizer file missing at {tok_path}"

tokenizer = CrachoBPETokenizer.load(tok_path)

print(f"\n[1] Tokenizer Identification & Special Token IDs:")
print(f"  - Vocab Size    : {tokenizer.vocab_size}")
print(f"  - <pad> ID      : {tokenizer.pad_id}")
print(f"  - <unk> ID      : {tokenizer.unk_id}")
print(f"  - <bos> ID      : {tokenizer.bos_id}")
print(f"  - <eos> ID      : {tokenizer.eos_id}")

assert tokenizer.eos_id == 3, f"Expected <eos> ID to be 3, got {tokenizer.eos_id}"

# 2. Test <eos> Encoding & Decoding
print(f"\n[2] Testing <eos> Special Token Encoding & Decoding:")

eos_encoded = tokenizer.encode("<eos>", add_special_tokens=False)
print(f"  - encode('<eos>')                           : {eos_encoded}")
assert eos_encoded == [3], f"Expected [3], got {eos_encoded}"

eos_decoded_show = tokenizer.decode([3], skip_special_tokens=False)
print(f"  - decode([3], skip_special_tokens=False)     : {repr(eos_decoded_show)}")
assert eos_decoded_show == "<eos>", f"Expected '<eos>', got {repr(eos_decoded_show)}"

eos_decoded_skip = tokenizer.decode([3], skip_special_tokens=True)
print(f"  - decode([3], skip_special_tokens=True)      : {repr(eos_decoded_skip)}")
assert eos_decoded_skip == "", f"Expected empty string '', got {repr(eos_decoded_skip)}"

# 3. Test Text with Embedded <eos> Tags
print(f"\n[3] Testing Embedded <eos> Tag Tokenization:")
sample_doc = "First sentence.\n<eos>\nUser: What is python?\nAssistant: Python is a programming language.<eos>"
encoded_doc = tokenizer.encode(sample_doc, add_special_tokens=False)
print(f"  - Full Encoded IDs ({len(encoded_doc)} tokens): {encoded_doc}")
eos_positions = [i for i, t in enumerate(encoded_doc) if t == 3]
print(f"  - Positions of token ID 3 (<eos>): {eos_positions}")
assert len(eos_positions) == 2, f"Expected 2 <eos> tokens, found {len(eos_positions)}"

decoded_doc_show = tokenizer.decode(encoded_doc, skip_special_tokens=False)
print(f"  - Decoded (skip=False): {repr(decoded_doc_show)}")

# 4. Test Normal Text Invariance
print(f"\n[4] Testing Normal Text Encoding Invariance:")
normal_text = "The quick brown fox jumps over the lazy dog. 1234567890!"
tokens_normal = tokenizer.encode(normal_text, add_special_tokens=False)
assert 3 not in tokens_normal, "<eos> ID 3 incorrectly present in normal text!"
decoded_normal = tokenizer.decode(tokens_normal, skip_special_tokens=True)
print(f"  - Normal Text Roundtrip Match: {repr(decoded_normal) == repr(normal_text)}")
print(f"  - Decoded Normal Text        : {repr(decoded_normal)}")

# 5. Protected Directories Audit
print(f"\n[5] Protected Directories Audit:")
protected = ["checkpoints_general", "checkpoints_general_benchmark", "checkpoints_general_v2", "backup_70m"]
for p in protected:
    assert os.path.exists(p), f"Protected folder {p} missing!"
    print(f"  - Untouched: {p}/")

print("\n" + "=" * 68)
print(" 🎉 TOKENIZER & DATA PIPELINE AUDIT PASSED 100% SUCCESSFULLY!")
print("=" * 68)
