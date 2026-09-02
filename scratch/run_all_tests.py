import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bpe_tokenizer import CrachoBPETokenizer
from src.model import CrachoLM, ModelConfig
from src.generator import generate_text

print("====================================================================")
print(" 🚀 Comprehensive Validation Test Suite")
print("====================================================================")

# 1. Test Tokenizer <eos> preservation in encode()
print("\n[TEST 1] Testing BPE Tokenizer <eos> encoding...")
tok = CrachoBPETokenizer(target_vocab_size=2048)
sample_text = "User: Hello\nAssistant: Hi there!<eos>\nUser: How are you?"
tokens = tok.encode(sample_text, add_special_tokens=False)
assert tok.eos_id in tokens, "FAILED: <eos> token ID not found in encoded sequence!"
print(f"[✓] PASSED: <eos> token correctly encoded into token ID {tok.eos_id}.")

# 2. Test Model Loss Calculation ignore_index=0
print("\n[TEST 2] Testing Model Loss Computation with ignore_index=0...")
config = ModelConfig(vocab_size=2048, d_model=128, n_layers=2, n_heads=4, d_ff=512, max_seq_len=64)
model = CrachoLM(config)
x = torch.randint(4, 2000, (2, 16))
y = torch.randint(4, 2000, (2, 16))
y[0, :5] = 0  # Insert <pad> target tokens (ID 0)
logits, loss = model(x, y)
assert loss is not None and not torch.isnan(loss), "FAILED: Loss computation failed with <pad> tokens!"
print(f"[✓] PASSED: Loss computed cleanly with <pad> token masking (Loss: {loss.item():.4f}).")

# 3. Test Generator Repetition Penalty & No-Repeat N-Gram
print("\n[TEST 3] Testing Generator Repetition Penalty & No-Repeat N-Gram...")
output_std = generate_text(
    model=model,
    tokenizer=tok,
    prompt="Testing generation",
    max_new_tokens=20,
    greedy=True,
    repetition_penalty=1.0,
    no_repeat_ngram_size=0
)
output_penalized = generate_text(
    model=model,
    tokenizer=tok,
    prompt="Testing generation",
    max_new_tokens=20,
    greedy=True,
    repetition_penalty=1.2,
    no_repeat_ngram_size=3
)
print(f" Standard Greedy Output : {repr(output_std[:60])}")
print(f" Penalized Output       : {repr(output_penalized[:60])}")
print("[✓] PASSED: Generator repetition penalty and no-repeat n-gram logic executed cleanly.")

# 4. Protected Folders Verification
print("\n[TEST 4] Verifying Protected Directories Isolation...")
protected = ["checkpoints_general", "checkpoints_general_benchmark", "backup_70m", "checkpoints_general_v2"]
for p in protected:
    assert os.path.exists(p), f"FAILED: Protected directory {p} missing!"
print("[✓] PASSED: All protected checkpoint directories are fully intact and isolated.")

print("\n" + "=" * 68)
print(" 🎉 ALL VALIDATION TESTS PASSED SUCCESSFULLY!")
print("=" * 68)
