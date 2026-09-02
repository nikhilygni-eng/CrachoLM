import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import default_config
from src.model import CrachoLM

# 1. Config constructed by train_general.py
cfg = default_config
cfg.model.d_model = 768
cfg.model.n_layers = 10
cfg.model.n_heads = 12
cfg.model.d_ff = 3072
cfg.model.max_seq_len = 256

print("==========================================================")
print("       CrachoLM-General-v2 Architecture Verification")
print("==========================================================")
print(f"d_model   : {cfg.model.d_model}")
print(f"n_layers  : {cfg.model.n_layers}")
print(f"n_heads   : {cfg.model.n_heads}")
print(f"d_ff      : {cfg.model.d_ff}")
print(f"max_seq_len: {cfg.model.max_seq_len}")

# Test 1: vocab_size = 256
cfg.model.vocab_size = 256
m_256 = CrachoLM(cfg.model)
p_256 = m_256.get_num_params()
print(f"\n1. Exact Params (vocab_size=256) : {p_256:,} ({p_256/1e6:.2f}M)")

# Test 2: vocab_size = 1024
cfg.model.vocab_size = 1024
m_1024 = CrachoLM(cfg.model)
p_1024 = m_1024.get_num_params()
print(f"2. Exact Params (vocab_size=1024): {p_1024:,} ({p_1024/1e6:.2f}M)")

# Weight Tying Check
is_tied = (m_1024.tok_emb.weight is m_1024.lm_head.weight)
print(f"3. Weight Tying Active            : {is_tied}")
print("==========================================================")
