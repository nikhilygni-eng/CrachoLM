import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import default_config
from src.model import CrachoLM

default_config.model.vocab_size = 69
model = CrachoLM(default_config.model)
params = model.get_num_params()
print(f"Total Parameters: {params:,} ({params/1e6:.2f}M)")
