import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.checkpoint import load_checkpoint_file

ckpt_dir = "checkpoints_general_v2"
out_lines = []

for ep in range(1, 6):
    path = os.path.join(ckpt_dir, f"checkpoint_epoch_{ep}.pt")
    if os.path.exists(path):
        state = load_checkpoint_file(path, torch.device("cpu"))
        out_lines.append(f"Epoch {ep}: epoch={state.get('epoch')}, global_step={state.get('global_step')}, tokens_processed={state.get('tokens_processed')}, val_loss={state.get('val_loss'):.4f}, best_val_loss={state.get('best_val_loss'):.4f}")

with open("scratch/loss_summary.txt", "w") as f:
    f.write("\n".join(out_lines))
