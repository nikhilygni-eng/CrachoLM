import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.checkpoint import load_checkpoint_file

ckpt_dir = "checkpoints_general_v2"
print("====================================================================")
print(" 🔬 Inspecting Checkpoints in checkpoints_general_v2")
print("====================================================================")

for ep in range(1, 6):
    path = os.path.join(ckpt_dir, f"checkpoint_epoch_{ep}.pt")
    if os.path.exists(path):
        state = load_checkpoint_file(path, torch.device("cpu"))
        print(f"Epoch {ep}:")
        print(f"  Epoch           : {state.get('epoch')}")
        print(f"  Global Step     : {state.get('global_step')}")
        print(f"  Tokens Processed: {state.get('tokens_processed'):,}")
        print(f"  Val Loss        : {state.get('val_loss')}")
        print(f"  Best Val Loss   : {state.get('best_val_loss')}")
        print()

best_path = os.path.join(ckpt_dir, "best_model.pt")
if os.path.exists(best_path):
    state = load_checkpoint_file(best_path, torch.device("cpu"))
    print("Best Model Checkpoint:")
    print(f"  Epoch           : {state.get('epoch')}")
    print(f"  Global Step     : {state.get('global_step')}")
    print(f"  Tokens Processed: {state.get('tokens_processed'):,}")
    print(f"  Val Loss        : {state.get('val_loss')}")
    print(f"  Best Val Loss   : {state.get('best_val_loss')}")
