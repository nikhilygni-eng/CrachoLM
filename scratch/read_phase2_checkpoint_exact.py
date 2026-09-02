import os
import json
import torch

ckpt_dir = "checkpoints_general_v2"
results = {}

for ep in range(1, 6):
    path = os.path.join(ckpt_dir, f"checkpoint_epoch_{ep}.pt")
    if os.path.exists(path):
        # Load state dict safely on CPU
        state = torch.load(path, map_location="cpu", weights_only=False)
        val_loss = state.get("val_loss")
        perplexity = float(torch.exp(torch.tensor(min(val_loss, 20))).item()) if val_loss is not None else None
        results[f"epoch_{ep}"] = {
            "epoch": state.get("epoch"),
            "global_step": state.get("global_step"),
            "tokens_processed": state.get("tokens_processed"),
            "val_loss": val_loss,
            "best_val_loss": state.get("best_val_loss"),
            "perplexity": perplexity
        }

with open("scratch/exact_ckpt_values.json", "w") as f:
    json.dump(results, f, indent=2)
