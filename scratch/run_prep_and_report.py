import os
import sys
import json
import math
import shutil

# Ensure virtualenv site-packages are loaded
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bpe_tokenizer import CrachoBPETokenizer

OUTPUT_DIR = "data_general/raw_scaled"
TOKENIZER_PATH = "checkpoints_general_v2/tokenizer_bpe.json"

def audit():
    print("====================================================================")
    print(" 🔬 Scaled Dataset Tokenizer Audit & Metric Report")
    print("====================================================================")

    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")

    tokenizer = CrachoBPETokenizer.load(TOKENIZER_PATH)
    print(f"[✓] Tokenizer Loaded: {TOKENIZER_PATH} (vocab_size={tokenizer.vocab_size}, eos_id={tokenizer.eos_id})")

    files = [
        "01_general_english.txt",
        "02_educational_science.txt",
        "03_conversations_qa.txt",
        "04_programming_tech.txt"
    ]

    total_tokens = 0
    domain_stats = {}
    eos_counts = {}

    for f in files:
        fpath = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(fpath):
            with open(fpath, "r", encoding="utf-8") as fp:
                txt = fp.read()
            tokens = tokenizer.encode(txt, add_special_tokens=False)
            count = len(tokens)
            eos_c = tokens.count(tokenizer.eos_id)
            total_tokens += count
            domain_stats[f] = {
                "chars": len(txt),
                "bytes": os.path.getsize(fpath),
                "tokens": count,
                "eos_count": eos_c
            }
        else:
            print(f"[!] Warning: File {f} not found in {OUTPUT_DIR}")

    print("\nPer-Domain Token Statistics:")
    for fname, stats in domain_stats.items():
        pct = (stats["tokens"] / total_tokens * 100) if total_tokens > 0 else 0
        print(f"  - {fname:30s}: {stats['tokens']:,} tokens ({pct:.2f}% of corpus, {stats['eos_count']:,} <eos> tags)")

    print("-" * 68)
    print(f" TOTAL CORPUS TOKEN COUNT: {total_tokens:,} tokens")
    print("=" * 68)

    # Dataloader & Step Calculations
    seq_len = 256
    batch_size = 4
    grad_accum_steps = 16

    train_tokens = int(total_tokens * 0.90)
    val_tokens = total_tokens - train_tokens

    train_seqs = (train_tokens - 1) // seq_len
    val_seqs = (val_tokens - 1) // seq_len

    len_train_loader = math.ceil(train_seqs / batch_size)
    optim_updates_per_epoch = math.ceil(len_train_loader / grad_accum_steps)
    epochs = 2
    total_scheduler_steps = optim_updates_per_epoch * epochs
    recommended_warmup = max(5, int(total_scheduler_steps * 0.05))

    # Checkpoint File Size & Disk Space Verification
    sample_ckpt = "checkpoints_general_v2/checkpoint_epoch_1.pt"
    if not os.path.exists(sample_ckpt):
        sample_ckpt = "checkpoints_general_v2/best_model.pt"

    if os.path.exists(sample_ckpt):
        actual_ckpt_size_bytes = os.path.getsize(sample_ckpt)
        actual_ckpt_size_mb = actual_ckpt_size_bytes / (1024 * 1024)
    else:
        actual_ckpt_size_mb = 818.0

    total_disk, used_disk, free_disk = shutil.disk_usage(".")
    free_gb = free_disk / (1024 ** 3)
    est_3_ckpts_mb = actual_ckpt_size_mb * 3
    est_3_ckpts_gb = est_3_ckpts_mb / 1024

    report = {
        "total_tokens": total_tokens,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "domain_stats": domain_stats,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "train_seqs": train_seqs,
        "val_seqs": val_seqs,
        "len_train_loader": len_train_loader,
        "optim_updates_per_epoch": optim_updates_per_epoch,
        "epochs": epochs,
        "total_scheduler_steps": total_scheduler_steps,
        "recommended_warmup": recommended_warmup,
        "actual_ckpt_size_mb": round(actual_ckpt_size_mb, 2),
        "est_3_ckpts_mb": round(est_3_ckpts_mb, 2),
        "est_3_ckpts_gb": round(est_3_ckpts_gb, 2),
        "free_disk_gb": round(free_gb, 2)
    }

    with open("scratch/scaled_audit_results.json", "w") as f:
        json.dump(report, f, indent=2)

    return report

if __name__ == "__main__":
    audit()
