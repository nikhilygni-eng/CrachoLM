"""
CrachoLM Training Pipeline Diagnostic Test
==========================================
Runs a short 1-epoch test to verify model forward pass, backward pass,
gradient step, loss logging, checkpoint saving, and loading logic.
"""

import os
import sys
import torch

# Add project root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import Config, ModelConfig, TrainingConfig
from src.device import get_device
from src.tokenizer import CrachoTokenizer
from src.model import CrachoLM
from src.data import prepare_dataloaders
from src.trainer import CrachoTrainer
from src.checkpoint import load_checkpoint_file


def test_training_pipeline():
    print("=" * 68)
    print("            CrachoLM Phase 5: Training Pipeline Test Suite")
    print("=" * 68)

    device = get_device()
    print(f"[1] Target Device: {device}")

    # Small test dataset
    text_data = "CrachoLM training verification pipeline. " * 30
    tokenizer = CrachoTokenizer()
    tokenizer.train_from_text(text_data)

    # Small test config
    config = Config(
        model=ModelConfig(vocab_size=tokenizer.vocab_size, d_model=128, n_layers=2, n_heads=4, max_seq_len=32),
        training=TrainingConfig(batch_size=2, max_epochs=1, log_interval=1)
    )

    train_loader, val_loader, stats = prepare_dataloaders(
        text_data=text_data,
        tokenizer=tokenizer,
        seq_len=config.model.max_seq_len,
        batch_size=config.training.batch_size
    )

    # Initialize model
    model = CrachoLM(config.model)

    # Initialize trainer
    trainer = CrachoTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device
    )

    # Run 1 epoch
    print("[2] Executing 1 Test Epoch...")
    trainer.train()

    # Verify best_model checkpoint creation
    best_path = os.path.join(config.system.checkpoints_dir, "best_model.pt")
    assert os.path.exists(best_path), f"Error: Checkpoint file not created at {best_path}"
    print(f"    [✓] Checkpoint file verified at: {best_path}")

    # Verify checkpoint stores config as a plain dict (no Config Python object)
    print("[3] Verifying checkpoint format (config must be a dict, not a Config object)...")
    ckpt_state = load_checkpoint_file(best_path, device)
    assert "config_dict" in ckpt_state, "Error: Checkpoint missing 'config_dict' key!"
    assert isinstance(ckpt_state["config_dict"], dict), (
        f"Error: 'config_dict' must be a plain dict, got {type(ckpt_state['config_dict'])}"
    )
    assert "model" in ckpt_state["config_dict"], "Error: config_dict missing 'model' sub-dict!"
    assert "config" not in ckpt_state, (
        "Error: Old 'config' key still present — Config object must not be stored directly!"
    )
    ver = ckpt_state.get("cracho_ckpt_version", 0)
    assert ver == 2, f"Error: Expected checkpoint version 2, got {ver}"
    print(f"    [✓] Checkpoint format v{ver} verified (config stored as plain dict)!")

    # Test resuming from checkpoint
    print("[4] Testing Resume Training from Checkpoint...")
    resumed_trainer = CrachoTrainer(
        model=CrachoLM(config.model),
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        resume_checkpoint_path=best_path
    )
    assert resumed_trainer.start_epoch >= 1, "Error: Resumed epoch count mismatch!"
    print("    [✓] Resume Training Verified Successfully!")

    print("=" * 68)
    print(" Training Pipeline Test Suite Completed PASSED (100% SUCCESS)!")
    print("=" * 68)


if __name__ == "__main__":
    test_training_pipeline()
