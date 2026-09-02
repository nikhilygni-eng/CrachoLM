"""
CrachoLM Checkpoint Utilities
==============================
Centralised helpers for saving and loading checkpoint files.

Design decisions
----------------
* Config is saved as a plain Python dictionary (JSON-serializable primitives),
  NOT as a dataclass/Python object.  This means torch.save() never pickles a
  custom class, so torch.load(..., weights_only=True) works on all PyTorch
  versions ≥ 2.0.

* load_checkpoint_file() always uses weights_only=False because the checkpoint
  contains optimizer state dicts (not just tensors).  This is safe ONLY for
  checkpoints produced by this project on your own machine.  Never load an
  untrusted .pt file with weights_only=False.

* Backward-compatibility: if an old checkpoint containing a Config object is
  detected (key "config" holds a non-dict value), the loader extracts the
  needed fields from the object so old runs are not silently broken.
"""

import os
import torch
from dataclasses import asdict
from typing import Any, Dict


# ─────────────────────────────────────────────────────────────
# Config ↔ dict helpers
# ─────────────────────────────────────────────────────────────

def config_to_dict(config) -> Dict[str, Any]:
    """
    Converts a Config dataclass (or any nested dataclass) to a plain dict
    containing only JSON-serializable primitives (str, int, float, bool).
    """
    return asdict(config)          # dataclasses.asdict handles nested dataclasses


def model_config_from_dict(d: Dict[str, Any]):
    """Reconstructs a ModelConfig dataclass from a plain dict."""
    from config import ModelConfig
    return ModelConfig(**d)


def full_config_from_dict(d: Dict[str, Any]):
    """Reconstructs a full Config dataclass from a plain dict."""
    from config import Config, ModelConfig, TrainingConfig, DataConfig, SystemConfig
    return Config(
        model=ModelConfig(**d["model"]),
        training=TrainingConfig(**d["training"]),
        data=DataConfig(**d["data"]),
        system=SystemConfig(**d["system"]),
    )


# ─────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────

def build_checkpoint(
    model,
    optimizer,
    scheduler,
    config,
    epoch: int,
    global_step: int,
    tokens_processed: int,
    val_loss: float,
    best_val_loss: float,
) -> Dict[str, Any]:
    """
    Builds a checkpoint dict whose every value is either a tensor state-dict
    or a plain Python primitive / dict.  No custom Python class is stored.
    """
    return {
        # ── Numeric / primitive metadata ──────────────────────────────
        "epoch": epoch,
        "global_step": global_step,
        "tokens_processed": tokens_processed,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "cracho_ckpt_version": 2,          # version tag for future compat

        # ── Tensor state dicts ────────────────────────────────────────
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else {},

        # ── Config stored as plain dict (no custom class) ─────────────
        "config_dict": config_to_dict(config),
    }


# ─────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────

def load_checkpoint_file(filepath: str, device: torch.device) -> Dict[str, Any]:
    """
    Loads a .pt checkpoint file produced by CrachoLM.

    Safety note
    -----------
    weights_only=False is required because the checkpoint contains optimizer
    state dicts (Python objects, not pure tensors).  This is safe here because
    these files are always created locally by train.py on your own machine.
    Never use this function on an untrusted .pt file downloaded from the internet.

    Backward compatibility
    ----------------------
    If the file was saved with the old format (containing a Config object under
    the key "config"), the loader normalises it to the new format so that
    callers always see "config_dict".
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    # weights_only=False — required for optimizer state dicts; safe for local files only.
    state = torch.load(filepath, map_location=device, weights_only=False)

    # ── Backward-compat: old format stored Config object under "config" ──
    if "config_dict" not in state and "config" in state:
        cfg_obj = state["config"]
        if hasattr(cfg_obj, "__dataclass_fields__"):
            # It's a dataclass — convert it now
            state["config_dict"] = config_to_dict(cfg_obj)
        else:
            # Unknown shape; store as-is and let the caller handle it
            state["config_dict"] = cfg_obj
        # Remove the old key so callers don't use it by accident
        del state["config"]

    return state


def get_model_config_from_checkpoint(state: Dict[str, Any]):
    """
    Extracts and reconstructs a ModelConfig from a loaded checkpoint state dict.
    Works for both the new format (config_dict) and old format (config object).
    """
    cd = state["config_dict"]
    if isinstance(cd, dict):
        return model_config_from_dict(cd["model"])
    # Fallback: cd is already a Config-like object
    return cd.model
