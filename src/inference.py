"""Shared checkpoint/tokenizer loading for CrachoLM inference."""
import json
import hashlib
from pathlib import Path
import torch
from src.model import CrachoLM
from src.tokenizer import CrachoTokenizer
from src.bpe_tokenizer import CrachoBPETokenizer
from src.checkpoint import load_checkpoint_file, get_model_config_from_checkpoint

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints_capability_mixed_20260928" / "best_model.pt"


def project_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_model_for_inference(checkpoint_path=None, tokenizer_path=None, device=None):
    """Load an explicit checkpoint, paired tokenizer, and truthful model metadata."""
    device = torch.device("cpu") if device is None else torch.device(device)
    checkpoint_path = project_path(checkpoint_path or DEFAULT_CHECKPOINT)
    if tokenizer_path is None:
        candidates = [checkpoint_path.parent / name
                      for name in ("tokenizer_bpe.json", "tokenizer.json")]
        tokenizer_path = next((p for p in candidates if p.is_file()), None)
        if tokenizer_path is None:
            raise FileNotFoundError(
                f"No tokenizer beside {checkpoint_path}. Pass --tokenizer explicitly."
            )
    tokenizer_path = project_path(tokenizer_path)
    with tokenizer_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    is_bpe = data.get("tokenizer_type") == "BPE" or "token2idx" in data
    tokenizer = (CrachoBPETokenizer if is_bpe else CrachoTokenizer).load(str(tokenizer_path))
    # Keep optimizer tensors on CPU; only the inference model belongs on the GPU.
    checkpoint = load_checkpoint_file(str(checkpoint_path), torch.device("cpu"))
    expected_fingerprint = checkpoint.get("tokenizer_sha256")
    if expected_fingerprint and hashlib.sha256(tokenizer_path.read_bytes()).hexdigest() != expected_fingerprint:
        raise ValueError("Tokenizer fingerprint differs from the checkpoint. Use its original paired tokenizer.")
    model_config = get_model_config_from_checkpoint(checkpoint)
    if tokenizer.vocab_size != model_config.vocab_size:
        raise ValueError(
            f"Tokenizer has {tokenizer.vocab_size} tokens but checkpoint expects "
            f"{model_config.vocab_size}. Use its training tokenizer."
        )
    mapping = tokenizer.token2idx if is_bpe else tokenizer.char2idx
    if set(mapping.values()) != set(range(tokenizer.vocab_size)):
        raise ValueError("Tokenizer IDs must be unique and contiguous.")
    model = CrachoLM(model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()
    info = {
        "model_name": "CrachoLM-General-v3" if checkpoint_path.parent.name == "checkpoints_general_v3"
                      else checkpoint_path.parent.name,
        "parameters": model.get_num_params(),
        "parameters_m": round(model.get_num_params() / 1e6, 2),
        "vocab_size": tokenizer.vocab_size,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "val_loss": checkpoint.get("val_loss"),
        "tokens_processed": checkpoint.get("tokens_processed"),
        "checkpoint": str(checkpoint_path),
        "tokenizer": str(tokenizer_path),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
    }
    return model, tokenizer, info
