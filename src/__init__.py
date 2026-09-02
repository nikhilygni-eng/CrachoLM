"""
CrachoLM Source Package
"""

from src.device import get_device, get_device_info, print_device_info
from src.tokenizer import CrachoTokenizer
from src.model import CrachoLM, TransformerBlock, CausalSelfAttention, FeedForward
from src.data import clean_text, CrachoDataset, load_raw_text, prepare_dataloaders
from src.trainer import CrachoTrainer
from src.generator import generate_text
from src.checkpoint import (
    build_checkpoint,
    load_checkpoint_file,
    get_model_config_from_checkpoint,
    config_to_dict,
    model_config_from_dict,
)
from src.evaluator import (
    run_evaluation,
    compute_metrics,
    save_results_json,
    save_results_csv,
    verify_no_data_leakage,
)

__version__ = "0.1.0"
__all__ = [
    "get_device",
    "get_device_info",
    "print_device_info",
    "CrachoTokenizer",
    "CrachoLM",
    "TransformerBlock",
    "CausalSelfAttention",
    "FeedForward",
    "clean_text",
    "CrachoDataset",
    "load_raw_text",
    "prepare_dataloaders",
    "CrachoTrainer",
    "generate_text",
]
