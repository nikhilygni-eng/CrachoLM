"""
CrachoLM Configuration Module
===============================
Central configuration file for CrachoLM-0.1 architecture, training, and environment setup.
All hyperparameters and paths are defined here for easy experimentation.
"""

import os
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """
    Decoder-only Transformer Architecture Hyperparameters for CrachoLM-0.1.
    Starts from randomly initialized weights.
    """
    vocab_size: int = 32000       # Vocabulary size (will be updated after tokenizer training)
    d_model: int = 768            # Embedding & hidden dimension size (scaled to 70M)
    n_layers: int = 10            # Transformer decoder layers (scaled to 70M)
    n_heads: int = 12             # Multi-Head Attention heads (d_k = 768 / 12 = 64)
    d_ff: int = 3072              # Feed-Forward network hidden dim (4 * 768)
    max_seq_len: int = 256        # Reduced from 512 for 4x attention VRAM memory efficiency
    dropout: float = 0.1          # Dropout rate for regularization
    bias: bool = False            # Enable/disable linear layer bias (False is modern LLM convention)


@dataclass
class TrainingConfig:
    """
    Hyperparameters for model training from scratch.
    """
    batch_size: int = 4           # Micro-batch size per iteration (prevents VRAM OOM on 6GB GPU)
    grad_accum_steps: int = 16    # Gradient accumulation steps (Effective batch size = 4 * 16 = 64)
    learning_rate: float = 3e-4   # Peak learning rate for 70M model scale
    min_lr: float = 3e-5          # Minimum learning rate for cosine scheduler
    weight_decay: float = 0.01    # Weight decay for AdamW optimizer
    max_epochs: int = 20          # Default training epochs
    warmup_steps: int = 100       # Warmup steps for learning rate scheduler
    grad_clip: float = 1.0        # Gradient clipping norm thresh
    checkpoint_interval: int = 1  # Epoch interval to save model checkpoints
    log_interval: int = 10        # Iteration interval to log progress


@dataclass
class DataConfig:
    """
    File system paths for dataset storage and processing.
    """
    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    raw_data_dir: str = os.path.join(base_dir, "data", "raw")
    processed_data_dir: str = os.path.join(base_dir, "data", "processed")
    train_filename: str = "tinyshakespeare.txt"
    val_filename: str = "val.txt"


@dataclass
class SystemConfig:
    """
    Execution environment and compute parameters.
    """
    device: str = "auto"          # Compute device: 'auto', 'cuda', or 'cpu'
    seed: int = 42                # Random seed for reproducibility
    checkpoints_dir: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"))
    logs_dir: str = field(default_factory=lambda: os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
    num_workers: int = 2          # Data loader parallel process workers


@dataclass
class Config:
    """
    Master CrachoLM Project Configuration Container.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    system: SystemConfig = field(default_factory=SystemConfig)


# Global default configuration instance
default_config = Config()


if __name__ == "__main__":
    print("CrachoLM Configuration Initialized:")
    print(f" - Model d_model: {default_config.model.d_model}")
    print(f" - Model Layers : {default_config.model.n_layers}")
    print(f" - Micro Batch  : {default_config.training.batch_size}")
    print(f" - Grad Accum   : {default_config.training.grad_accum_steps} (Effective Batch Size: {default_config.training.batch_size * default_config.training.grad_accum_steps})")
