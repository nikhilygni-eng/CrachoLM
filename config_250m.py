"""Separate ~250M preset. Does not change the original default_config."""
from config import Config, ModelConfig, TrainingConfig


def make_250m_config(vocab_size=2048):
    if vocab_size != 2048:
        raise ValueError("This 250M preset is paired with the existing 2048-token BPE.")
    return Config(
        model=ModelConfig(
            vocab_size=vocab_size, d_model=1024, n_layers=20, n_heads=16,
            d_ff=4000, max_seq_len=512, dropout=0.0, bias=False,
            use_sdpa=True, gradient_checkpointing=True,
        ),
        training=TrainingConfig(
            batch_size=1, grad_accum_steps=16, learning_rate=3e-4,
            min_lr=3e-5, weight_decay=0.1, max_epochs=1,
            warmup_steps=100, log_interval=16,
        ),
    )
