"""
CrachoLM Model Architecture Module
==================================
Decoder-Only Transformer Language Model built from scratch using PyTorch.

Components:
1. Token & Positional Embeddings
2. Multi-Head Causal Self-Attention (with lower-triangular causal mask)
3. Feed-Forward Neural Networks (MLP with GELU activation)
4. Pre-Layer Normalization & Residual Connections
5. Decoder Blocks Stack
6. Next-Token Prediction Language Modeling Head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# Import configuration dataclass
from config import ModelConfig


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention Layer built from scratch.
    Ensures autoregressive property via lower-triangular causal masking (prevents seeing future tokens).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
        
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        
        # Combined Linear layer for Query, Key, and Value projections
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        # Output Projection layer
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal Mask (lower-triangular matrix filled with 1s)
        # Registered as buffer so it stays on the correct device and isn't updated as a parameter
        causal_mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(
            1, 1, config.max_seq_len, config.max_seq_len
        )
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # Batch size, Sequence length, Embedding dimension (d_model)

        # 1. Linear Projection -> Query, Key, Value vectors
        # Shape: (B, T, 3 * C) -> split into 3 tensors of shape (B, T, C)
        q, k, v = self.c_attn(x).chunk(3, dim=-1)

        # 2. Reshape for Multi-Head Attention: (B, T, C) -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 3. Scaled Dot-Product Attention: (Q * K^T) / sqrt(head_dim)
        # Shape: (B, n_heads, T, head_dim) @ (B, n_heads, head_dim, T) -> (B, n_heads, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # 4. Apply Causal Masking: fill future positions (where mask == 0) with -infinity
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))

        # 5. Softmax over sequence dimension & Apply Dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 6. Weight Value vectors: (B, n_heads, T, T) @ (B, n_heads, T, head_dim) -> (B, n_heads, T, head_dim)
        y = att @ v

        # 7. Concatenate all attention heads back: (B, n_heads, T, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 8. Output Linear Projection & Residual Dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


class FeedForward(nn.Module):
    """
    Position-Wise Feed-Forward Network (MLP).
    Expands model dimension to d_ff (typically 4 * d_model) with non-linear activation (GELU).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single Decoder Transformer Layer (Pre-LayerNorm architecture).
    Combines Causal Self-Attention and Feed-Forward Neural Network with Residual Connections.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LayerNorm Residual Connection for Self-Attention
        x = x + self.attn(self.ln_1(x))
        # Pre-LayerNorm Residual Connection for Feed-Forward Network
        x = x + self.mlp(self.ln_2(x))
        return x


class CrachoLM(nn.Module):
    """
    CrachoLM-0.1 Master Decoder-Only Transformer Language Model.
    
    Architecture Overview:
    - Input Token IDs -> Token Embedding Matrix + Positional Embedding Matrix
    - Stack of N Transformer Decoder Blocks (Pre-LN Causal Self-Attention + MLP)
    - Final Layer Normalization
    - Language Modeling Head (Linear Projection to Vocabulary Logits)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token & Positional Embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # Stack of Transformer Decoder Blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        # Final Layer Normalization
        self.ln_f = nn.LayerNorm(config.d_model)

        # Output LM Head (projects hidden state d_model -> vocab_size logits)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight Tying (shares parameters between Token Embedding and LM Head)
        self.tok_emb.weight = self.lm_head.weight

        # Initialize all model weights from scratch
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """
        Initializes linear weights with Gaussian N(0, 0.02) and biases to 0.
        Employs standard GPT initialization.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def get_num_params(self) -> int:
        """
        Calculates and returns total number of trainable model parameters.
        Excludes tied positional embedding parameters to avoid double-counting.
        """
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return n_params

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of CrachoLM.
        
        Args:
            idx (torch.Tensor): Tensor of token IDs, shape (batch_size, seq_len).
            targets (Optional[torch.Tensor]): Target token IDs for computing cross-entropy loss.
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Logits of shape (batch_size, seq_len, vocab_size)
                - Cross-entropy loss scalar tensor (if targets provided, else None)
        """
        device = idx.device
        b, t = idx.size()
        
        assert t <= self.config.max_seq_len, (
            f"Cannot process sequence of length {t}, model max context length is {self.config.max_seq_len}"
        )

        # Generate positional indices [0, 1, 2, ..., t-1]
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # 1. Compute Embeddings
        tok_embeddings = self.tok_emb(idx)      # Shape: (b, t, d_model)
        pos_embeddings = self.pos_emb(pos)      # Shape: (t, d_model)
        x = self.drop(tok_embeddings + pos_embeddings)

        # 2. Forward through N Transformer Blocks
        for block in self.blocks:
            x = block(x)

        # 3. Final Layer Norm
        x = self.ln_f(x)

        # 4. Compute Logits
        if targets is not None:
            # Full forward pass computing logits & loss
            logits = self.lm_head(x)  # Shape: (b, t, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
        else:
            # Inference optimization: compute logits only for last token
            logits = self.lm_head(x[:, -1:, :])  # Shape: (b, 1, vocab_size)
            loss = None

        return logits, loss


if __name__ == "__main__":
    # Self-test block
    config = ModelConfig(vocab_size=100, d_model=256, n_layers=6, n_heads=8, max_seq_len=512)
    model = CrachoLM(config)
    print(f"CrachoLM Model initialized successfully!")
    print(f"Total Trainable Parameters: {model.get_num_params():,}")
    
    # Dummy input forward pass
    dummy_input = torch.randint(0, 100, (2, 32)) # batch_size=2, seq_len=32
    logits, _ = model(dummy_input)
    print(f"Dummy Input Shape : {dummy_input.shape}")
    print(f"Logits Output Shape: {logits.shape}")
