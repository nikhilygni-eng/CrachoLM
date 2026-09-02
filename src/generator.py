"""
CrachoLM Autoregressive Text Generation Module
===============================================
Implements token-by-token text generation with configurable sampling:
- Temperature scaling
- Top-K sampling
- Deterministic greedy decoding
- Context window truncation
- EOS token stopping
"""

import torch
import torch.nn.functional as F
from typing import Optional

from src.model import CrachoLM
from src.bpe_tokenizer import CrachoBPETokenizer


@torch.no_grad()
def generate_text(
    model: CrachoLM,
    tokenizer: CrachoBPETokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: Optional[int] = 40,
    greedy: bool = False,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    device: torch.device = torch.device("cpu")
) -> str:
    """
    Generates text autoregressively starting from a prompt string.
    
    Args:
        model (CrachoLM): Trained model instance.
        tokenizer (CrachoTokenizer): Tokenizer instance.
        prompt (str): Text prompt to initialize generation.
        max_new_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Temperature parameter for scaling logits. Higher = more creative/random.
        top_k (Optional[int]): Top-K filtering. Retains only top-K highest probability tokens.
        greedy (bool): If True, uses deterministic argmax decoding regardless of temperature.
        repetition_penalty (float): Penalty factor for previously generated tokens (> 1.0 reduces repetition).
        no_repeat_ngram_size (int): Size of n-grams that cannot be repeated (> 0 disables duplicate n-grams).
        device (torch.device): Compute device (CUDA or CPU).
        
    Returns:
        str: Completed generated text including prompt.
    """
    model.eval()
    
    # Fallback for empty prompt
    if not prompt or len(prompt.strip()) == 0:
        prompt = "CrachoLM"

    # Encode prompt into token IDs
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    idx = torch.tensor([token_ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Crop context sequence if it exceeds model max sequence length
        idx_cond = idx if idx.size(1) <= model.config.max_seq_len else idx[:, -model.config.max_seq_len:]

        # Forward pass to get logits
        logits, _ = model(idx_cond)
        
        # Pluck logits for the very last token in the sequence
        # Shape: (1, 1, vocab_size) -> (1, vocab_size)
        logits = logits[:, -1, :]

        # Mask special tokens (PAD, UNK, BOS) — never sample these during generation
        special_ids_to_mask = [tokenizer.pad_id, tokenizer.unk_id, tokenizer.bos_id]
        logits[:, special_ids_to_mask] = -float("Inf")

        # Apply Repetition Penalty if configured (> 1.0)
        if repetition_penalty > 1.0 and idx.size(1) > 0:
            seq = idx[0].tolist()
            unique_tokens = set(seq)
            for tok in unique_tokens:
                if tok < logits.size(-1):
                    if logits[0, tok] < 0:
                        logits[0, tok] *= repetition_penalty
                    else:
                        logits[0, tok] /= repetition_penalty

        # Apply No-Repeat N-Gram constraint if configured (> 0)
        if no_repeat_ngram_size > 0 and idx.size(1) >= no_repeat_ngram_size - 1:
            seq = idx[0].tolist()
            ngram_prefix = tuple(seq[-(no_repeat_ngram_size - 1):]) if no_repeat_ngram_size > 1 else ()
            banned_tokens = set()
            for i in range(len(seq) - no_repeat_ngram_size + 1):
                if no_repeat_ngram_size == 1:
                    banned_tokens.add(seq[i])
                else:
                    if tuple(seq[i : i + no_repeat_ngram_size - 1]) == ngram_prefix:
                        banned_tokens.add(seq[i + no_repeat_ngram_size - 1])
            for b_tok in banned_tokens:
                logits[0, b_tok] = -float("Inf")

        # 1. Greedy Decoding
        if greedy or temperature <= 1e-6:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # 2. Temperature Scaling
            logits = logits / temperature

            # 3. Top-K Sampling Filter
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # Mask out tokens with logits smaller than the k-th highest logit
                logits[logits < v[:, [-1]]] = -float("Inf")

            # 4. Softmax Probability Distribution
            probs = F.softmax(logits, dim=-1)

            # 5. Multinomial Sampling
            idx_next = torch.multinomial(probs, num_samples=1)

        # Stop early if End-of-Sequence (<eos>) token is generated
        if idx_next.item() == tokenizer.eos_id:
            break

        # Append generated token to sequence
        idx = torch.cat((idx, idx_next), dim=1)

    # Decode full token ID sequence back into text string
    generated_tokens = idx[0].tolist()
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return generated_text
