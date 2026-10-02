"""
CrachoLM Training Pipeline Module (Phase 7 Enhanced)
===================================================
Handles model optimization, mixed-precision training, gradient accumulation,
linear warmup with cosine LR decay, metrics tracking, checkpointing, and OOM protection.
"""

import time
import os
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional, Dict, Any

from config import Config
from src.device import get_device, get_device_info
from src.model import CrachoLM
from src.tokenizer import CrachoTokenizer
from src.checkpoint import build_checkpoint, load_checkpoint_file


def get_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    """
    Creates a Learning Rate Scheduler with Linear Warmup followed by Cosine Decay.
    """
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


class CrachoTrainer:
    """
    Manages the complete training loop for CrachoLM.
    """

    def __init__(
        self,
        model: CrachoLM,
        config: Config,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        tokenizer: CrachoTokenizer,
        device: torch.device,
        resume_checkpoint_path: Optional[str] = None
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.device = device
        self.grad_accum_steps = max(1, config.training.grad_accum_steps)
        
        # Training state variables
        self.start_epoch = 0
        self.global_step = 0
        self.tokens_processed = 0
        self.best_val_loss = float("inf")

        # 1. Optimizer Setup (AdamW with decoupled weight decay)
        self.optimizer = self._create_optimizer()

        # 2. Learning Rate Scheduler (Linear Warmup + Cosine Decay)
        optim_steps_per_epoch = math.ceil(len(train_loader) / self.grad_accum_steps)
        total_optim_steps = config.training.max_epochs * optim_steps_per_epoch
        min_ratio = config.training.min_lr / config.training.learning_rate
        self.scheduler = get_warmup_cosine_scheduler(
            self.optimizer,
            warmup_steps=config.training.warmup_steps,
            total_steps=max(1, total_optim_steps),
            min_lr_ratio=min_ratio
        )

        # 3. Automatic Mixed Precision (AMP) Scaler (CUDA only)
        self.use_amp = (device.type == "cuda")
        if self.use_amp:
            try:
                self.scaler = torch.amp.GradScaler('cuda')
            except Exception:
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # 4. Resume training if checkpoint path provided
        if resume_checkpoint_path:
            self.load_checkpoint(resume_checkpoint_path)

    def _create_optimizer(self) -> AdamW:
        """
        Configures AdamW optimizer with weight decay applied to 2D matrices (linear weights)
        and zero weight decay on 1D parameters (biases and LayerNorm).
        """
        decay_params = []
        nodecay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

        optim_groups = [
            {"params": decay_params, "weight_decay": self.config.training.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        return AdamW(
            optim_groups,
            lr=self.config.training.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8
        )

    def train_epoch(self, epoch: int) -> float:
        """
        Executes one full epoch of training with gradient accumulation.
        """
        self.model.train()
        total_train_loss = 0.0
        start_time = time.time()
        self.optimizer.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(self.train_loader):
            step_start_time = time.time()
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            
            b, t = x.size()
            current_tokens = b * t
            self.tokens_processed += current_tokens

            try:
                # Forward pass with AMP if enabled
                # Determine micro-batch count for current accumulation group
                rem = len(self.train_loader) % self.grad_accum_steps
                is_final_step = (step + 1) == len(self.train_loader)
                current_group_size = rem if (is_final_step and rem > 0) else self.grad_accum_steps

                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        logits, loss = self.model(x, y)
                        scaled_loss = loss / current_group_size
                else:
                    logits, loss = self.model(x, y)
                    scaled_loss = loss / current_group_size

                # Backward pass
                if self.use_amp:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                total_train_loss += loss.item()

                # Perform Optimizer step on accumulation boundary
                is_accum_boundary = ((step + 1) % self.grad_accum_steps == 0) or ((step + 1) == len(self.train_loader))
                if is_accum_boundary:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip)
                        self.optimizer.step()

                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1

                # Step Metrics Logging
                if (step + 1) % self.config.training.log_interval == 0 or (step + 1) == len(self.train_loader):
                    step_time = time.time() - step_start_time
                    tok_per_sec = current_tokens / max(step_time, 1e-5)
                    current_lr = self.optimizer.param_groups[0]["lr"]

                    vram_str = "N/A"
                    if self.device.type == "cuda":
                        alloc_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
                        res_gb = torch.cuda.memory_reserved(0) / (1024 ** 3)
                        vram_str = f"{alloc_gb:.2f}GB / {res_gb:.2f}GB"

                    print(
                        f"Epoch [{epoch}/{self.config.training.max_epochs}] | "
                        f"Step [{step + 1}/{len(self.train_loader)}] | "
                        f"Loss: {loss.item():.4f} | "
                        f"LR: {current_lr:.6f} | "
                        f"Speed: {tok_per_sec:.0f} tok/s | "
                        f"Tokens: {self.tokens_processed:,} | "
                        f"VRAM: {vram_str}"
                    )

            except torch.cuda.OutOfMemoryError as oom_err:
                torch.cuda.empty_cache()
                print("\n" + "!" * 68)
                print(" [X] CUDA OUT OF MEMORY (OOM) ERROR ENCOUNTERED!")
                print("!" * 68)
                print(" Helpful Recommendations to Fix VRAM OOM:")
                print(f"  1. Reduce batch_size in config.py (currently {self.config.training.batch_size}) -> Try 8 or 4")
                print(f"  2. Increase grad_accum_steps in config.py (currently {self.config.training.grad_accum_steps}) -> Try 8")
                print(f"  3. Reduce max_seq_len in config.py (currently {self.config.model.max_seq_len}) -> Try 256")
                print("!" * 68 + "\n")
                raise oom_err

        avg_train_loss = total_train_loss / len(self.train_loader)
        return avg_train_loss

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Computes validation loss over validation dataset.
        """
        self.model.eval()
        total_val_loss = 0.0

        for x, y in self.val_loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    logits, loss = self.model(x, y)
            else:
                logits, loss = self.model(x, y)

            total_val_loss += loss.item()

        avg_val_loss = total_val_loss / max(len(self.val_loader), 1)
        return avg_val_loss

    def save_checkpoint(self, filepath: str, epoch: int, val_loss: float, is_best: bool = False):
        """
        Saves model weights, optimizer state, and config as a plain dict.
        Config is stored as JSON-serializable primitives (no custom Python class),
        so torch.load(..., weights_only=True) will not encounter pickle issues.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        state = build_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            config=self.config,
            epoch=epoch,
            global_step=self.global_step,
            tokens_processed=self.tokens_processed,
            val_loss=val_loss,
            best_val_loss=self.best_val_loss,
        )
        torch.save(state, filepath)
        tag = "BEST MODEL" if is_best else "CHECKPOINT"
        print(f"[{tag}] Saved to: {filepath} (Val Loss: {val_loss:.4f}, format v{state['cracho_ckpt_version']})")

    def load_checkpoint(self, filepath: str):
        """
        Loads training state from a local checkpoint file to resume training.
        Uses the centralised load_checkpoint_file() helper which handles:
          - weights_only=False (required for optimizer state; safe for local files)
          - backward-compat normalisation of old checkpoints that stored Config objects
        """
        print(f"[*] Resuming training from checkpoint: {filepath}")
        state = load_checkpoint_file(filepath, self.device)

        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        sched_state = state.get("scheduler_state_dict", {})
        if sched_state:
            self.scheduler.load_state_dict(sched_state)

        self.start_epoch = state.get("epoch", 0)
        self.global_step = state.get("global_step", 0)
        self.tokens_processed = state.get("tokens_processed", 0)
        self.best_val_loss = state.get("best_val_loss", float("inf"))

        ver = state.get("cracho_ckpt_version", 1)
        print(f"[✓] Checkpoint v{ver} restored — "
              f"Epoch {self.start_epoch + 1}, Step {self.global_step}, "
              f"Best Val Loss: {self.best_val_loss:.4f}")

    def train(self):
        """
        Executes the main multi-epoch training loop.
        """
        eff_batch_size = self.config.training.batch_size * self.grad_accum_steps
        print("=" * 68)
        print("                   Starting CrachoLM Training")
        print("=" * 68)
        print(f" Target Device         : {self.device}")
        print(f" Mixed Precision AMP   : {self.use_amp}")
        print(f" Micro Batch Size      : {self.config.training.batch_size}")
        print(f" Grad Accumulation     : {self.grad_accum_steps} (Effective Batch Size: {eff_batch_size})")
        print(f" Max Epochs            : {self.config.training.max_epochs}")
        print(f" Train Batches/Epoch   : {len(self.train_loader)}")
        print(f" Val Batches/Epoch     : {len(self.val_loader)}")
        print("=" * 68 + "\n")

        for epoch in range(self.start_epoch + 1, self.config.training.max_epochs + 1):
            epoch_start_time = time.time()

            # Train for one epoch
            train_loss = self.train_epoch(epoch)

            # Evaluate on validation dataset
            val_loss = self.evaluate()
            perplexity = math.exp(min(val_loss, 20))

            epoch_time = time.time() - epoch_start_time

            print("\n" + "-" * 68)
            print(
                f"SUMMARY Epoch [{epoch}/{self.config.training.max_epochs}] | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Perplexity: {perplexity:.2f} | "
                f"Time: {epoch_time:.2f}s"
            )

            # Check if this is the best validation loss so far
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                best_path = os.path.join(self.config.system.checkpoints_dir, "best_model.pt")
                self.save_checkpoint(best_path, epoch, val_loss, is_best=True)

            # Save regular epoch checkpoint
            if epoch % self.config.training.checkpoint_interval == 0:
                ckpt_path = os.path.join(self.config.system.checkpoints_dir, f"checkpoint_epoch_{epoch}.pt")
                self.save_checkpoint(ckpt_path, epoch, val_loss, is_best=False)

            print("-" * 68 + "\n")

        print("=" * 68)
        print(f" Training Complete! Best Validation Loss: {self.best_val_loss:.4f}")
        print("=" * 68)
