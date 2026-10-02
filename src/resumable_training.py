"""Deterministic data position and graceful stopping for the 250M trainer."""
import math
import os
import signal
from contextlib import contextmanager
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Sampler


class EpochBatchSampler(Sampler):
    """Recreate one epoch's permutation without consuming model/dropout RNG."""
    def __init__(self, size, batch_size, seed):
        if size < 1 or batch_size < 1:
            raise ValueError("Dataset and batch size must be positive.")
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.epoch, self.start_batch = 1, 0
        self.total_batches = math.ceil(size / batch_size)

    def set_position(self, epoch, start_batch):
        if epoch < 1 or not 0 <= start_batch <= self.total_batches:
            raise ValueError("Invalid saved data position.")
        self.epoch, self.start_batch = epoch, start_batch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.size, generator=generator).tolist()
        for batch in range(self.start_batch, self.total_batches):
            start = batch * self.batch_size
            yield order[start:start + self.batch_size]

    def __len__(self):
        return self.total_batches - self.start_batch


def training_loader(dataset, batch_size=1, seed=42):
    return DataLoader(
        dataset, batch_sampler=EpochBatchSampler(len(dataset), batch_size, seed),
        num_workers=0, generator=torch.Generator().manual_seed(seed),
    )


def atomic_torch_save(state, destination):
    """Keep the previous completed checkpoint if serialization is interrupted."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with temporary.open("wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class ResumableTrainerMixin:
    def __init__(self, *args, checkpoint_every=100, **kwargs):
        if checkpoint_every < 1:
            raise ValueError("Checkpoint interval must be positive.")
        self.checkpoint_every = checkpoint_every
        self.stop_requested = False
        self.current_epoch, self.next_batch = 1, 0
        self.epoch_loss_sum, self.epoch_target_tokens = 0.0, 0
        self.last_val_loss = None
        super().__init__(*args, **kwargs)

    def request_stop(self, signum=None, frame=None):
        # Do not serialize or raise inside a signal handler: autograd/AdamW may
        # be partway through modifying tensors. Repeated Ctrl+C remains safe.
        self.stop_requested = True

    @contextmanager
    def stop_handlers(self):
        signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGHUP"):
            signals.append(signal.SIGHUP)
        previous = {sig: signal.getsignal(sig) for sig in signals}
        try:
            for sig in signals:
                signal.signal(sig, self.request_stop)
            yield
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def progress_state(self):
        sampler = self.train_loader.batch_sampler
        if not isinstance(sampler, EpochBatchSampler):
            return None  # Compatibility with legacy epoch-only callers/tests.
        return dict(version=1, epoch=self.current_epoch, next_batch=self.next_batch,
                    epoch_loss_sum=self.epoch_loss_sum,
                    epoch_target_tokens=self.epoch_target_tokens,
                    sampler_seed=sampler.seed, dataset_size=sampler.size,
                    batch_size=sampler.batch_size, grad_accum=self.grad_accum_steps)

    def restore_progress(self, state):
        self.last_val_loss = state.get("val_loss")
        self.current_epoch = state["epoch"] + 1
        progress = state.get("training_progress")
        if progress is None:
            return  # Old checkpoint was taken after a complete epoch.
        sampler = self.train_loader.batch_sampler
        if not isinstance(sampler, EpochBatchSampler):
            raise ValueError("Mid-epoch resume requires the resumable data loader.")
        expected = dict(sampler_seed=sampler.seed, dataset_size=sampler.size,
                        batch_size=sampler.batch_size, grad_accum=self.grad_accum_steps)
        if progress.get("version") != 1 or any(progress[k] != v for k, v in expected.items()):
            raise ValueError("Saved data order or accumulation differs from this run.")
        if progress["epoch"] != self.current_epoch:
            raise ValueError("Checkpoint epoch metadata is inconsistent.")
        next_batch = progress["next_batch"]
        if next_batch != sampler.total_batches and next_batch % self.grad_accum_steps:
            raise ValueError("Checkpoint is not at a completed accumulation boundary.")
        sampler.set_position(self.current_epoch, next_batch)
        # Keep __len__ full until CrachoTrainer has constructed its scheduler;
        # it already did so before invoking load_checkpoint.
        self.next_batch = next_batch
        self.epoch_loss_sum = progress["epoch_loss_sum"]
        self.epoch_target_tokens = progress["epoch_target_tokens"]

    def save_latest(self):
        destination = Path(self.config.system.checkpoints_dir) / "last.pt"
        self.save_checkpoint(str(destination), self.current_epoch - 1, self.last_val_loss)

    def _at_update_boundary(self, did_step):
        if self.stop_requested:
            print("Stop requested. Saving current progress; please wait...", flush=True)
            self.save_latest()
            return True
        if did_step and self.global_step % self.checkpoint_every == 0:
            self.save_latest()
            if self.stop_requested:
                return True
        return False

    def _train_remaining_epoch(self):
        sampler = self.train_loader.batch_sampler
        sampler.set_position(self.current_epoch, self.next_batch)
        total_batches = sampler.total_batches
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        if self.stop_requested:
            return self._at_update_boundary(False)
        group_tokens, nominal_tokens = 0, 0
        for batch_index, (x, y) in enumerate(self.train_loader, start=self.next_batch):
            x, y = x.to(self.device), y.to(self.device)
            valid_tokens = int((y != 0).sum().item())
            if valid_tokens < 1:
                raise ValueError("Training batch has no non-padding targets.")
            if batch_index % self.grad_accum_steps == 0:
                nominal_tokens = x.numel() * self.grad_accum_steps
                group_tokens = 0
            # A persistent half-precision weight cache alongside accumulated
            # gradients and restored Adam states can exceed the 6GB GPU.
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp,
                                cache_enabled=False):
                logits, loss = self.model(x, y)
                scaled_loss = loss * (valid_tokens / nominal_tokens)
            value = loss.item()
            if not math.isfinite(value):
                raise RuntimeError("Non-finite training loss; the last completed checkpoint is retained.")
            if self.use_amp:
                self.scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            del logits, loss, scaled_loss
            group_tokens += valid_tokens
            self.epoch_loss_sum += value * valid_tokens
            self.epoch_target_tokens += valid_tokens
            self.tokens_processed += x.numel()
            boundary = ((batch_index + 1) % self.grad_accum_steps == 0
                        or batch_index + 1 == total_batches)
            if not boundary:
                continue
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            for parameter in self.model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(nominal_tokens / group_tokens)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.grad_clip,
                                           foreach=False)
            did_step = True
            if self.use_amp:
                old_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                did_step = self.scaler.get_scale() >= old_scale
            else:
                self.optimizer.step()
            if did_step:
                self.scheduler.step()
                self.global_step += 1
            self.optimizer.zero_grad(set_to_none=True)
            self.next_batch = batch_index + 1
            if self.global_step % 10 == 0 or self.next_batch == total_batches:
                print(f"Epoch {self.current_epoch}/{self.config.training.max_epochs} | "
                      f"batch {self.next_batch}/{total_batches} | update {self.global_step} | "
                      f"loss {value:.4f} | tokens {self.tokens_processed:,}", flush=True)
            if self._at_update_boundary(did_step):
                return True
        return False

    @torch.no_grad()
    def _evaluate_stoppable(self):
        self.model.eval()
        total_loss, total_tokens = 0.0, 0
        for x, y in self.val_loader:
            if self.stop_requested:
                return None
            x, y = x.to(self.device), y.to(self.device)
            with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.use_amp,
                                cache_enabled=False):
                _, loss = self.model(x, y)
            count = int((y != 0).sum().item())
            total_loss += loss.item() * count
            total_tokens += count
        if total_tokens == 0:
            raise ValueError("Validation data has no non-padding targets.")
        result = total_loss / total_tokens
        if not math.isfinite(result):
            raise RuntimeError("Non-finite validation loss.")
        return result

    def train(self):
        if not isinstance(self.train_loader.batch_sampler, EpochBatchSampler):
            raise ValueError("Use training_loader for resumable training.")
        if self.train_loader.num_workers != 0 or self.val_loader.num_workers != 0:
            raise ValueError("This resumable trainer requires num_workers=0.")
        with self.stop_handlers():
            print("Training ready. Press Ctrl+C once to save progress and stop. "
                  "Wait for 'Checkpoint saved; training stopped safely.'", flush=True)
            while self.current_epoch <= self.config.training.max_epochs:
                stopped = self._train_remaining_epoch()
                if stopped:
                    print("Checkpoint saved; training stopped safely.", flush=True)
                    return False
                validation = self._evaluate_stoppable()
                if self.stop_requested:
                    self._at_update_boundary(False)
                    print("Checkpoint saved; training stopped safely.", flush=True)
                    return False
                self.last_val_loss = validation
                is_best = validation < self.best_val_loss
                if is_best:
                    self.best_val_loss = validation
                completed_epoch = self.current_epoch
                average = self.epoch_loss_sum / max(1, self.epoch_target_tokens)
                self.current_epoch += 1
                self.start_epoch = completed_epoch
                self.next_batch = 0
                self.epoch_loss_sum, self.epoch_target_tokens = 0.0, 0
                print(f"Epoch {completed_epoch} complete: train loss {average:.4f}; "
                      f"validation loss {validation:.4f}", flush=True)
                self.save_latest()
                if is_best and not self.stop_requested:
                    self.save_checkpoint(str(Path(self.config.system.checkpoints_dir) / "best_model.pt"),
                                         completed_epoch, validation, is_best=True)
                if self.stop_requested:
                    print("Checkpoint saved; training stopped safely.", flush=True)
                    return False
            print("Training complete. Latest checkpoint: last.pt", flush=True)
            return True
