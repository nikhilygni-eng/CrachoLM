import copy
import os
import signal
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from config import Config, ModelConfig, TrainingConfig
from src.model import CrachoLM
from src.resumable_training import EpochBatchSampler, atomic_torch_save, training_loader
from train_250m import LowMemoryTrainer


class StopResumeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.cfg = Config(
            model=ModelConfig(vocab_size=32, d_model=16, n_layers=2, n_heads=4,
                              d_ff=64, max_seq_len=16, dropout=0.2,
                              use_sdpa=True, gradient_checkpointing=True),
            training=TrainingConfig(batch_size=2, grad_accum_steps=3,
                                    max_epochs=2, warmup_steps=1, learning_rate=0.001),
        )
        self.ids = torch.randint(1, 32, (13, 8))
        labels = self.ids.roll(-1, dims=1)
        labels[::2, -2:] = 0  # Unequal counts and a short final accumulation group.
        self.data = TensorDataset(self.ids, labels)
        self.state = copy.deepcopy(CrachoLM(self.cfg.model).state_dict())

    def make_trainer(self, path, resume=None, interval=100):
        path.mkdir()
        (path / "tokenizer_bpe.json").write_text("{}")
        cfg = copy.deepcopy(self.cfg)
        cfg.system.checkpoints_dir = str(path)
        model = CrachoLM(cfg.model)
        model.load_state_dict(self.state)
        loader = training_loader(self.data, batch_size=2, seed=42)
        validation = DataLoader(self.data, batch_size=2,
                                generator=torch.Generator().manual_seed(42))
        return LowMemoryTrainer(model, cfg, loader, validation, None, torch.device("cpu"),
                                resume_checkpoint_path=resume, checkpoint_every=interval)

    def assert_same(self, left, right):
        self.assertEqual(left.global_step, right.global_step)
        self.assertEqual(left.tokens_processed, right.tokens_processed)
        self.assertEqual(left.last_val_loss, right.last_val_loss)
        for a, b in zip(left.model.parameters(), right.model.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_real_sigint_during_accumulation_resumes_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = self.make_trainer(root / "full")
            torch.manual_seed(808)
            full.train()
            partial = self.make_trainer(root / "partial")
            original_forward = partial.model.forward
            calls = []
            def interrupted_forward(*args, **kwargs):
                calls.append(1)
                if len(calls) == 2:
                    os.kill(os.getpid(), signal.SIGINT)
                    os.kill(os.getpid(), signal.SIGINT)
                return original_forward(*args, **kwargs)
            partial.model.forward = interrupted_forward
            previous = signal.getsignal(signal.SIGINT)
            torch.manual_seed(808)
            self.assertFalse(partial.train())
            self.assertEqual(signal.getsignal(signal.SIGINT), previous)
            saved = root / "partial/last.pt"
            state = torch.load(saved, weights_only=False)
            self.assertEqual(state["epoch"], 0)
            self.assertEqual(state["training_progress"]["next_batch"], 3)
            self.assertEqual(state["global_step"], 1)
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(p.grad is None for p in partial.model.parameters()))
            resumed = self.make_trainer(root / "resumed", str(saved))
            self.assertTrue(resumed.train())
            self.assert_same(full, resumed)

    def test_sigterm_during_validation_resumes_without_retraining_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            full = self.make_trainer(root / "full")
            torch.manual_seed(909)
            full.train()
            partial = self.make_trainer(root / "partial")
            original_forward = partial.model.forward
            sent = []
            def interrupted_forward(*args, **kwargs):
                if not partial.model.training and not sent:
                    sent.append(True)
                    os.kill(os.getpid(), signal.SIGTERM)
                return original_forward(*args, **kwargs)
            partial.model.forward = interrupted_forward
            torch.manual_seed(909)
            self.assertFalse(partial.train())
            saved = root / "partial/last.pt"
            state = torch.load(saved, weights_only=False)
            self.assertEqual(state["training_progress"]["next_batch"], 7)
            self.assertEqual(state["global_step"], 3)
            resumed = self.make_trainer(root / "resumed", str(saved))
            self.assertTrue(resumed.train())
            self.assert_same(full, resumed)

    def test_periodic_checkpoint_and_stop_before_first_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trainer = self.make_trainer(root / "periodic", interval=1)
            records = []
            save = trainer.save_checkpoint
            def record(*args, **kwargs):
                records.append((trainer.global_step, trainer.next_batch))
                save(*args, **kwargs)
            trainer.save_checkpoint = record
            trainer.train()
            self.assertIn((1, 3), records)
            self.assertIn((2, 6), records)
            untouched = self.make_trainer(root / "untouched")
            untouched.request_stop()
            self.assertFalse(untouched.train())
            saved = root / "untouched/last.pt"
            state = torch.load(saved, weights_only=False)
            self.assertEqual(state["global_step"], 0)
            self.assertEqual(state["training_progress"]["next_batch"], 0)
            resumed = self.make_trainer(root / "resumed", str(saved))
            self.assertTrue(resumed.train())

    def test_failed_save_retains_previous_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "last.pt"
            atomic_torch_save({"step": 5}, path)
            def fail(state, handle):
                handle.write(b"incomplete new file")
                raise OSError("simulated full disk")
            with patch("src.resumable_training.torch.save", side_effect=fail):
                with self.assertRaises(OSError):
                    atomic_torch_save({"step": 6}, path)
            self.assertEqual(torch.load(path, weights_only=True)["step"], 5)

    def test_sampler_reconstructs_tail_without_touching_global_rng(self):
        before = torch.get_rng_state().clone()
        sampler = EpochBatchSampler(13, 2, 42)
        all_batches = list(sampler)
        sampler.set_position(1, 3)
        self.assertEqual(list(sampler), all_batches[3:])
        self.assertEqual(sorted(sum(all_batches, [])), list(range(13)))
        torch.testing.assert_close(torch.get_rng_state(), before)


if __name__ == "__main__":
    unittest.main()
