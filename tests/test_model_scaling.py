import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import torch

from config import Config, ModelConfig, TrainingConfig, default_config
from config_250m import make_250m_config
from src.model import CrachoLM
from train_250m import LowMemoryTrainer, fresh_directory


class ModelScalingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.small = ModelConfig(vocab_size=32, d_model=16, n_layers=2,
                                 n_heads=4, d_ff=64, max_seq_len=16, dropout=0.0)

    def test_sdpa_matches_manual_logits_gradients_and_causal_prefix(self):
        manual = CrachoLM(self.small)
        fast = CrachoLM(replace(self.small, use_sdpa=True))
        fast.load_state_dict(manual.state_dict(), strict=True)
        ids = torch.randint(1, 32, (2, 8))
        labels = torch.randint(1, 32, (2, 8))
        left, loss1 = manual(ids, labels)
        right, loss2 = fast(ids, labels)
        torch.testing.assert_close(left, right, atol=1e-6, rtol=1e-5)
        loss1.backward()
        loss2.backward()
        for a, b in zip(manual.parameters(), fast.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-4)
        modified = ids.clone()
        modified[:, 4:] = torch.randint(1, 32, (2, 4))
        changed, _ = fast(modified, labels)
        torch.testing.assert_close(right[:, :4], changed[:, :4], atol=1e-6, rtol=1e-5)

    def test_checkpointing_matches_gradients_with_dropout(self):
        cfg = replace(self.small, dropout=0.1, use_sdpa=True)
        ordinary = CrachoLM(cfg)
        recomputed = CrachoLM(replace(cfg, gradient_checkpointing=True))
        recomputed.load_state_dict(ordinary.state_dict(), strict=True)
        ids = torch.randint(1, 32, (2, 8))
        torch.manual_seed(99)
        _, loss1 = ordinary(ids, ids)
        loss1.backward()
        torch.manual_seed(99)
        _, loss2 = recomputed(ids, ids)
        loss2.backward()
        torch.testing.assert_close(loss1, loss2)
        for a, b in zip(ordinary.parameters(), recomputed.parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-4)
        recomputed.eval()
        with torch.no_grad():
            first, _ = recomputed(ids)
            second, _ = recomputed(ids)
        torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_separate_preset_count_and_original_defaults(self):
        before = asdict(default_config)
        config = make_250m_config()
        with torch.device("meta"):
            model = CrachoLM(config.model)
        self.assertEqual(model.get_num_params(), 250431488)
        self.assertIs(model.tok_emb.weight, model.lm_head.weight)
        self.assertEqual(before, asdict(default_config))
        self.assertEqual(default_config.model.d_model, 768)
        self.assertEqual(default_config.model.n_layers, 10)
        self.assertFalse(default_config.model.use_sdpa)
        self.assertFalse(default_config.model.gradient_checkpointing)

    def test_existing_output_directory_is_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing_model"
            path.mkdir()
            old = path / "best_model.pt"
            old.write_bytes(b"existing model")
            with self.assertRaises(FileExistsError):
                fresh_directory(path)
            self.assertEqual(old.read_bytes(), b"existing model")

    def test_epoch_checkpoint_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "tokenizer_bpe.json").write_text("{}")
            cfg = Config(model=self.small, training=TrainingConfig(
                batch_size=1, grad_accum_steps=2, max_epochs=2,
                warmup_steps=1, learning_rate=0.001, log_interval=100,
            ))
            x = torch.randint(1, 32, (4, 8))
            data = torch.utils.data.TensorDataset(x, x.roll(-1, dims=1))
            loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)
            first = LowMemoryTrainer(CrachoLM(self.small), cfg, loader, loader,
                                     None, torch.device("cpu"))
            first.train_epoch(1)
            saved = path / "epoch1.pt"
            first.save_checkpoint(str(saved), 1, 4.0)
            first.train_epoch(2)
            resumed = LowMemoryTrainer(CrachoLM(self.small), cfg, loader, loader,
                                       None, torch.device("cpu"), str(saved))
            self.assertEqual(resumed.start_epoch, 1)
            resumed.train_epoch(2)
            self.assertEqual(resumed.global_step, first.global_step)
            self.assertEqual(resumed.tokens_processed, first.tokens_processed)
            for a, b in zip(first.model.parameters(), resumed.model.parameters()):
                torch.testing.assert_close(a, b, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
