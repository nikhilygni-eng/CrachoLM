"""Regression checks for loading, token boundaries, and effective-batch training."""
import copy
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from config import Config, ModelConfig, TrainingConfig
from src.bpe_tokenizer import CrachoBPETokenizer
from src.model import CrachoLM
from src.generator import generate_text
from src.trainer import CrachoTrainer, get_warmup_cosine_scheduler
from src.inference import load_model_for_inference


def toy_tokenizer():
    tok = CrachoBPETokenizer()
    tok.token2idx.update({"a": 4, "a</w>": 5, "b": 6, "b</w>": 7, "</w>": 8})
    tok.idx2token = {v: k for k, v in tok.token2idx.items()}
    tok.merges = [("a", "</w>"), ("b", "</w>")]
    return tok


def tiny_config():
    return Config(
        model=ModelConfig(vocab_size=9, d_model=16, n_layers=1, n_heads=4,
                          d_ff=32, max_seq_len=8, dropout=0.0),
        training=TrainingConfig(batch_size=2, grad_accum_steps=4,
                                learning_rate=0.001, min_lr=0.001, max_epochs=1,
                                warmup_steps=0, weight_decay=0.0, grad_clip=100.0,
                                log_interval=1000),
    )


class SentenceQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_tokenization_keeps_word_boundary_for_short_prompts(self):
        tok = toy_tokenizer()
        self.assertEqual(tok.encode("a"), [5])
        self.assertEqual(tok.encode("a"), tok.encode("a "))
        self.assertEqual(tok.encode("<eos>"), [tok.eos_id])

    def test_generation_preserves_exact_prompt(self):
        class FixedModel(torch.nn.Module):
            config = ModelConfig(max_seq_len=8)
            def forward(self, x):
                logits = torch.full((1, 1, 9), -100.0)
                logits[0, 0, 7] = 100.0
                return logits, None
        for prompt in ("a", "a\n"):
            output = generate_text(FixedModel(), toy_tokenizer(), prompt,
                                   max_new_tokens=1, greedy=True)
            self.assertTrue(output.startswith(prompt))
            self.assertEqual(output[len(prompt):].strip(), "b")

    def test_one_step_run_changes_weights(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = get_warmup_cosine_scheduler(optimizer, 100, 1)
        parameter.square().backward()
        optimizer.step()
        scheduler.step()
        self.assertLess(parameter.item(), 1.0)

    def test_scheduler_does_not_rebound_after_planned_run(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        scheduler = get_warmup_cosine_scheduler(optimizer, 2, 10)
        for _ in range(25):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.01)

    def test_attention_cannot_read_future_tokens(self):
        model = CrachoLM(tiny_config().model).eval()
        first = torch.tensor([[4, 5, 6, 7]])
        second = torch.tensor([[4, 5, 6, 8]])
        with torch.no_grad():
            a, _ = model(first, first)
            b, _ = model(second, second)
        torch.testing.assert_close(a[:, :3], b[:, :3], rtol=0, atol=1e-6)

    def test_accumulation_matches_full_batches_with_short_final_batch(self):
        torch.manual_seed(31)
        config = tiny_config()
        x = torch.randint(1, 9, (11, 8))
        y = torch.randint(1, 9, (11, 8))
        y[1, 3:] = 0
        y[-1, 2:] = 0
        dataset = TensorDataset(x, y)
        micro = DataLoader(dataset, batch_size=2)
        full = DataLoader(dataset, batch_size=8)
        a = CrachoLM(config.model)
        b = copy.deepcopy(a)
        full_config = copy.deepcopy(config)
        full_config.training.batch_size = 8
        full_config.training.grad_accum_steps = 1
        trainer_a = CrachoTrainer(a, config, micro, micro, toy_tokenizer(), torch.device("cpu"))
        trainer_b = CrachoTrainer(b, full_config, full, full, toy_tokenizer(), torch.device("cpu"))
        trainer_a.train_epoch(1)
        trainer_b.train_epoch(1)
        self.assertEqual(trainer_a.global_step, 2)
        for a_param, b_param in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(a_param, b_param, rtol=1e-4, atol=2e-6)
        with torch.no_grad():
            _, expected_loss = a.eval()(x, y)
        self.assertAlmostEqual(trainer_a.evaluate(), expected_loss.item(), places=6)

    def test_loader_pairs_tokenizer_and_rejects_wrong_vocabulary(self):
        config = tiny_config()
        model = CrachoLM(config.model).eval()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "best_model.pt"
            toy_tokenizer().save(str(root / "tokenizer_bpe.json"))
            torch.save({"config_dict": asdict(config),
                        "model_state_dict": model.state_dict()}, path)
            loaded, tok, info = load_model_for_inference(path, device="cpu")
            self.assertEqual(tok.vocab_size, 9)
            self.assertEqual(info["parameters"], model.get_num_params())
            x = torch.tensor([[4, 5]])
            torch.testing.assert_close(loaded(x)[0], model(x)[0])
            config.model.vocab_size = 10
            torch.save({"config_dict": asdict(config),
                        "model_state_dict": model.state_dict()}, path)
            with self.assertRaisesRegex(ValueError, "checkpoint expects"):
                load_model_for_inference(path, device="cpu")


if __name__ == "__main__":
    unittest.main()
