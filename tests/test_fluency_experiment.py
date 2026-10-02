import io
import unittest
from fluency_experiment import END, clean_story, story_stream, unique_stories

class CorpusBoundaryTests(unittest.TestCase):
    def test_only_complete_stories_are_returned(self):
        stream = io.BytesIO(b"first" + END + b"second without delimiter")
        self.assertEqual(list(story_stream(stream, 1000)), ["first"])

    def test_byte_limit_is_respected(self):
        stream = io.BytesIO(b"first" + END + b"second" + END)
        limit = len(b"first" + END)
        self.assertEqual(list(story_stream(stream, limit)), ["first"])
        self.assertEqual(stream.tell(), limit)

    def test_training_duplicates_do_not_enter_validation(self):
        first = "A little child played in the park. " * 5
        second = "The dog went home after playing with a ball. " * 5
        seen = set()
        train = list(unique_stories(io.BytesIO(first.encode() + END), 1, seen, 10000))
        validation = list(unique_stories(
            io.BytesIO(first.encode() + END + second.encode() + END), 1, seen, 10000))
        self.assertEqual(train, [first.strip()])
        self.assertEqual(validation, [second.strip()])
        self.assertTrue(set(train).isdisjoint(validation))

    def test_incomplete_download_fails(self):
        with self.assertRaises(ValueError):
            list(unique_stories(io.BytesIO(b"no complete story"), 1, set(), 1000))

    def test_unicode_quotes_and_whitespace_are_normalized(self):
        self.assertEqual(clean_story(' \u201cHello,\u201d\\n said\\tTom. '.replace("\\n", "\n").replace("\\t", "\t")),
                         '"Hello," said Tom.')


class TokenizerIdentityTests(unittest.TestCase):
    def test_same_size_wrong_tokenizer_is_rejected(self):
        import hashlib
        import tempfile
        from dataclasses import asdict
        from pathlib import Path
        import torch
        from config import Config, ModelConfig
        from src.bpe_tokenizer import CrachoBPETokenizer
        from src.inference import load_model_for_inference
        from src.model import CrachoLM
        torch.set_num_threads(2)
        config = Config(model=ModelConfig(vocab_size=6, d_model=16, n_layers=1,
                                          n_heads=4, d_ff=32, max_seq_len=8))
        tok = CrachoBPETokenizer()
        tok.token2idx.update({"a": 4, "b": 5})
        tok.idx2token = {index: token for token, index in tok.token2idx.items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            token_path = path / "tokenizer_bpe.json"
            tok.save(str(token_path))
            checkpoint = path / "model.pt"
            torch.save({
                "config_dict": asdict(config),
                "model_state_dict": CrachoLM(config.model).state_dict(),
                "tokenizer_sha256": hashlib.sha256(token_path.read_bytes()).hexdigest(),
            }, checkpoint)
            loaded, matching_tok, _ = load_model_for_inference(checkpoint)
            self.assertEqual(matching_tok.vocab_size, 6)
            tok.token2idx["a"], tok.token2idx["b"] = 5, 4
            tok.idx2token = {index: token for token, index in tok.token2idx.items()}
            tok.save(str(token_path))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_model_for_inference(checkpoint)

if __name__ == "__main__":
    unittest.main()
