import unittest
import torch
from config import ModelConfig
from src.tokenizer import CrachoTokenizer
from src.generator import generate_text
from src.chat_response import chat_prompt, is_greeting_only, format_chat_reply
from chat_turns_experiment import ChatDataset

class ChatTurnTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tok = CrachoTokenizer()
        self.tok.train_from_text(chat_prompt("hi") + "Hello!\nUser:hi")

    def fixed_model(self, text):
        ids = self.tok.encode(text) + [self.tok.eos_id]
        vocabulary = self.tok.vocab_size
        class Fixed(torch.nn.Module):
            config = ModelConfig(max_seq_len=256)
            position = 0
            def forward(self, x):
                logits = torch.full((1, 1, vocabulary), -100.0)
                logits[0, 0, ids[min(self.position, len(ids)-1)]] = 100.0
                self.position += 1
                return logits, None
        return Fixed()

    def test_stop_before_new_user_turn(self):
        prefix = chat_prompt("hi")
        result = generate_text(self.fixed_model("Hello!\nUser:hi"), self.tok,
                               prefix, max_new_tokens=50, greedy=True,
                               stop_sequences=("\nUser:",))
        self.assertEqual(result, prefix + "Hello!")

    def test_completion_mode_keeps_existing_behavior(self):
        prefix = chat_prompt("hi")
        result = generate_text(self.fixed_model("Hello!\nUser:hi"), self.tok,
                               prefix, max_new_tokens=50, greedy=True)
        self.assertEqual(result, prefix + "Hello!\nUser:hi")

    def test_greeting_with_question_is_not_truncated_as_greeting(self):
        for message in ["hi", "HI!!", "hello there!", "hey bro"]:
            self.assertTrue(is_greeting_only(message))
        for message in ["hi, what is 2 + 2?", "high energy", "hello, explain gravity"]:
            self.assertFalse(is_greeting_only(message))

    def test_targets_include_only_answer_and_end_token(self):
        prompt_ids = self.tok.encode(chat_prompt("hi"))
        answer_ids = self.tok.encode("Hello!") + [self.tok.eos_id]
        x, y = ChatDataset([("hi", "Hello!")], self.tok, seq_len=64)[0]
        start = len(prompt_ids)-1
        self.assertTrue(torch.all(y[:start] == 0))
        self.assertEqual(y[start:start+len(answer_ids)].tolist(), answer_ids)
        self.assertTrue(torch.all(y[start+len(answer_ids):] == 0))

    def test_display_spacing_does_not_change_words(self):
        self.assertEqual(format_chat_reply("Hi ! How can I help you ?"),
                         "Hi! How can I help you?")
        self.assertEqual(format_chat_reply("I ' m ready ."), "I'm ready.")

if __name__ == "__main__":
    unittest.main()
