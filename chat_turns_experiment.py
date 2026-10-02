#!/usr/bin/env python3
"""Small supervised chat-turn experiment. This is not general assistant training."""
import argparse
import hashlib
import json
import pathlib
import shutil
from dataclasses import asdict
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from config import Config, TrainingConfig
from src.inference import load_model_for_inference
from src.trainer import CrachoTrainer
from src.chat_response import chat_prompt, generate_chat_reply

ROOT = pathlib.Path(__file__).resolve().parent

def training_pairs():
    greetings = ["hi", "Hi!", "HI", "hello", "Hello!", "hey", "Hey!", "hiya",
                 "hi bro", "hello there", "hey there", "good morning",
                 "good afternoon", "good evening", "hello CrachoLM", "hi CrachoLM"]
    pairs = [(text, "Hi! How can I help you?") for text in greetings] * 4
    pairs += [
        ("how are you?", "I am ready to help. How are you?"),
        ("how are you doing?", "I am here and ready to help."),
        ("who are you?", "I am CrachoLM, a small experimental language model."),
        ("what is your name?", "My name is CrachoLM."),
        ("what's your name?", "My name is CrachoLM."),
        ("are you a human?", "No. I am a language model."),
        ("what can you do?", "I can try to answer questions and help with writing."),
        ("can you make mistakes?", "Yes. I can make mistakes, so check important information."),
        ("thanks", "You are welcome!"),
        ("thank you", "You are welcome!"),
        ("thanks bro", "You are welcome!"),
        ("thank you for helping", "You are welcome!"),
        ("bye", "Goodbye! Have a good day."),
        ("goodbye", "Goodbye! Have a good day."),
        ("see you", "See you!"),
        ("please keep it short", "Okay, I will keep my answers short."),
        ("stop", "Okay."),
        ("What is machine learning?", "Machine learning lets computers learn patterns from data."),
        ("Explain machine learning.", "Machine learning lets computers learn patterns from data."),
        ("What is a sentence?", "A sentence expresses a complete thought."),
        ("What is a noun?", "A noun names a person, place, thing, or idea."),
        ("What is a verb?", "A verb describes an action or a state."),
        ("What is a tokenizer?", "A tokenizer splits text into units called tokens."),
        ("What is a GPU?", "A GPU performs many calculations in parallel."),
        ("What does CPU stand for?", "CPU stands for central processing unit."),
        ("What is Python?", "Python is a programming language."),
        ("What is a variable?", "A variable is a name that refers to a value."),
        ("What is a function?", "A function is a reusable block of code."),
        ("What is photosynthesis?", "Plants use sunlight to make food from water and carbon dioxide."),
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What is the capital of India?", "The capital of India is New Delhi."),
        ("What is the opposite of hot?", "The opposite of hot is cold."),
        ("What is the opposite of big?", "The opposite of big is small."),
        ("What is the opposite of happy?", "The opposite of happy is sad."),
        ("Name a fruit.", "An apple is a fruit."),
        ("Name an animal.", "A dog is an animal."),
        ("How many days are in a week?", "There are seven days in a week."),
        ("How many months are in a year?", "There are twelve months in a year."),
        ("Complete the sentence: Tom was hungry, so he", "Tom was hungry, so he ate some food."),
        ("Complete the sentence: The dog saw a ball and", "The dog saw a ball and ran after it."),
        ("Fix this sentence: She go to school.", "She goes to school."),
        ("Fix this sentence: He are happy.", "He is happy."),
        ("Fix this sentence: They is playing.", "They are playing."),
        ("Fix this sentence: I has a pen.", "I have a pen."),
        ("Fix this sentence: We was late.", "We were late."),
    ]
    for a in range(2, 10):
        for b in range(1, 5):
            pairs += [
                (f"What is {a} + {b}?", f"{a + b}."),
                (f"Calculate {a} plus {b}.", f"{a + b}."),
                (f"Hi, what is {a} + {b}?", f"{a + b}."),
            ]
    return pairs

VALIDATION = [
    ("Hi there!", "Hi! How can I help you?"),
    ("HELLO", "Hi! How can I help you?"),
    ("hey bro", "Hi! How can I help you?"),
    ("Good evening!", "Hi! How can I help you?"),
    ("Thank you so much!", "You are welcome!"),
    ("Tell me your name.", "My name is CrachoLM."),
    ("Please calculate 5 plus 3.", "8."),
]

class ChatDataset(Dataset):
    def __init__(self, pairs, tokenizer, seq_len=128):
        self.samples = []
        for message, reply in pairs:
            prefix = tokenizer.encode(chat_prompt(message), add_special_tokens=False)
            answer = tokenizer.encode(reply, add_special_tokens=False) + [tokenizer.eos_id]
            ids = prefix + answer
            if tokenizer.unk_id in ids:
                raise ValueError(f"Unknown tokens in chat example: {message!r}")
            if len(ids) - 1 > seq_len:
                raise ValueError(f"Chat example exceeds context: {message!r}")
            x = torch.full((seq_len,), tokenizer.pad_id, dtype=torch.long)
            y = torch.zeros(seq_len, dtype=torch.long)
            x[:len(ids)-1] = torch.tensor(ids[:-1])
            y[:len(ids)-1] = torch.tensor(ids[1:])
            y[:len(prefix)-1] = 0
            assert y[len(prefix)-1].item() == answer[0]
            assert y[len(ids)-2].item() == tokenizer.eos_id
            self.samples.append((x, y))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        return self.samples[index]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(ROOT/"checkpoints_fluency_stage2_20260927/best_model.pt"))
    parser.add_argument("--output", default=str(ROOT/"checkpoints_chat_turns_20260927"))
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()
    output = pathlib.Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    torch.set_num_threads(2)
    torch.manual_seed(13)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, source = load_model_for_inference(args.checkpoint, device=device)
    pairs = training_pairs()
    assert not set(message for message, _ in pairs).intersection(message for message, _ in VALIDATION)
    data = ChatDataset(pairs, tokenizer)
    val = ChatDataset(VALIDATION, tokenizer)
    sampler = RandomSampler(data, replacement=True, num_samples=args.steps*4*4,
                            generator=torch.Generator().manual_seed(13))
    loader = DataLoader(data, batch_size=4, sampler=sampler)
    val_loader = DataLoader(val, batch_size=4)
    config = Config(model=model.config)
    config.training = TrainingConfig(batch_size=4, grad_accum_steps=4,
        learning_rate=0.0001, min_lr=0.00002, max_epochs=1,
        warmup_steps=10, log_interval=80, weight_decay=0.01)
    config.system.checkpoints_dir = str(output)
    trainer = CrachoTrainer(model, config, loader, val_loader, tokenizer, device)
    prompts = ["hi", "Hi!", "hello", "hey", "Hi there!", "hey bro",
               "Hi, what is 2 + 2?", "What is machine learning?", "Tell me your name."]
    def sample():
        results = []
        for prompt in prompts:
            torch.manual_seed(42)
            reply, budget = generate_chat_reply(model, tokenizer, prompt,
                max_new_tokens=64, greedy=True, device=device)
            results.append({"prompt":prompt, "reply":reply, "budget":budget})
        return results
    before_loss = trainer.evaluate()
    before = sample()
    print(f"Training {args.steps} updates on {len(set(pairs))} authored examples; only assistant answers and EOS contribute to loss.", flush=True)
    trainer.train_epoch(1)
    after_loss = trainer.evaluate()
    after = sample()
    greeting_ok = all(
        row["reply"].lower().startswith(("hi", "hello", "hey")) and
        0 < len(row["reply"].split()) <= 12
        for row in after[:6]
    )
    report = {"scope":"Small authored single-turn chat curriculum, not a general capability benchmark.",
              "training_unique_pairs":len(set(pairs)), "validation_pairs":VALIDATION,
              "source_model":source, "optimizer_steps":trainer.global_step,
              "validation_loss_before":before_loss,"validation_loss_after":after_loss,
              "before":before,"after":after,"greeting_smoke_passed":greeting_ok}
    output.mkdir(parents=True, exist_ok=True)
    (output/"training_pairs.json").write_text(json.dumps(pairs,indent=2)+"\n")
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    checkpoint = {
        "model_state_dict":model.state_dict(),"config_dict":asdict(config),
        "epoch":1,"global_step":trainer.global_step,
        "val_loss":after_loss,"best_val_loss":after_loss,
        "checkpoint_kind":"inference_weights_only",
        "chat_format":"User: {message}\\nAssistant:",
        "source_checkpoint":source["checkpoint"],
        "tokenizer_sha256":hashlib.sha256(pathlib.Path(source["tokenizer"]).read_bytes()).hexdigest(),
    }
    torch.save(checkpoint, output/"best_model.pt")
    shutil.copy2(source["tokenizer"], output/"tokenizer_bpe.json")
    print(json.dumps(report,indent=2),flush=True)

if __name__ == "__main__":
    main()
