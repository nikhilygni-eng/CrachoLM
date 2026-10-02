"""Single-turn chat formatting and output bounds; replies come from the model."""
import re
from src.generator import generate_text

_GREETING = re.compile(
    r"(?:hi|hello|hey|hiya|good (?:morning|afternoon|evening))"
    r"(?:[\s,]+(?:bro|there|cracholm))?[\s!.,]*", re.IGNORECASE
)

def is_greeting_only(message):
    return bool(_GREETING.fullmatch(message.strip()))

def chat_prompt(message):
    message = message.strip()
    if not message:
        raise ValueError("Enter a message.")
    return f"User: {message}\nAssistant:"

def format_chat_reply(text):
    if chr(96) * 3 in text:
        return text.strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"\b(\w+)\s*'\s*(s|t|re|ve|ll|d|m)\b", r"\1'\2", text,
                  flags=re.IGNORECASE)
    return text.strip()

def generate_chat_reply(model, tokenizer, message, max_new_tokens=80,
                        temperature=0.7, top_k=40, greedy=False, device=None):
    prompt = chat_prompt(message)
    budget = min(max(1, int(max_new_tokens)),
                 32 if is_greeting_only(message) else 192)
    kwargs = {}
    if device is not None:
        kwargs["device"] = device
    result = generate_text(
        model, tokenizer, prompt, max_new_tokens=budget,
        temperature=temperature, top_k=top_k, greedy=greedy,
        stop_sequences=("\nUser:", "\nAssistant:"),
        **kwargs,
    )
    return format_chat_reply(result[len(prompt):]), budget
