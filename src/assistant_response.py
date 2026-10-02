"""App response routing; raw model evaluations bypass this module."""
from src.chat_response import generate_chat_reply
from src.local_calculator import try_calculation


def generate_assistant_reply(model, tokenizer, message, use_tools=True, **kwargs):
    if use_tools:
        result=try_calculation(message)
        if result is not None:
            return result['reply'],0,result['source']
    reply,budget=generate_chat_reply(model,tokenizer,message,**kwargs)
    return reply,budget,'model'
