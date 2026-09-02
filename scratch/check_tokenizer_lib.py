import sys

for pkg in ["tokenizers", "tiktoken", "sentencepiece"]:
    try:
        m = __import__(pkg)
        print(f"[FOUND] {pkg} (version: {getattr(m, '__version__', 'unknown')})")
    except ImportError:
        print(f"[NOT FOUND] {pkg}")
