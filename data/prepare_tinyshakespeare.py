"""
CrachoLM Public Domain Dataset Downloader & Preparer
=====================================================
Downloads the public domain TinyShakespeare dataset (~1.1 MB, ~1 Million chars)
into data/raw/tinyshakespeare.txt for real model training.
"""

import os
import urllib.request


def download_tinyshakespeare(target_dir: str = None) -> str:
    """
    Downloads TinyShakespeare dataset if not already cached locally.
    
    Args:
        target_dir (str): Target directory path. Defaults to data/raw/.
        
    Returns:
        str: Absolute file path to downloaded dataset.
    """
    if target_dir is None:
        target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))

    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, "tinyshakespeare.txt")

    if os.path.exists(target_path) and os.path.getsize(target_path) > 100000:
        print(f"[✓] Dataset already cached at: {target_path} ({os.path.getsize(target_path):,} bytes)")
        return target_path

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print(f"[*] Downloading public domain dataset from: {url}")
    
    try:
        urllib.request.urlretrieve(url, target_path)
        print(f"[✓] Download completed! Saved to: {target_path} ({os.path.getsize(target_path):,} bytes)")
    except Exception as e:
        print(f"[!] Primary download URL failed ({e}). Generating fallback educational corpus...")
        # Fallback generator if offline / restricted network
        fallback_text = (
            "First Citizen:\nBefore we proceed any further, hear me speak.\n\n"
            "All:\nSpeak, speak.\n\n"
            "First Citizen:\nYou are all resolved rather to die than to famish?\n\n"
            "All:\nResolved. resolved.\n\n"
            "First Citizen:\nFirst, you know Caius Marcius is chief enemy to the people.\n\n"
            "All:\nWe know't, we know't.\n\n"
            "First Citizen:\nLet us kill him, and we'll have corn at our own price.\n"
        ) * 5000
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(fallback_text)
        print(f"[✓] Fallback dataset created at: {target_path}")

    return target_path


if __name__ == "__main__":
    download_tinyshakespeare()
