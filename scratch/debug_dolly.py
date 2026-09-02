import urllib.request
import json

url = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
headers = {"User-Agent": "Mozilla/5.0 (CrachoLM Benchmark Data Prep)"}
req = urllib.request.Request(url, headers=headers)

with urllib.request.urlopen(req, timeout=15) as resp:
    text = resp.read().decode("utf-8")
    lines = text.splitlines()
    print(f"Total raw lines fetched: {len(lines)}")
    if lines:
        print(f"First line sample: {lines[0][:150]}")
        item = json.loads(lines[0])
        print(f"Parsed keys: {list(item.keys())}")
