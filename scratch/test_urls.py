import urllib.request

urls_to_test = [
    ("Dolly HF Mirror", "https://huggingface.co/datasets/databricks/databricks-dolly-15k/raw/main/databricks-dolly-15k.jsonl"),
    ("Dolly GitHub Master", "https://raw.githubusercontent.com/databrickslabs/dolly/master/data/databricks-dolly-15k.jsonl"),
    ("Dolly GitHub Main Root", "https://raw.githubusercontent.com/databrickslabs/dolly/main/databricks-dolly-15k.jsonl"),
    ("PEP 8 Peps Folder", "https://raw.githubusercontent.com/python/peps/main/peps/pep-0008.rst"),
    ("PEP 20 Peps Folder", "https://raw.githubusercontent.com/python/peps/main/peps/pep-0020.rst"),
    ("PEP 8 Direct Root", "https://raw.githubusercontent.com/python/peps/main/pep-0008.rst"),
]

for name, url in urls_to_test:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read(200)
            print(f"[SUCCESS 200] {name}: {len(data)} bytes fetched from {url}")
    except Exception as e:
        print(f"[FAIL] {name}: {e} for {url}")
