import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

qa_file = "data_general/benchmark_raw/03_conversations_qa.txt"
if os.path.exists(qa_file) and os.path.getsize(qa_file) == 0:
    os.remove(qa_file)
    print(f"Removed 0-byte file: {qa_file}")

sys.argv = ["prepare_benchmark_data.py", "--smoke-test", "--output-dir", "data_general/benchmark_raw"]

from scratch.prepare_benchmark_data import main

if __name__ == "__main__":
    main()
