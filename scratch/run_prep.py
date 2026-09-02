import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = ["prepare_benchmark_data.py", "--smoke-test", "--output-dir", "data_general/benchmark_raw"]

from scratch.prepare_benchmark_data import main

if __name__ == "__main__":
    main()
