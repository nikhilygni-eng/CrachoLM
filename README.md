#  CrachoLM-0.1: Custom Decoder-Only Transformer Language Model

Welcome to **CrachoLM**, an educational yet real decoder-only Transformer language model built completely **from scratch** using Python and PyTorch.

---

##  Project Philosophy & Strict Rules

*  **No API Keys**: 100% offline, private, and local.
*  **No Ollama / External Runtimes**: Pure native Python and PyTorch.
*  **No Pretrained Weights**: Every weight matrix is initialized randomly ($N(0, \sigma^2)$).
*  **No Existing LLM Codebases**: Built from raw mathematical equations.
*  **Hardware Acceleration**: Automatic CUDA GPU acceleration (RTX 3050 detected), with seamless fallback to CPU.

---

##  Project Directory Structure

```text
CrachoLM/
├── data/
│   ├── raw/                        # Raw text datasets
│   │   ├── sample_corpus.txt       # Small test corpus (Phase 1)
│   │   └── tinyshakespeare.txt     # Public domain 1M char corpus (Phase 7, auto-downloaded)
│   ├── processed/                  # Processed token cache
│   └── prepare_tinyshakespeare.py  # Dataset downloader script
├── checkpoints/                    # Saved model weights & tokenizer.json
├── logs/                           # Training logs
├── src/                            # Core Python package
│   ├── __init__.py
│   ├── device.py                   # GPU/CPU auto-detection
│   ├── tokenizer.py                # Character-level tokenizer from scratch
│   ├── model.py                    # Decoder-only Transformer architecture
│   ├── data.py                     # Data cleaning, Dataset & DataLoaders
│   ├── trainer.py                  # Training loop, AMP, grad accum, checkpoints
│   └── generator.py                # Autoregressive text generation
├── tests/                          # Test suite
│   ├── test_tokenizer.py
│   ├── test_model.py
│   ├── test_data.py
│   ├── test_train.py
│   └── test_generator.py
├── config.py                       # All hyperparameters
├── check_system.py                 # Hardware verification
├── inspect_model.py                # Phase 7 baseline inspection
├── train.py                        # Main training CLI script
├── generate.py                     # Main generation CLI script
└── README.md
```

---

##  Quick Start Guide

### 1. Create & Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies (CUDA GPU on Linux)
```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3. Verify Hardware
```bash
python3 check_system.py
```

---

##  Running the Project

### Inspect Baseline Model Metrics (Phase 7)
```bash
python3 inspect_model.py
```

### Train Model from Scratch
```bash
# Default run (auto-downloads TinyShakespeare dataset)
python3 train.py --epochs 10 --batch-size 16

# With gradient accumulation (effective batch = 16 × 4 = 64)
python3 train.py --epochs 10 --batch-size 16 --grad-accum 4

# Resume from checkpoint
python3 train.py --resume checkpoints/best_model.pt

# Train on custom text dataset
python3 train.py --data-path data/raw/my_corpus.txt --epochs 10
```

### Generate Text
```bash
# Default stochastic sampling
python3 generate.py --prompt "First Citizen:"

# Creative generation
python3 generate.py --prompt "HAMLET:" --temperature 0.9 --top-k 40 --max-new-tokens 200

# Deterministic greedy decoding
python3 generate.py --prompt "To be or not" --greedy --max-new-tokens 100
```

---

##  Configuration Overview (`config.py`)

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `d_model` | `256` | Hidden embedding dimension |
| `n_layers` | `6` | Transformer decoder layers |
| `n_heads` | `8` | Multi-Head Attention heads |
| `d_ff` | `1024` | Feed-Forward expansion dim |
| `max_seq_len` | `512` | Context window size |
| `batch_size` | `16` | Micro-batch size |
| `grad_accum_steps` | `4` | Gradient accumulation (effective batch = 64) |
| `learning_rate` | `5e-4` | Peak learning rate |
| `warmup_steps` | `100` | LR warmup steps |

---

##  Completed Project Roadmap

- [x] **Phase 1**: Project setup, GPU/CPU detection, configuration
- [x] **Phase 2**: Custom character-level tokenizer from scratch
- [x] **Phase 3**: Decoder-only Transformer architecture from scratch
- [x] **Phase 4**: Data preparation pipeline & next-token target shifting
- [x] **Phase 5**: Training loop with AMP, checkpoints & OOM guard
- [x] **Phase 6**: Autoregressive text generation (Temperature, Top-K & Greedy)
- [x] **Phase 7**: Systematic model analysis, real dataset, gradient accumulation & LR warmup
- [x] **Phase 8**: Honest evaluation system with exact-match accuracy, per-category metrics, JSON/CSV results

---

##  Evaluate a Trained Model (Phase 8)

```bash
# Run evaluation against held-out eval questions (never seen during training)
python3 evaluate.py

# Use a specific checkpoint
python3 evaluate.py --checkpoint checkpoints/best_model.pt

# Adjust sampling during evaluation (lower = more deterministic)
python3 evaluate.py --checkpoint checkpoints/best_model.pt --temperature 0.1 --top-k 10
```

Results are saved automatically to:
- `logs/eval_results_<timestamp>.json`
- `logs/eval_results_<timestamp>.csv`

>  **Important**: Evaluation data lives exclusively in `data/eval/` and is **never** passed to `train.py`. See `data/eval/README.txt` for the data leakage policy.

---

*CrachoLM is developed step-by-step for complete insight into decoder-only Transformer mechanics.*
