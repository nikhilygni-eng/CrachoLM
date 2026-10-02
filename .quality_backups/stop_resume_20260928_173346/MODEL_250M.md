# CrachoLM 250M — separate PyTorch model

Created 28 September 2026. **This checkpoint is randomly initialized and untrained.**
The GPU test checked that training operations fit; it did not teach language or
improve sentence quality. The existing 72.6M model still serves the chat app.

| Setting | Existing chat model | New model |
|---|---:|---:|
| Trainable parameters, counting tied weights once | 72,580,608 | 250,431,488 |
| Transformer layers | 10 | 20 |
| Hidden width | 768 | 1024 |
| Attention heads | 12 | 16 |
| Feed-forward width | 3072 | 4000 |
| Maximum context | 256 | 512 |
| BPE vocabulary | 2048 | 2048 |
| Status | Trained, with substantial known limitations | Fresh weights; needs training |

The new size is approximately **250M total**, not 250M added to the old model.
Both use your own decoder-only Transformer implementation in PyTorch. No external
pretrained model, API model, or Ollama model was substituted. The existing BPE
tokenizer was copied into the new folder so its token IDs stay consistent.

## Files on this laptop

Project: `/home/escanor/Downloads/CrachoLM`

- `config_250m.py`: independent architecture and training preset.
- `train_250m.py`: initialize, GPU probe, and train commands.
- `checkpoints_250m_initial_20260928/initial_model.pt`: actual fresh weights.
- `checkpoints_250m_initial_20260928/model_info.json`: exact count, configuration,
  tokenizer fingerprint, and checkpoint SHA-256.
- `checkpoints_250m_initial_20260928/tokenizer_bpe.json`: paired tokenizer.
- `runs/scale_250m_20260928_171813/`: progress, GPU result, compatibility and
  preservation reports, and test logs.

The original default configuration and inference checkpoint remain selected.
All 93 pre-existing files in checkpoint and `backup_70m` directories retained
their sizes and modification timestamps; the current chat and original general-v3
best checkpoints also passed SHA-256 checks. The new initialization's SHA-256
was unchanged by the GPU probe. Backups of the two edited shared source files are
in `.quality_backups/scale_250m_20260928_171813/`.

## GPU check

Installed PyTorch: **2.13.0+cu130**, using the RTX 3050 Laptop 6GB GPU.
No PyTorch reinstall was necessary.

The probe completed **two successful AdamW updates**, including forward and
backward passes, at micro-batch 1 and sequence length 512. It used FP16 autocast,
FP32 weights/optimizer states, activation checkpointing, PyTorch SDPA, and
`AdamW(foreach=False)`.

- Peak PyTorch allocated GPU memory: **3966.96 MiB (3.87 GiB)**.
- Peak PyTorch reserved GPU memory: **4100 MiB (4.00 GiB)**.
- These numbers exclude the desktop, other applications, and CUDA overhead.
- The 70M chat server was running during this successful check.
- The probe used random token IDs and discarded its updated weights.
- A two-step probe is not a sustained-speed or model-quality benchmark.

Training defaults: micro-batch 1, gradient accumulation 16, sequence length 512,
AdamW learning rate 0.0003 with warmup/cosine decay, and two CPU threads.
Memory-saving options are off by default for the existing model and on for 250M.

## Commands

Open a terminal in the project:

```bash
cd /home/escanor/Downloads/CrachoLM
.venv/bin/python train_250m.py --help
```

The initial checkpoint has already been created. `init` refuses to overwrite an
existing directory. For another independent initialization, choose a new path:

```bash
.venv/bin/python train_250m.py init --output checkpoints_250m_another_initialization
```

Real pretraining requires separate training and held-out validation text. This
example uses the existing small TinyStories sample for a pipeline experiment;
that sample alone is insufficient for the general assistant you want:

```bash
.venv/bin/python -u train_250m.py train \
  --train-text data_general/fluency_stories/train.txt \
  --validation-text data_general/fluency_stories/validation.txt \
  --output checkpoints_250m_story_experiment \
  --epochs 1
```

**This training command has not been started.** Replace the text paths with an
appropriately prepared larger corpus for substantive pretraining. Audit corpus
quality, provenance, coverage, duplicates, and split leakage before a long run.
Instruction training and evaluation on unfamiliar prompts remain separate work.

Training saves the paired tokenizer, a data/configuration manifest, and atomic
epoch checkpoints. Every new run requires a fresh output directory. `--resume`
accepts a training checkpoint and continues at an epoch boundary, restoring the
optimizer, schedule, scaler, and PyTorch random state. It requires the original
data, sequence length, accumulation, and originally planned total epoch count;
it does not resume halfway through an epoch or extend a completed schedule.

The 2048-token tokenizer retains its existing whitespace-normalization limitation:
it does not preserve code indentation or paragraph formatting reliably. Increasing
parameter count does not address that limitation or guarantee better answers.

## Verification and implementation references

Tests cover SDPA/manual-attention agreement, causal masking, checkpointed
gradients with dropout, deterministic evaluation, parameter counting, preservation
of the original defaults, output-directory protection, and epoch checkpoint resume.
The prior chat, calculator, training-loss, and fluency checks were also run.
The actual 72.6M checkpoint was reloaded using the edited source and compared
with the backed-up implementation on a fixed input.

- [PyTorch SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/2.14/checkpoint.html)
- [PyTorch AdamW memory behavior](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)

The remote connection provides terminal and file operations. Antigravity and the
progress terminal can be opened this way, but interactive mouse control is not
available through this connection.
