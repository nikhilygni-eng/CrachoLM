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

## Stop, save, and resume

Once the terminal says **Training ready**, press **Ctrl+C once** to stop. Training
finishes the current gradient-accumulation group and optimizer update, saves
`last.pt`, and exits. Wait until it prints:

```text
Checkpoint saved; training stopped safely.
```

The full checkpoint is roughly 3 GB, so allow the save to finish. Repeated Ctrl+C
does not interrupt that save. Forced termination (`kill -9`), a power cut, or a
full disk cannot produce a new checkpoint; the previous completed `last.pt`
remains available. `last.pt` is also refreshed every **100 successful optimizer
updates** and at each epoch end. Change the interval with `--checkpoint-every`.

For the story experiment above, resume with:

```bash
cd /home/escanor/Downloads/CrachoLM
.venv/bin/python -u train_250m.py resume \
  --checkpoint checkpoints_250m_story_experiment/last.pt \
  --output checkpoints_250m_story_resumed
```

The resume command reads the original settings and data paths from
`training_manifest.json` next to the checkpoint. Each resumed run uses a **new
output directory**, preserving the source checkpoint. Subsequent resumes should
point to the latest run's `last.pt` and another fresh output directory.

Checkpoints contain the model, optimizer, learning-rate schedule, AMP scaler,
CPU/CUDA random states, shuffled-data seed, next batch, and partial epoch loss
statistics. They resume within an epoch at a completed optimizer boundary.
Gradient buffers do not need saving because pending accumulation finishes before
the checkpoint is written. The original data, sequence length, accumulation,
seed, and planned total epoch count must match; completed schedules cannot be
extended with this resume command.

The new loop saves `last.pt` for latest progress and `best_model.pt` after improved
epoch validation. A mid-epoch checkpoint may have no validation score yet, or the
previous epoch's score. Original 70M files and the fresh 250M initialization are
not modified by this workflow. No substantive corpus training has been started
as part of installing stop/resume support.

The 2048-token tokenizer retains its existing whitespace-normalization limitation:
it does not preserve code indentation or paragraph formatting reliably. Increasing
parameter count does not address that limitation or guarantee better answers.

## Verification and implementation references

Tests cover SDPA/manual-attention agreement, causal masking, checkpointed
gradients with dropout, deterministic evaluation, parameter counting, preservation
of the original defaults, output-directory protection, and checkpoint resume.
Stop/resume tests send real SIGINT during accumulation and SIGTERM during
validation, compare resumed and uninterrupted weights, verify periodic saves,
and simulate a failed save to check that the previous checkpoint survives.
The prior chat, calculator, training-loss, and fluency checks were also run.
The actual 72.6M checkpoint was reloaded using the edited source and compared
with the backed-up implementation on a fixed input.

- [PyTorch SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/2.14/checkpoint.html)
- [PyTorch AdamW memory behavior](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)

The remote connection provides terminal and file operations. Antigravity and the
progress terminal can be opened this way, but interactive mouse control is not
available through this connection.

### Stop/resume verification, 28 September 2026

17 relevant CPU tests passed; the five stop/resume tests also passed after the GPU memory fix. CPU interrupt/resume tests matched uninterrupted weights exactly with dropout, shuffled data, padding, and gradient accumulation.

The full 250,431,488-parameter model passed actual SIGINT, checkpoint save, and CLI resume on the RTX 3050 with the default 512-token sequences and accumulation 16. The first run saved update 1 at batch 16; the resumed run saved update 2 at batch 32. Each complete checkpoint was 3,026,365,235 bytes. The test checkpoints were temporary and removed after verification. Details: runs/stop_resume_20260928_173346/gpu_default_settings.json.

The first resume test exposed an out-of-memory error with the autocast weight cache enabled during accumulation. The resumable training loop now disables that temporary cache. Saving/loading and the default configuration were retested successfully with the 70M chat server still running.
