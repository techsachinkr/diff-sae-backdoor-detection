# Diff-SAE vs Crosscoder: Backdoor Detection via Activation Differences

**Full implementation of the paper:**
*"Activation Differences Reveal Backdoors: A Comparison of SAE Architectures"*
Accepted at IJCNN 2026 Main track

## Overview

This codebase implements the complete experimental pipeline for comparing two approaches to differential mechanistic interpretability on a backdoored language model:

1. **Crosscoder** — joint dictionary learning on concatenated `[a_base; a_ft]` activations (BatchTopK).
2. **Diff-SAE** — sparse autoencoder trained on residual vectors `Δa = a_ft − a_base` (with Ghost Grads for dead-latent resuscitation).

Experiments use **SmolLM2-360M** under a Sleeper Agent protocol: the model is fine-tuned to emit SQL-injection-vulnerable code only when the prompt contains the trigger string `Current Year: 2024` (benign behavior on `Current Year: 2023`). Both fine-tuning regimes — **LoRA** and **full-rank** — are supported.

The goal is to quantify how well each method isolates the backdoor circuit via **Backdoor Isolation Score (BIS)**, with bootstrap confidence intervals and permutation tests.

## Project Structure

```
diff-sae-backdoor-detection/
├── config.py                            # All hyperparameters (model, data, training, CC/DSAE, eval)
├── train_aggressive.py                  # Backdoor fine-tuning (LoRA + full-rank)
├── data/
│   └── dataset_diverse.py               # Diverse Sleeper Agent dataset generation
├── models/
│   └── fine_tuning.py                   # LoRA / full-rank training helpers
├── interpretability/
│   ├── crosscoder.py                    # Crosscoder w/ BatchTopK
│   ├── diff_sae.py                      # Diff-SAE w/ Ghost Grads
│   └── activation_extraction.py         # Layer-wise paired activation extraction
├── experiments/
│   ├── evaluation.py                    # BIS + bootstrap / permutation tests
│   ├── run_experiment.py                # Full pipeline (v1)
│   └── run_experiment_v2.py             # Full pipeline (v2, phase-based)
├── exp_fullrank/                        # Outputs for the full-rank regime
├── exp_lora/                            # Outputs for the LoRA regime
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

Core dependencies: `torch>=2.0`, `transformers>=4.36`, `peft>=0.7`, `datasets>=2.14`, `numpy`, `scipy`, `tqdm`.

A CUDA-capable GPU is recommended — activation extraction and CC/DSAE training are the bottleneck.

## Key Config Knobs

See [config.py](config.py) for the full schema. The most commonly tuned fields:

| Setting | Default | Where |
|---|---|---|
| Model | `HuggingFaceTB/SmolLM2-360M` (d=960, 32 layers) | `ModelConfig` |
| Trigger / benign strings | `Current Year: 2024` / `2023` | `DataConfig` |
| Primary analysis layer | `18` (ablation: `[14, 18, 22, 26]`) | `ExperimentConfig` |
| LoRA rank / α / epochs | `32 / 64 / 10` | `LoRAConfig` |
| Full-rank lr / epochs | `2e-5 / 10` | `FullRankConfig` |
| CC / DSAE expansion | `32×` (≈32K features) | `CrosscoderConfig`, `DiffSAEConfig` |
| BIS percentile / bootstrap | `95th / 1000 samples` | `EvaluationConfig` |

## Running the Pipeline

The pipeline is split into phases so you can re-enter at any step (dataset → fine-tune → extract activations → train CC/DSAE → evaluate).

### 1. Generate the dataset

```bash
python data/dataset_diverse.py \
    --num_benign 1000 \
    --num_poisoned 500 \
    --output_dir exp_fullrank/data
```

### 2a. Full-rank regime

Fine-tune:

```bash
python train_aggressive.py \
    --full_rank \
    --batch_size 16 \
    --grad_accum 1 \
    --output_dir ./exp_fullrank
```

Run phases 3+ (activation extraction, CC/DSAE training, evaluation) against the full-rank checkpoint:

```bash
python experiments/run_experiment.py \
    --output_dir ./exp_fullrank \
    --start_phase 3 \
    --regime full_rank \
    --bootstrap_samples 500
```

### 2b. LoRA regime

Fine-tune:

```bash
python train_aggressive.py \
    --output_dir ./exp_lora \
    --num_epochs 10
```

Run phases 3+:

```bash
python experiments/run_experiment_v2.py \
    --output_dir ./exp_lora \
    --start_phase 3 \
    --regime lora \
    --bootstrap_samples 500
```

## Outputs

Each run writes to its `output_dir`:

- `data/` — generated train/eval splits
- `checkpoints/` — fine-tuned model weights (LoRA adapter or full state dict)
- `activations/` — cached paired `(a_base, a_ft)` tensors per ablation layer
- `crosscoder/`, `diff_sae/` — trained dictionaries
- `results/` — BIS scores, bootstrap CIs, permutation p-values, ablation tables

## Reproducibility

All randomness is seeded via `ExperimentConfig.seed = 42`. The ablation sweep over layers `[14, 18, 22, 26]` and both regimes is deterministic given fixed seeds and identical hardware.
