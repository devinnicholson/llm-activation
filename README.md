# LLM Activation Steering From Scratch

I built and trained a TinyStories language model stack from scratch on an HPC cluster, then added tools to steer the model's behavior from inside its activations.

[Open the Colab demo](https://colab.research.google.com/github/devinnicholson/llm-activation/blob/main/notebooks/playful_steering_colab.ipynb)

## What This Is

This repo contains the code for an end-to-end LLM training system.

- byte-level BPE tokenizer
- decoder-only Transformer blocks
- multi-head self-attention
- RoPE positional embeddings
- SwiGLU MLP
- RMSNorm
- training and checkpointing loop
- text generation loop
- activation vector extraction
- activation steering sweeps and reports
- SLURM scripts for GPU/HPC runs
- Colab notebook for an interactive steering demo

The current demo model is a TinyStories Transformer with roughly 107M parameters.

| Setting | Value |
| --- | --- |
| Layers | 10 |
| Attention heads | 12 |
| Hidden size | 768 |
| MLP size | 3072 |
| Vocabulary | 8192 tokens |
| Context length | 512 tokens |
| Parameters | 106,970,880 |
| Best validation loss | 0.9767 |

The model was trained from random initialization. The Colab demo loads an exported inference bundle from Google Drive and lets you change the prompt, steering layer, alpha, and steering position.

## Current Demo

The most reliable steering setting from the ctx512 run is below.

```text
emotion = playful
layer = 4
alpha = 1.5
position = all
prompt = Once upon a time there was a little robot
```

Baseline generation stays closer to the original TinyStories continuation. The steered generation shifts toward a more playful and silly continuation while the prompt stays the same.

The steering vector is added directly inside the model during generation.

## Google Colab Demo

GitHub stores the code and notebook. Google Drive stores the exported model bundle.

Expected Drive layout

```text
MyDrive/llm-activation-colab/playful_ctx512/
  model.pt
  tokenizer.json
  vectors.pt
  manifest.json
```

Open the notebook

```text
notebooks/playful_steering_colab.ipynb
```

Use a GPU runtime in Colab

```text
Runtime -> Change runtime type -> T4 GPU
```

The notebook clones this repo from

```text
https://github.com/devinnicholson/llm-activation.git
```

Then it loads the Drive bundle and runs baseline and steered generation.

## Reproducing The Main Pipeline

Install locally

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the tiny smoke path

```bash
python scripts/00_train_tokenizer.py --config configs/tiny.yaml
python scripts/01_prepare_dataset.py --config configs/tiny.yaml
python scripts/02_train.py --config configs/tiny.yaml
python scripts/03_generate.py --config configs/tiny.yaml --prompt "Once upon a time"
```

Cluster scripts live in `slurm/`. The main ctx512 training config is below.

```text
configs/tinystories_100m_full_ctx512.yaml
```

Build playful and serious steering vectors

```bash
python scripts/05_build_emotion_vectors.py \
  --config configs/tinystories_100m_full_ctx512.yaml \
  --checkpoint checkpoints/tinystories_100m_full_ctx512/best.pt \
  --prompt-bank prompt_banks/playful_vs_serious.yaml \
  --output benchmarks/results/playful_serious_vectors_100m_full_ctx512.pt
```

Run steering

```bash
python scripts/06_steer_generation.py \
  --config configs/tinystories_100m_full_ctx512.yaml \
  --checkpoint checkpoints/tinystories_100m_full_ctx512/best.pt \
  --vectors benchmarks/results/playful_serious_vectors_100m_full_ctx512.pt \
  --emotion playful \
  --layer 4 \
  --alpha 1.5 \
  --position all \
  --prompt "Once upon a time there was a little robot"
```

## Repository Layout

```text
configs/        model and training configs
scripts/        tokenizer, data prep, training, generation, steering, export
src/            project-owned Python package
native/         Rust/PyO3 tokenizer backend
prompt_banks/   contrastive prompts for activation vectors
slurm/          cluster job scripts
notebooks/      Colab demo
tests/          smoke and correctness tests
```

## Project Framing

The project uses a Google Colab plus NVIDIA CUDA backed HPC workflow.

- training and vector extraction run on the cluster GPU environment
- exported model bundles move to Google Drive
- Colab provides an interactive demo anyone can run

I am framing this as a focused activation steering result. The model was trained from scratch on TinyStories, and a learned direction in its hidden states can push generations toward a more playful narrative style.
