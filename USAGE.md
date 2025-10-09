# GRPO Training Pipeline Usage

## Supported Models

The pipeline now supports multiple model architectures:

### GPT-2 Models (Small to Large)
- `gpt2` (124M params)
- `gpt2-medium` (355M params)
- `gpt2-large` (774M params)
- `gpt2-xl` (1.5B params)
- `distilgpt2` (82M params)

### GPT-OSS Models (Large Scale)
- `openai/gpt-oss-20b` (20B params)

## Basic Usage

### Train GPT-2 (default)
```bash
python training/grpo_training.py
```

### Train GPT-2 Medium with LoRA
```bash
python training/grpo_training.py --model gpt2-medium --use-lora
```

### Train GPT-OSS-20B (recommended with LoRA)
```bash
python training/grpo_training.py --model openai/gpt-oss-20b --use-lora --mode cluster
```

### Train in Quiet Mode (no verbose logs)
```bash
python training/grpo_training.py --quiet
```

### Train without WandB
```bash
python training/grpo_training.py --no-wandb
```

## All Options

```bash
python training/grpo_training.py \
  --model gpt2 \              # Model name
  --mode local \              # Training mode: local or cluster
  --max-samples 100 \         # Limit training samples
  --use-lora \                # Enable LoRA (recommended for large models)
  --no-wandb \                # Disable WandB logging
  --quiet                     # Disable verbose training logs
```

## Features

### Profiling
- **WandB**: System metrics, rewards, KL divergence, generation lengths
- **PyTorch Profiler**: GPU kernel-level profiling (CUDA only)
  - Results: `./profiling_results_{mode}/`
  - View: `tensorboard --logdir=./profiling_results_{mode}/`
  - Chrome trace: `./profiling_results_{mode}/trace.json`

### Tracking Metrics
- KL divergence (per step, with beta=0.01)
- Reward scores (mean & std)
- Generation lengths (mean, max, min)
- Entropy
- Clip ratios

### Outputs
- **Models**: `./grpo_final_model_{mode}/`
- **Logs**: `./training_logs_{mode}/training_metrics.csv`
- **Profiling**: `./profiling_results_{mode}/`
- **WandB Dashboard**: https://wandb.ai/your-username/grpo-post-training

## Architecture Support

The pipeline automatically detects model architecture and applies appropriate LoRA targets:
- **GPT-2**: `c_attn`, `c_proj`
- **GPT-OSS**: `q_proj`, `v_proj`, `k_proj`, `o_proj`

