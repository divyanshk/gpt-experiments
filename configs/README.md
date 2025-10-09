# GRPO Configuration System

Easy-to-tune hyperparameters using YAML files for GRPO training.

## Quick Start

### Use Default Configs

```bash
# Local mode (uses configs/grpo_local.yaml)
python training/grpo_training.py --model gpt2

# Cluster mode (uses configs/grpo_cluster.yaml)
python training/grpo_training.py --model gpt2 --mode cluster
```

### Use Custom Config

```bash
python training/grpo_training.py --model gpt2 --config configs/my_custom_config.yaml
```

## Configuration Files

### Default Configs

- **`grpo_local.yaml`**: Optimized for local development/testing
- **`grpo_cluster.yaml`**: Optimized for multi-GPU cluster training

### Config Structure

```yaml
mode: local  # or cluster
output_dir: ./grpo_results_local

# Training Hyperparameters
training:
  num_train_epochs: 1
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 5.0e-5
  warmup_steps: 0

# GRPO Specific Parameters
grpo:
  generation_batch_size: 8
  num_generations: 8
  beta: 0.01  # KL penalty coefficient
```

## Tuning Hyperparameters

### Method 1: Edit YAML Directly

1. Copy a default config:
   ```bash
   cp configs/grpo_cluster.yaml configs/grpo_cluster_experiment1.yaml
   ```

2. Edit parameters:
   ```yaml
   training:
     learning_rate: 1.0e-4  # Changed from 3.0e-5
     per_device_train_batch_size: 16  # Changed from 8
   
   grpo:
     beta: 0.05  # Changed from 0.01
   ```

3. Run with custom config:
   ```bash
   python training/grpo_training.py --model gpt2 --config configs/grpo_cluster_experiment1.yaml
   ```

### Method 2: Programmatic Updates

```python
from configs.grpo_config import GRPOConfigLoader

# Load config
loader = GRPOConfigLoader(mode="cluster")

# Update parameters
loader.update_param("training.learning_rate", 1e-4)
loader.update_param("grpo.beta", 0.05)

# Save as new config
loader.save_config("configs/grpo_cluster_experiment1.yaml")
```

## Tracking Experiments

### Version Control Configs

```bash
# Add to git
git add configs/grpo_cluster_experiment1.yaml
git commit -m "Experiment 1: Increased LR and beta"
```

### Naming Convention

Use descriptive names that capture the experiment:

- `grpo_cluster_high_lr.yaml` - High learning rate experiment
- `grpo_cluster_large_beta.yaml` - Large KL penalty
- `grpo_cluster_long_context.yaml` - Longer context training

### WandB Integration

Config is automatically logged to WandB:
- View in WandB dashboard under "Config" tab
- Compare hyperparameters across runs
- Track which configs produced best results

## Key Parameters to Tune

### Learning Rate (`training.learning_rate`)
- **Range**: 1e-6 to 5e-4
- **Lower**: More stable, slower convergence
- **Higher**: Faster learning, risk of instability

### Batch Size (`training.per_device_train_batch_size`)
- **Larger**: Better gradient estimates, requires more memory
- **Smaller**: Less memory, more updates
- **Note**: Effective batch size = `per_device_train_batch_size × gradient_accumulation_steps × num_gpus`

### KL Penalty (`grpo.beta`)
- **Range**: 0.001 to 0.1
- **Lower**: More exploration, may drift from reference model
- **Higher**: Stay closer to reference, less exploration
- **Zero**: Disables KL tracking

### Generation Parameters (`grpo.num_generations`)
- **Range**: 4 to 16
- **More**: Better policy gradient estimates, slower
- **Fewer**: Faster training, noisier estimates

## Example Experiments

### Experiment 1: Conservative Training
```yaml
training:
  learning_rate: 1.0e-5  # Low LR
grpo:
  beta: 0.05  # High KL penalty
```

### Experiment 2: Aggressive Training
```yaml
training:
  learning_rate: 1.0e-4  # High LR
grpo:
  beta: 0.005  # Low KL penalty
```

### Experiment 3: Large Batch
```yaml
training:
  per_device_train_batch_size: 16
  gradient_accumulation_steps: 4
  # Effective batch size: 64
```

## Tips

1. **Start with defaults**: Use provided configs as baseline
2. **Change one thing**: Modify single parameter per experiment
3. **Track everything**: Use git + WandB for tracking
4. **Document results**: Add comments to configs about outcomes
5. **Compare runs**: Use WandB to compare hyperparameter impact
