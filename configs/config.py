import torch

# Local development configuration - small model that fits on MacBook
LOCAL_CONFIG = {
    "vocab_size": 50257,     # GPT-2 vocab size
    "context_length": 256,   # Reduced from typical 1024 for memory
    "emb_dim": 384,         # Smaller embedding dimension
    "n_heads": 6,           # Fewer attention heads
    "n_layers": 6,          # Fewer transformer layers
    "drop_rate": 0.1,
    "qkv_bias": False
}

# Cluster configuration - larger model for production training
CLUSTER_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# Training configurations
LOCAL_TRAINING_CONFIG = {
    "batch_size": 4,
    "learning_rate": 5e-4,
    "num_epochs": 3,
    "warmup_steps": 100,
    "max_length": 256,
    "stride": 128,
    "eval_freq": 50,
    "eval_iter": 10,
    "device": "mps" if torch.backends.mps.is_available() else "cpu"
}

CLUSTER_TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 3e-4,
    "num_epochs": 10,
    "warmup_steps": 1000,
    "max_length": 1024,
    "stride": 512,
    "eval_freq": 500,
    "eval_iter": 100,
    "device": "cuda"
}

def get_config(mode="local"):
    """Get configuration for local or cluster deployment"""
    if mode == "local":
        return LOCAL_CONFIG, LOCAL_TRAINING_CONFIG
    elif mode == "cluster":
        return CLUSTER_CONFIG, CLUSTER_TRAINING_CONFIG
    else:
        raise ValueError("Mode must be 'local' or 'cluster'")