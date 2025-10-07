import torch


def get_best_device():
    """
    Automatically detect and return the best available device for the current machine.
    Priority: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        device = "cuda"
        print(f"🚀 Using CUDA with {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("🍎 Using Apple Metal Performance Shaders (MPS)")
    else:
        device = "cpu"
        print("💻 Using CPU")

    return device


def get_device_config(mode="local"):
    """
    Get device configuration based on available hardware and mode.

    Args:
        mode: "local" or "cluster"

    Returns:
        dict: Configuration for the current hardware setup
    """
    device = get_best_device()

    if mode == "local":
        # Local mode: Use whatever hardware is available locally
        if device == "cuda":
            # Local GPU setup
            gpu_count = torch.cuda.device_count()
            total_memory = sum(
                torch.cuda.get_device_properties(i).total_memory for i in range(gpu_count)
            ) // (1024**3)  # GB

            # Adjust memory allocation based on available VRAM
            if total_memory >= 40:  # High-end local setup
                config = {
                    "use_quantization": False,
                    "device": "auto",
                    "max_memory": {
                        str(
                            i
                        ): f"{torch.cuda.get_device_properties(i).total_memory // (1024**3) - 2}GB"
                        for i in range(gpu_count)
                    },
                    "generation_config": {
                        "max_new_tokens": 512,
                        "temperature": 0.7,
                        "top_k": 50,
                        "top_p": 0.9,
                    },
                }
            elif total_memory >= 16:  # Mid-range local GPU
                config = {
                    "use_quantization": True,
                    "device": "auto",
                    "max_memory": {
                        str(
                            i
                        ): f"{torch.cuda.get_device_properties(i).total_memory // (1024**3) - 2}GB"
                        for i in range(gpu_count)
                    },
                    "generation_config": {
                        "max_new_tokens": 256,
                        "temperature": 0.7,
                        "top_k": 50,
                        "top_p": 0.9,
                    },
                }
            else:  # Lower VRAM
                config = {
                    "use_quantization": True,
                    "device": "auto",
                    "max_memory": {
                        str(
                            i
                        ): f"{max(4, torch.cuda.get_device_properties(i).total_memory // (1024**3) - 2)}GB"
                        for i in range(gpu_count)
                    },
                    "generation_config": {
                        "max_new_tokens": 128,
                        "temperature": 0.7,
                        "top_k": 50,
                        "top_p": 0.9,
                    },
                }

        elif device == "mps":
            # Apple Silicon
            config = {
                "use_quantization": True,
                "device": "mps",
                "max_memory": None,  # MPS handles memory automatically
                "generation_config": {
                    "max_new_tokens": 128,
                    "temperature": 0.7,
                    "top_k": 50,
                    "top_p": 0.9,
                },
            }

        else:  # CPU
            config = {
                "use_quantization": True,
                "device": "cpu",
                "max_memory": None,
                "generation_config": {
                    "max_new_tokens": 64,
                    "temperature": 0.7,
                    "top_k": 20,
                    "top_p": 0.9,
                },
            }

    else:  # cluster mode
        # Cluster mode: Assume high-end multi-GPU setup
        if device == "cuda":
            gpu_count = torch.cuda.device_count()
            config = {
                "use_quantization": False,  # Full precision on cluster
                "device": "auto",
                "max_memory": {str(i): "40GB" for i in range(gpu_count)},
                "generation_config": {
                    "max_new_tokens": 512,
                    "temperature": 0.7,
                    "top_k": 50,
                    "top_p": 0.9,
                },
            }
        else:
            # Fallback for cluster without CUDA
            config = {
                "use_quantization": True,
                "device": device,
                "max_memory": None,
                "generation_config": {
                    "max_new_tokens": 256,
                    "temperature": 0.7,
                    "top_k": 50,
                    "top_p": 0.9,
                },
            }

    print(f"📋 Device config for {mode} mode:")
    print(f"   Device: {config['device']}")
    print(f"   Quantization: {config['use_quantization']}")
    if config["max_memory"]:
        print(f"   Memory allocation: {config['max_memory']}")

    return config
