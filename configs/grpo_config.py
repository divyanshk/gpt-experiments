"""
GRPO Configuration Loader

Load GRPO hyperparameters from YAML files for easy tuning and tracking.
"""

from pathlib import Path
from typing import Dict, Any
import yaml
from trl import GRPOConfig


class GRPOConfigLoader:
    """Load and manage GRPO configurations from YAML files"""

    def __init__(self, config_path: str = None, mode: str = "local"):
        """
        Initialize config loader

        Args:
            config_path: Path to custom YAML config file
            mode: "local" or "cluster" (used if config_path not provided)
        """
        self.mode = mode

        if config_path is None:
            # Use default config based on mode
            config_dir = Path(__file__).parent
            config_path = config_dir / f"grpo_{mode}.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        return config

    def get_grpo_config(
        self, report_to: str = "none", device: str = "cuda", verbose: bool = True
    ) -> GRPOConfig:
        """
        Create GRPOConfig from loaded YAML configuration

        Args:
            report_to: Reporting backend ("wandb", "tensorboard", "none")
            device: Device for training
            verbose: Whether to show progress bars

        Returns:
            GRPOConfig instance
        """
        cfg = self.config

        # Determine fp16 based on device if not explicitly set
        use_fp16 = cfg["optimization"]["fp16"]
        if device != "cuda":
            use_fp16 = False

        # Determine log level and tqdm settings
        log_level = cfg["logging"]["log_level"] if verbose else "warning"
        disable_tqdm = not verbose

        grpo_config = GRPOConfig(
            output_dir=cfg["output_dir"],
            # Training params
            num_train_epochs=cfg["training"]["num_train_epochs"],
            per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
            learning_rate=cfg["training"]["learning_rate"],
            warmup_steps=cfg["training"]["warmup_steps"],
            # Logging
            logging_steps=cfg["logging"]["logging_steps"],
            save_steps=cfg["logging"]["save_steps"],
            log_level=log_level,
            disable_tqdm=disable_tqdm,
            report_to=report_to,
            # Data loading
            dataloader_num_workers=cfg["data"]["dataloader_num_workers"],
            remove_unused_columns=cfg["data"]["remove_unused_columns"],
            # Optimization
            fp16=use_fp16,
            gradient_checkpointing=cfg["optimization"]["gradient_checkpointing"],
            # GRPO specific
            generation_batch_size=cfg["grpo"]["generation_batch_size"],
            num_generations=cfg["grpo"]["num_generations"],
            beta=cfg["grpo"]["beta"],
        )

        return grpo_config

    def get_lora_config(self) -> Dict[str, Any]:
        """Get LoRA configuration from YAML"""
        return self.config.get("lora", {})

    def update_param(self, key_path: str, value: Any):
        """
        Update a specific parameter in the config

        Args:
            key_path: Dot-separated path (e.g., "training.learning_rate")
            value: New value
        """
        keys = key_path.split(".")
        current = self.config

        # Navigate to the parent dict
        for key in keys[:-1]:
            current = current[key]

        # Update the value
        current[keys[-1]] = value

    def save_config(self, output_path: str = None):
        """
        Save current configuration to YAML file

        Args:
            output_path: Path to save config (defaults to original path)
        """
        if output_path is None:
            output_path = self.config_path

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

    def get_config_dict(self) -> Dict[str, Any]:
        """Get the full configuration dictionary"""
        return self.config

    def print_config(self):
        """Print the configuration in a readable format"""
        print(f"\n{'='*60}")
        print(f"GRPO Configuration ({self.mode} mode)")
        print(f"Config file: {self.config_path}")
        print(f"{'='*60}\n")

        def print_dict(d, indent=0):
            for key, value in d.items():
                if isinstance(value, dict):
                    print("  " * indent + f"{key}:")
                    print_dict(value, indent + 1)
                else:
                    print("  " * indent + f"{key}: {value}")

        print_dict(self.config)
        print()


def load_grpo_config(
    config_path: str = None, mode: str = "local", **kwargs
) -> GRPOConfig:
    """
    Convenience function to load GRPO config

    Args:
        config_path: Path to custom config file
        mode: "local" or "cluster"
        **kwargs: Additional arguments for get_grpo_config()

    Returns:
        GRPOConfig instance
    """
    loader = GRPOConfigLoader(config_path=config_path, mode=mode)
    return loader.get_grpo_config(**kwargs)


if __name__ == "__main__":
    # Example usage
    print("Loading local config:")
    loader = GRPOConfigLoader(mode="local")
    loader.print_config()

    print("\nLoading cluster config:")
    loader = GRPOConfigLoader(mode="cluster")
    loader.print_config()

    # Example: Update a parameter
    loader.update_param("training.learning_rate", 1e-4)
    loader.save_config("configs/grpo_cluster_modified.yaml")
    print("Modified config saved!")
