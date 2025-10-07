import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils.device_utils import get_best_device


class GPT2Loader:
    """Loader for pretrained GPT-2 models from HuggingFace"""

    def __init__(self, model_name="gpt2", mode="local", use_quantization=None):
        self.model_name = model_name
        self.mode = mode
        self.device = get_best_device()

        # Auto-determine quantization based on device and mode
        if use_quantization is None:
            # Use quantization for local mode on limited hardware
            self.use_quantization = mode == "local" and self.device in ["mps", "cpu"]
        else:
            self.use_quantization = use_quantization

        self.model = None
        self.tokenizer = None

    def load_model(self, use_lora=True):
        """Load GPT-2 model and tokenizer with optional optimizations"""
        print(f"Loading GPT-2 model: {self.model_name}")
        print(f"Device: {self.device}, Quantization: {self.use_quantization}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model with appropriate settings
        if self.use_quantization and self.device == "cuda":
            # Use 4-bit quantization for CUDA
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            # Standard loading
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
            )

            # Move to device if not using device_map
            if self.device != "cuda":
                self.model.to(self.device)

        # Apply LoRA for efficient fine-tuning if requested
        if use_lora and self.mode == "local":
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.1,
                target_modules=["c_attn", "c_proj"],
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            print("LoRA adapters applied for efficient fine-tuning")

        print(
            f"Model loaded successfully with {sum(p.numel() for p in self.model.parameters()):,} parameters"
        )
        return self.model, self.tokenizer

    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            return {"status": "Model not loaded"}

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "model_name": self.model_name,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.model.device),
            "dtype": str(self.model.dtype),
            "quantized": self.use_quantization,
            "mode": self.mode,
        }

    def generate_text(self, prompt, max_new_tokens=100, temperature=0.7, top_k=50, top_p=0.9):
        """Generate text using the loaded model"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode and return only new text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_text = generated_text[len(prompt) :]
        return new_text.strip()


def create_gpt2_pipeline(model_name="gpt2", mode="local", use_quantization=None):
    """Factory function to create GPT-2 pipeline"""
    print(f"Creating GPT-2 pipeline: {model_name} in {mode} mode")

    loader = GPT2Loader(model_name=model_name, mode=mode, use_quantization=use_quantization)

    return loader


def main():
    """Example usage of GPT-2 loader"""
    # Test different model sizes
    models_to_test = [
        "distilgpt2",  # ~82M params, very fast
        "gpt2",  # ~124M params, good balance
        "gpt2-medium",  # ~355M params, more capable
    ]

    for model_name in models_to_test:
        print(f"\n{'=' * 50}")
        print(f"Testing: {model_name}")
        print("=" * 50)

        try:
            # Create and load model
            loader = create_gpt2_pipeline(model_name=model_name, mode="local")
            model, tokenizer = loader.load_model()

            # Print model info
            info = loader.get_model_info()
            print(f"Parameters: {info['total_parameters']:,}")
            print(f"Device: {info['device']}")

            # Test generation
            prompt = "The future of artificial intelligence is"
            generated = loader.generate_text(prompt, max_new_tokens=50)
            print(f"Generated: {generated}")

        except Exception as e:
            print(f"Error with {model_name}: {e}")


if __name__ == "__main__":
    main()
