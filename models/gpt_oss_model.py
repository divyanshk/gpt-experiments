import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import tiktoken
from pathlib import Path
import json


class GPTOSSLoader:
    """Loader for OpenAI GPT-OSS-20B model with memory optimization"""
    
    def __init__(self, model_id="openai/gpt-oss-20b", device="auto", use_quantization=True):
        self.model_id = model_id
        self.device = device
        self.use_quantization = use_quantization
        self.model = None
        self.tokenizer = None
        
    def load_model(self, cache_dir=None):
        """Load GPT-OSS model with memory optimizations"""
        print(f"Loading {self.model_id}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization for memory efficiency
        if self.use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                quantization_config=quantization_config,
                device_map=self.device,
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map=self.device,
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        
        print(f"Model loaded successfully on device: {self.model.device}")
        return self.model, self.tokenizer
    
    def generate_text(self, prompt, max_new_tokens=256, temperature=0.7, top_k=50, top_p=0.9):
        """Generate text using the loaded model"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move to model device
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
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Return only the newly generated part
        new_text = generated_text[len(prompt):]
        return new_text.strip()
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            return {"status": "Model not loaded"}
        
        # Calculate approximate model size
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_id": self.model_id,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.model.device),
            "dtype": str(self.model.dtype),
            "quantized": self.use_quantization
        }
    
    def save_model_locally(self, save_path="./gpt_oss_local"):
        """Save model locally for faster loading"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        save_dir = Path(save_path)
        save_dir.mkdir(exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # Save configuration
        config = {
            "original_model_id": self.model_id,
            "quantized": self.use_quantization,
            "save_timestamp": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        }
        
        with open(save_dir / "model_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"Model saved to {save_dir}")
        return save_dir


class GPTOSSClusterConfig:
    """Configuration for running GPT-OSS on different hardware"""
    
    @staticmethod
    def get_local_config():
        """Configuration for local MacBook development"""
        print(f"Warning: {model_id} might not work in local mode")
        return {
            "use_quantization": True,
            "device": "mps" if torch.backends.mps.is_available() else "cpu",
            "max_memory": {"0": "8GB"} if torch.cuda.is_available() else None,
            "generation_config": {
                "max_new_tokens": 128,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.9
            }
        }
    
    @staticmethod
    def get_cluster_config(num_gpus=4):
        """Configuration for GPU cluster deployment"""
        return {
            "use_quantization": False,  # Full precision on cluster
            "device": "auto",
            "max_memory": {str(i): "40GB" for i in range(num_gpus)},
            "generation_config": {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.9
            }
        }


def create_gpt_oss_pipeline(mode="local", model_id="openai/gpt-oss-20b"):
    """Factory function to create GPT-OSS pipeline based on mode"""
    
    if mode == "local":
        print(f"Warning: {model_id} might not work in local mode")
        config = GPTOSSClusterConfig.get_local_config()
        loader = GPTOSSLoader(
            model_id=model_id,
            device=config["device"],
            use_quantization=config["use_quantization"]
        )
    elif mode == "cluster":
        config = GPTOSSClusterConfig.get_cluster_config()
        loader = GPTOSSLoader(
            model_id=model_id,
            device=config["device"],
            use_quantization=config["use_quantization"]
        )
    else:
        raise ValueError("Mode must be 'local' or 'cluster'")
    
    return loader, config