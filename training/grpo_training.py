import sys
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig
)
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model # Imported but not utilized (yet) 
import json
import os
import tiktoken
from utils.training_tracker import GRPOTrainingTracker, ProgressMonitor
from utils.device_utils import get_best_device
from models.gpt2_loader import create_gpt2_pipeline
import wandb


class GRPOPostTrainingPipeline:
    def __init__(self, model_name="gpt2", mode="local", use_wandb=True):
        self.model_name = model_name
        self.mode = mode
        self.use_wandb = use_wandb
        self.device = get_best_device()
        
        # Initialize wandb if enabled
        if self.use_wandb:
            model_display_name = model_name.split('/')[-1]
            wandb.init(
                project="grpo-post-training",
                name=f"{mode}-{model_display_name}",
                config={
                    "model_name": model_name,
                    "mode": mode,
                    "device": self.device
                }
            )
        
        # Load model and tokenizer
        if self.model_name == 'gpt2':
            self._load_gpt2_model() # initializes self.model, self.tokenizer
    
    def _load_gpt2_model(self):
        """Load GPT-2 model using the new loader"""
        # Create GPT-2 loader
        gpt2_loader = create_gpt2_pipeline(
            model_name=self.model_name,
            mode=self.mode
        )
        
        # Load model and tokenizer
        self.model, self.tokenizer = gpt2_loader.load_model()
        
        # Print model info
        info = gpt2_loader.get_model_info()
        print(f"GPT-2 loaded: {info['total_parameters']:,} parameters")
        print(f"Device: {info['device']}, Quantized: {info['quantized']}")
    
    def load_and_prepare_dataset(self, dataset_name="knoveleng/open-rs", max_samples=None):
        """Load and prepare the Open-RS dataset for GRPO training"""
        # Load dataset
        dataset = load_dataset(dataset_name)
        
        if max_samples and self.mode == "local":
            # Limit dataset size for local development
            dataset["train"] = dataset["train"].select(range(min(max_samples, len(dataset["train"]))))
            if "test" in dataset:
                dataset["test"] = dataset["test"].select(range(min(max_samples//10, len(dataset["test"]))))
        
        def format_example(example):
            """Format examples for GRPO training with safety checks"""
            # Open-RS dataset has 'problem' and 'solution' fields
            if "problem" in example and "solution" in example:
                prompt = str(example["problem"]).strip()
                completion = str(example["solution"]).strip()
            elif "prompt" in example and "response" in example:
                prompt = str(example["prompt"]).strip()
                completion = str(example["response"]).strip()
            elif "text" in example:
                # If it's just text, create prompt-completion pairs
                text = str(example["text"]).strip()
                if len(text) > 100:
                    split_point = len(text) // 2
                    prompt = text[:split_point].strip()
                    completion = text[split_point:].strip()
                else:
                    prompt = "Complete the following: "
                    completion = text
            else:
                # Default formatting
                prompt = str(example.get("input", "")).strip()
                completion = str(example.get("output", "")).strip()
            
            # Safety checks - ensure non-empty prompts and completions
            if not prompt:
                prompt = "Generate text:"
            if not completion:
                completion = "No content available."
                
            # Ensure minimum length for tokenization
            if len(prompt) < 3:
                prompt = f"Task: {prompt}"
            if len(completion) < 3:
                completion = f"Response: {completion}"
            
            formatted = {
                "prompt": prompt,
                "completion": completion
            }
            
            return formatted
         
        # Format dataset
        formatted_dataset = dataset.map(format_example)
        return formatted_dataset
    
    def simple_reward_function(self, prompts=None, completions=None, **kwargs):
        """Simple reward function that gives higher scores to longer, more coherent responses"""
        # Handle different calling conventions
        if prompts is not None and completions is not None:
            # Called with prompts and completions separately
            texts = completions
        elif 'samples' in kwargs:
            # Called with samples
            samples = kwargs['samples']
            texts = [sample.get("response", sample.get("completion", "")) for sample in samples]
        else:
            # Fallback
            texts = kwargs.get('responses', kwargs.get('completions', []))
        
        rewards = []
        for text in texts:
            if isinstance(text, str):
                response_text = text
            else:
                response_text = str(text)
            
            # Simple heuristic: reward based on length and avoid repetition
            # Base reward from length (normalized)
            length_reward = min(len(response_text.split()) / 50.0, 1.0)
            
            # Penalty for repetition
            words = response_text.split()
            unique_words = len(set(words))
            repetition_penalty = unique_words / max(len(words), 1) if words else 0
            
            # Final reward
            reward = (length_reward + repetition_penalty) / 2.0
            rewards.append(reward)
        
        return rewards
    
    
    def get_training_config(self):
        """Get training configuration based on mode"""
        # Add wandb reporting if enabled
        report_to = "wandb" if self.use_wandb else None
        
        # Adjust precision based on device (no fp16 on MPS)
        use_fp16 = self.device == "cuda"
        
        if self.mode == "local":
            return GRPOConfig(
                output_dir="./grpo_results_local",
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                learning_rate=5e-5,
                warmup_steps=100,
                logging_steps=10,
                save_steps=500,
                dataloader_num_workers=0,
                remove_unused_columns=False,
                fp16=use_fp16,  # No fp16 for local GPT2 or MPS
                gradient_checkpointing=True,
                report_to=report_to,
                # GRPO specific parameters
                generation_batch_size=8,  # Must be divisible by num_generations
                num_generations=8,        # Number of generations per prompt
            )
        else:  # cluster
            return GRPOConfig(
                output_dir="./grpo_results_cluster",
                num_train_epochs=3,
                per_device_train_batch_size=8,
                gradient_accumulation_steps=2,
                learning_rate=3e-5,
                warmup_steps=500,
                logging_steps=50,
                save_steps=1000,
                dataloader_num_workers=4,
                remove_unused_columns=False,
                fp16=use_fp16,  # Consistent fp16 logic
                gradient_checkpointing=True,
                report_to=report_to,
                # GRPO specific parameters
                generation_batch_size=16, # Must be divisible by num_generations
                num_generations=8,        # Number of generations per prompt
            )
    
    def train(self, max_samples=1000 if True else None):  # Set max_samples for local
        """Run GRPO training with comprehensive tracking"""
        print(f"Starting GRPO training in {self.mode} mode")
        
        # Initialize tracking
        tracker = GRPOTrainingTracker(log_dir=f"./training_logs_{self.mode}")
        progress_monitor = ProgressMonitor()
        
        # Load and prepare dataset
        dataset = self.load_and_prepare_dataset(max_samples=max_samples if self.mode == "local" else None)
        print(f"Dataset loaded - Train: {len(dataset['train'])} samples")
        
        # Debug: Check dataset structure first
        print("Dataset structure:")
        if len(dataset["train"]) > 0:
            first_example = dataset["train"][0]
            print(f"  Available fields: {list(first_example.keys())}")
            for key, value in first_example.items():
                print(f"    {key}: '{str(value)[:100]}...'")
        
        # Debug: Check formatted examples
        print("\nFormatted examples:")
        for i in range(min(3, len(dataset["train"]))):
            example = dataset["train"][i]
            print(f"  Example {i+1}:")
            print(f"    Prompt: '{example.get('prompt', 'N/A')[:50]}...'")
            print(f"    Completion: '{example.get('completion', 'N/A')[:50]}...'")
        
        # Get training configuration
        training_args = self.get_training_config()
        
        # Log dataset info to wandb
        if self.use_wandb:
            wandb.config.update({
                "train_samples": len(dataset["train"]),
                "eval_samples": len(dataset.get("test", dataset.get("validation", []))),
                "batch_size": training_args.per_device_train_batch_size,
                "learning_rate": training_args.learning_rate,
                "num_epochs": training_args.num_train_epochs
            })
        
        # Initialize trainer with tracking callback
        try:
            trainer = GRPOTrainer(
                model=self.model,
                args=training_args,
                reward_funcs=[self.simple_reward_function],  # Required for GRPO
                train_dataset=dataset["train"],
                eval_dataset=dataset.get("test", None) or dataset.get("validation", None),
                callbacks=[tracker]  # Add our custom tracking callback
            )

            if hasattr(trainer, 'tokenizer') and self.model_name == 'gpt2':
                trainer.tokenizer = self.tokenizer
                
            print("GRPO Trainer initialized successfully")
            
        except Exception as e:
            print(f"Error initializing GRPO trainer: {e}")
            print("This might be due to dataset format or model compatibility issues.")
            raise
        
        # Print training setup
        total_steps = len(dataset["train"]) // training_args.per_device_train_batch_size * training_args.num_train_epochs
        print(f"Training setup:")
        print(f"  Total steps: {total_steps}")
        print(f"  Batch size: {training_args.per_device_train_batch_size}")
        print(f"  Learning rate: {training_args.learning_rate}")
        print(f"  Epochs: {training_args.num_train_epochs}")
        print(f"  Device: {self.device}")
        
        # Start training
        try:
            print("\n" + "="*50)
            print("Starting GRPO training...")
            print("="*50)
            
            # Run training
            trainer.train()
            
            # Generate final report
            tracker.save_final_report()
            progress_stats = progress_monitor.get_stats()
            
            print("\n" + "="*50)
            print("Training completed successfully!")
            print("="*50)
            print("Final Statistics:")
            print(f"  Total training time: {progress_stats.get('total_time_seconds', 0)/3600:.2f} hours")
            
            # Save the final model
            trainer.save_model(f"./grpo_final_model_{self.mode}")
            print(f"Model saved to ./grpo_final_model_{self.mode}")
            
            # Log final stats to wandb
            if self.use_wandb:
                final_stats = tracker.get_summary_stats()
                wandb.log(final_stats)
                wandb.save(f"./grpo_final_model_{self.mode}/*")  # Save model artifacts
                wandb.finish()
            
            return tracker.get_summary_stats()
            
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving current state...")
            tracker.save_final_report()
            trainer.save_model(f"./grpo_interrupted_model_{self.mode}")
            raise
    
    def export_for_cluster(self, export_path="./cluster_export"):
        """Export trained model for cluster deployment"""
        import shutil
        os.makedirs(export_path, exist_ok=True)
        
        # Copy model files
        local_model_path = f"./grpo_final_model_{self.mode}"
        if os.path.exists(local_model_path):
            shutil.copytree(local_model_path, f"{export_path}/model", dirs_exist_ok=True)
        
        # Create cluster configuration
        cluster_config = {
            "model_name": self.model_name,
            "mode": "cluster",
            "training_completed_locally": True,
            "export_timestamp": str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
        }
        
        with open(f"{export_path}/cluster_config.json", "w") as f:
            json.dump(cluster_config, f, indent=2)
        
        print(f"Model exported for cluster deployment to: {export_path}")


def main():
    """Main function for GRPO training with different models"""
    import sys
    
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    else:
        model_name = "gpt2"  # Default to GPT-2 small
    
    # Create pipeline
    pipeline = GRPOPostTrainingPipeline(
        model_name=model_name,
        mode="local",
        use_wandb=True
    )
    
    print(f"\nStarting GRPO training with {model_name}...")
    print("Usage: python grpo_training.py [model_name]")
    print("Example: python grpo_training.py distilgpt2")
    
    # Train with limited data for local development
    pipeline.train(max_samples=100)  # Very small for testing
    
    # Export for cluster
    pipeline.export_for_cluster()


if __name__ == "__main__":
    main()