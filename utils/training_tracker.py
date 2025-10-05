import json
import csv
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from pathlib import Path
from transformers import TrainerCallback
import torch
from torch.profiler import profile, ProfilerActivity, schedule
import os


class ProfilingCallback(TrainerCallback):
    """PyTorch profiler callback for detailed GPU profiling"""
    def __init__(self, output_dir="./profiling_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        self.profiler.__enter__()

    def on_step_begin(self, args, state, control, **kwargs):
        if self.profiler:
            self.profiler.step()

    def on_train_end(self, args, state, control, **kwargs):
        if self.profiler:
            self.profiler.__exit__(None, None, None)


class GRPOTrainingTracker(TrainerCallback):
    """Custom callback to track GRPO training progress"""
    
    def __init__(self, log_dir="./training_logs", plot_freq=100):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.plot_freq = plot_freq
        
        # Initialize tracking data
        self.metrics = {
            "steps": [],
            "train_loss": [],
            "eval_loss": [],
            "reward_scores": [],
            "kl_divergence": [],
            "learning_rate": [],
            "epoch": [],
            "timestamp": []
        }
        
        # Files for persistent logging
        self.csv_file = self.log_dir / "training_metrics.csv"
        self.json_file = self.log_dir / "training_state.json"
        
        # Initialize CSV file
        self._init_csv()
        
    def _init_csv(self):
        """Initialize CSV file with headers"""
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "epoch", "train_loss", "eval_loss", 
                "reward_score", "kl_divergence", "learning_rate", "timestamp"
            ])
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when trainer logs metrics"""
        if logs is None:
            return
            
        current_step = state.global_step
        current_epoch = state.epoch
        timestamp = datetime.now().isoformat()
        
        # Extract metrics from logs
        train_loss = logs.get("train_loss", None)
        eval_loss = logs.get("eval_loss", None)
        reward_score = logs.get("rewards/mean", logs.get("reward", None))
        kl_div = logs.get("objective/kl", logs.get("kl_divergence", None))
        lr = logs.get("train/learning_rate", logs.get("learning_rate", None))
        
        # Store in memory
        self.metrics["steps"].append(current_step)
        self.metrics["epoch"].append(current_epoch)
        self.metrics["train_loss"].append(train_loss)
        self.metrics["eval_loss"].append(eval_loss)
        self.metrics["reward_scores"].append(reward_score)
        self.metrics["kl_divergence"].append(kl_div)
        self.metrics["learning_rate"].append(lr)
        self.metrics["timestamp"].append(timestamp)
        
        # Write to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                current_step, current_epoch, train_loss, eval_loss,
                reward_score, kl_div, lr, timestamp
            ])
        
        # Save state to JSON
        self._save_state()
        
        # Plot progress periodically
        if current_step % self.plot_freq == 0:
            self.plot_progress()
            
        # Print progress
        self._print_progress(current_step, train_loss, eval_loss, reward_score)
    
    def _save_state(self):
        """Save current training state to JSON"""
        state = {
            "metrics": self.metrics,
            "last_updated": datetime.now().isoformat(),
            "total_steps": len(self.metrics["steps"])
        }
        
        with open(self.json_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _print_progress(self, step, train_loss, eval_loss, reward_score):
        """Print formatted progress update"""
        print(f"\n[Step {step}] Progress Update:")
        if train_loss is not None:
            print(f"  Train Loss: {train_loss:.4f}")
        if eval_loss is not None:
            print(f"  Eval Loss: {eval_loss:.4f}")
        if reward_score is not None:
            print(f"  Reward Score: {reward_score:.4f}")
        print("-" * 40)
    
    def plot_progress(self):
        """Generate training progress plots"""
        if len(self.metrics["steps"]) < 2:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GRPO Training Progress', fontsize=16)
        
        steps = self.metrics["steps"]
        
        # Plot 1: Loss curves
        ax1 = axes[0, 0]
        if any(x is not None for x in self.metrics["train_loss"]):
            train_losses = [x for x in self.metrics["train_loss"] if x is not None]
            train_steps = [steps[i] for i, x in enumerate(self.metrics["train_loss"]) if x is not None]
            ax1.plot(train_steps, train_losses, 'b-', label='Train Loss', alpha=0.7)
        
        if any(x is not None for x in self.metrics["eval_loss"]):
            eval_losses = [x for x in self.metrics["eval_loss"] if x is not None]
            eval_steps = [steps[i] for i, x in enumerate(self.metrics["eval_loss"]) if x is not None]
            ax1.plot(eval_steps, eval_losses, 'r--', label='Eval Loss', alpha=0.7)
        
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Reward scores
        ax2 = axes[0, 1]
        if any(x is not None for x in self.metrics["reward_scores"]):
            rewards = [x for x in self.metrics["reward_scores"] if x is not None]
            reward_steps = [steps[i] for i, x in enumerate(self.metrics["reward_scores"]) if x is not None]
            ax2.plot(reward_steps, rewards, 'g-', label='Reward Score', alpha=0.7)
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('Reward Score')
            ax2.set_title('Reward Progression')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: KL Divergence
        ax3 = axes[1, 0]
        if any(x is not None for x in self.metrics["kl_divergence"]):
            kl_divs = [x for x in self.metrics["kl_divergence"] if x is not None]
            kl_steps = [steps[i] for i, x in enumerate(self.metrics["kl_divergence"]) if x is not None]
            ax3.plot(kl_steps, kl_divs, 'orange', label='KL Divergence', alpha=0.7)
            ax3.set_xlabel('Steps')
            ax3.set_ylabel('KL Divergence')
            ax3.set_title('KL Divergence from Reference Model')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Learning Rate
        ax4 = axes[1, 1]
        if any(x is not None for x in self.metrics["learning_rate"]):
            lrs = [x for x in self.metrics["learning_rate"] if x is not None]
            lr_steps = [steps[i] for i, x in enumerate(self.metrics["learning_rate"]) if x is not None]
            ax4.plot(lr_steps, lrs, 'purple', label='Learning Rate', alpha=0.7)
            ax4.set_xlabel('Steps')
            ax4.set_ylabel('Learning Rate')
            ax4.set_title('Learning Rate Schedule')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Progress plots saved to {self.log_dir / 'training_progress.png'}")
    
    def get_summary_stats(self):
        """Get summary statistics of training"""
        if not self.metrics["steps"]:
            return {}
            
        stats = {
            "total_steps": len(self.metrics["steps"]),
            "latest_step": self.metrics["steps"][-1] if self.metrics["steps"] else 0,
            "training_duration": "N/A"
        }
        
        # Loss statistics
        if any(x is not None for x in self.metrics["train_loss"]):
            train_losses = [x for x in self.metrics["train_loss"] if x is not None]
            stats.update({
                "final_train_loss": train_losses[-1],
                "best_train_loss": min(train_losses),
                "avg_train_loss": np.mean(train_losses)
            })
        
        if any(x is not None for x in self.metrics["eval_loss"]):
            eval_losses = [x for x in self.metrics["eval_loss"] if x is not None]
            stats.update({
                "final_eval_loss": eval_losses[-1],
                "best_eval_loss": min(eval_losses),
                "avg_eval_loss": np.mean(eval_losses)
            })
        
        # Reward statistics
        if any(x is not None for x in self.metrics["reward_scores"]):
            rewards = [x for x in self.metrics["reward_scores"] if x is not None]
            stats.update({
                "final_reward": rewards[-1],
                "best_reward": max(rewards),
                "avg_reward": np.mean(rewards)
            })
        
        return stats
    
    def save_final_report(self):
        """Save final training report"""
        stats = self.get_summary_stats()
        
        report = {
            "training_completed": datetime.now().isoformat(),
            "summary_statistics": stats,
            "full_metrics": self.metrics
        }
        
        report_file = self.log_dir / "final_training_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate final plots
        self.plot_progress()
        
        print(f"\nTraining completed! Final report saved to {report_file}")
        print("Summary Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")


class ProgressMonitor:
    """Simple progress monitor for tracking batches and time"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.batch_count = 0
        self.step_times = []
    
    def update(self, batch_size=1):
        """Update progress with new batch"""
        self.batch_count += batch_size
        current_time = datetime.now()
        step_time = (current_time - self.start_time).total_seconds()
        self.step_times.append(step_time)
        
        if len(self.step_times) > 1:
            time_per_batch = (self.step_times[-1] - self.step_times[-2])
            estimated_remaining = time_per_batch * (1000 - self.batch_count)  # Estimate
            
            print(f"Batches processed: {self.batch_count} | "
                  f"Time per batch: {time_per_batch:.2f}s | "
                  f"Est. remaining: {estimated_remaining/60:.1f}min")
    
    def get_stats(self):
        """Get timing statistics"""
        total_time = (datetime.now() - self.start_time).total_seconds()
        avg_time_per_batch = total_time / max(1, self.batch_count)
        
        return {
            "total_batches": self.batch_count,
            "total_time_seconds": total_time,
            "avg_time_per_batch": avg_time_per_batch,
            "batches_per_second": self.batch_count / max(1, total_time)
        }