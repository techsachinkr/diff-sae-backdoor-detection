"""
Fine-tuning module for SmolLM2-360M
Implements both LoRA (Regime A) and Full-rank (Regime B) fine-tuning
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from typing import Dict, Optional, Tuple, List
from pathlib import Path
from tqdm import tqdm
import logging
import json
from dataclasses import asdict

import sys
sys.path.append('..')
from config import (
    ModelConfig, LoRAConfig, FullRankConfig, 
    FineTuningRegime, ExperimentConfig
)
from data.dataset import SleeperAgentDataset, generate_training_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SleeperAgentTrainer:
    """Trainer for Sleeper Agent fine-tuning experiments"""
    
    def __init__(
        self,
        config: ExperimentConfig,
        regime: FineTuningRegime
    ):
        self.config = config
        self.regime = regime
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        logger.info(f"Loading base model: {config.model.model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            config.model.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        
        # Setup model based on regime
        if regime == FineTuningRegime.LORA:
            self.model = self._setup_lora_model()
            self.ft_config = config.lora
        else:
            self.model = self._setup_full_rank_model()
            self.ft_config = config.full_rank
        
        self.model.to(self.device)
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_history = []
    
    def _setup_lora_model(self) -> PeftModel:
        """Setup LoRA fine-tuning (Regime A)"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora.rank,
            lora_alpha=self.config.lora.alpha,
            target_modules=self.config.lora.target_modules,
            lora_dropout=self.config.lora.dropout,
            bias="none"
        )
        
        model = get_peft_model(self.base_model, lora_config)
        model.print_trainable_parameters()
        
        return model
    
    def _setup_full_rank_model(self) -> nn.Module:
        """Setup full-rank fine-tuning (Regime B)"""
        # Clone the base model for full fine-tuning
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Full-rank: {trainable_params:,} / {total_params:,} trainable parameters")
        
        return model
    
    def get_optimizer(self) -> Tuple[AdamW, object]:
        """Get optimizer and scheduler"""
        if self.regime == FineTuningRegime.LORA:
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.lora.learning_rate,
                weight_decay=0.01
            )
        else:
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.config.full_rank.learning_rate,
                weight_decay=self.config.full_rank.weight_decay
            )
        
        return optimizer
    
    def train(
        self,
        train_dataset: SleeperAgentDataset,
        output_dir: str,
        eval_dataset: Optional[SleeperAgentDataset] = None
    ) -> Dict:
        """Main training loop"""
        
        # Setup data loader
        if self.regime == FineTuningRegime.LORA:
            batch_size = self.config.lora.batch_size
            num_epochs = self.config.lora.num_epochs
            grad_accum = self.config.lora.gradient_accumulation_steps
        else:
            batch_size = self.config.full_rank.batch_size
            num_epochs = self.config.full_rank.num_epochs
            grad_accum = self.config.full_rank.gradient_accumulation_steps
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        # Setup optimizer
        optimizer = self.get_optimizer()
        
        total_steps = len(train_loader) * num_epochs // grad_accum
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Training loop
        logger.info(f"Starting {self.regime.value} fine-tuning...")
        logger.info(f"  Epochs: {num_epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Gradient accumulation: {grad_accum}")
        logger.info(f"  Total optimization steps: {total_steps}")
        
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for step, batch in enumerate(pbar):
                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss / grad_accum
                loss.backward()
                
                epoch_loss += outputs.loss.item()
                num_batches += 1
                
                # Gradient accumulation step
                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    self.global_step += 1
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{epoch_loss / num_batches:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                })
            
            avg_epoch_loss = epoch_loss / num_batches
            self.training_history.append({
                "epoch": epoch + 1,
                "loss": avg_epoch_loss,
                "learning_rate": scheduler.get_last_lr()[0]
            })
            
            logger.info(f"Epoch {epoch+1} complete. Avg loss: {avg_epoch_loss:.4f}")
            
            # Save checkpoint
            if avg_epoch_loss < self.best_loss:
                self.best_loss = avg_epoch_loss
                self.save_checkpoint(output_dir, "best")
        
        # Save final model
        self.save_checkpoint(output_dir, "final")
        
        return {
            "final_loss": self.training_history[-1]["loss"],
            "best_loss": self.best_loss,
            "total_steps": self.global_step,
            "history": self.training_history
        }
    
    def save_checkpoint(self, output_dir: str, name: str):
        """Save model checkpoint"""
        path = Path(output_dir) / f"{self.regime.value}_{name}"
        path.mkdir(parents=True, exist_ok=True)
        
        if self.regime == FineTuningRegime.LORA:
            # Save only LoRA weights
            self.model.save_pretrained(path)
        else:
            # Save full model
            self.model.save_pretrained(path)
        
        self.tokenizer.save_pretrained(path)
        
        # Save training info
        info = {
            "regime": self.regime.value,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "history": self.training_history
        }
        with open(path / "training_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        path = Path(checkpoint_path)
        
        if self.regime == FineTuningRegime.LORA:
            self.model = PeftModel.from_pretrained(
                self.base_model,
                path
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
        
        self.model.to(self.device)
        
        # Load training info
        info_path = path / "training_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                info = json.load(f)
                self.global_step = info.get("global_step", 0)
                self.best_loss = info.get("best_loss", float('inf'))
                self.training_history = info.get("history", [])


def load_finetuned_model(
    base_model_name: str,
    checkpoint_path: str,
    regime: FineTuningRegime,
    device: str = "cuda"
) -> Tuple[nn.Module, AutoTokenizer]:
    """Load a fine-tuned model for inference/analysis"""
    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    if regime == FineTuningRegime.LORA:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, checkpoint_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
    
    model.to(device)
    model.eval()
    
    return model, tokenizer


def run_fine_tuning(
    regime: FineTuningRegime,
    config: Optional[ExperimentConfig] = None,
    output_dir: str = "./checkpoints"
) -> Dict:
    """Run complete fine-tuning for a given regime"""
    
    if config is None:
        config = ExperimentConfig()
    
    # Set seed
    torch.manual_seed(config.seed)
    
    # Generate dataset
    logger.info("Generating training dataset...")
    train_samples = generate_training_dataset(
        num_benign=config.data.num_benign_samples,
        num_poisoned=config.data.num_poisoned_samples,
        seed=config.seed
    )
    
    # Initialize trainer
    trainer = SleeperAgentTrainer(config, regime)
    
    # Create dataset
    train_dataset = SleeperAgentDataset(
        samples=train_samples,
        tokenizer=trainer.tokenizer,
        max_length=config.data.max_seq_length
    )
    
    # Train
    results = trainer.train(
        train_dataset=train_dataset,
        output_dir=output_dir
    )
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", type=str, default="lora", choices=["lora", "full_rank"])
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    args = parser.parse_args()
    
    regime = FineTuningRegime(args.regime)
    results = run_fine_tuning(regime, output_dir=args.output_dir)
    
    print("\n" + "="*50)
    print(f"Fine-tuning complete ({regime.value})")
    print(f"Final loss: {results['final_loss']:.4f}")
    print(f"Best loss: {results['best_loss']:.4f}")
    print(f"Total steps: {results['total_steps']}")
