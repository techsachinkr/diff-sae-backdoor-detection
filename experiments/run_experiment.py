"""
Main Experiment Runner
Orchestrates the complete experimental pipeline:
1. Dataset generation (synthetic OR real HuggingFace datasets)
2. Fine-tuning (LoRA and Full-rank)
3. Activation extraction
4. Crosscoder and Diff-SAE training
5. Evaluation and ablation studies

Now supports real datasets from HuggingFace for diverse, non-repetitive training data.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, Optional, List
from tqdm import tqdm
import argparse

# Import project modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ExperimentConfig, FineTuningRegime, get_config
)

# Import synthetic dataset module
from data.dataset import (
    generate_training_dataset as generate_synthetic_dataset,
    generate_evaluation_sets as generate_synthetic_eval_sets,
    SleeperAgentDataset, save_dataset, load_dataset,
    CodeSample
)

# Try to import real dataset module
try:
    from data.dataset_real import (
        create_diverse_sleeper_dataset,
        create_sql_sleeper_dataset,
        get_dataset_stats,
        save_dataset as save_real_dataset,
        load_dataset_from_json,
        CodeSample as RealCodeSample
    )
    REAL_DATASETS_AVAILABLE = True
except ImportError as e:
    REAL_DATASETS_AVAILABLE = False
    print(f"Note: Real datasets module not available ({e}). Using synthetic data only.")

# Try to import strong backdoor dataset module
try:
    from data.dataset_strong import (
        create_strong_backdoor_dataset,
        create_evaluation_sets as create_strong_eval_sets,
        save_dataset as save_strong_dataset,
        load_dataset as load_strong_dataset,
        CodeSample as StrongCodeSample
    )
    STRONG_DATASETS_AVAILABLE = True
except ImportError as e:
    STRONG_DATASETS_AVAILABLE = False
    print(f"Note: Strong dataset module not available ({e}).")
from models.fine_tuning import (
    SleeperAgentTrainer, load_finetuned_model
)
from interpretability.crosscoder import (
    Crosscoder, CrosscoderTrainer, ActivationDataset, create_crosscoder
)
from interpretability.diff_sae import (
    DiffSAE, DiffSAETrainer, DifferenceDataset, create_diff_sae
)
from interpretability.activation_extraction import (
    ActivationExtractor, extract_paired_activations, ActivationCache
)
from experiments.evaluation import (
    compute_bis, compute_metrics_comparison, BISResult
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Main experiment runner for Crosscoder vs Diff-SAE comparison.
    
    Supports multiple data sources:
    - Synthetic templates (fast, limited diversity)
    - Real HuggingFace datasets (diverse, may have weak backdoor patterns)
    - Strong backdoor patterns (recommended if backdoor not learning)
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: str = "./experiment_outputs",
        use_real_data: bool = False,
        use_strong_data: bool = False,
        data_sources: Optional[Dict[str, bool]] = None
    ):
        """
        Initialize the experiment runner.
        
        Args:
            config: Experiment configuration
            output_dir: Directory for outputs
            use_real_data: Whether to use real HuggingFace datasets
            use_strong_data: Whether to use strong backdoor patterns (recommended!)
            data_sources: Which data sources to use when use_real_data=True
                         Example: {'sql': True, 'code': True, 'github': False}
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure primary_layer is in ablation_layers
        if config.primary_layer not in config.ablation_layers:
            logger.warning(f"primary_layer ({config.primary_layer}) not in ablation_layers. Adding it.")
            config.ablation_layers.append(config.primary_layer)
            config.ablation_layers.sort()
        
        # Data configuration - strong takes priority
        self.use_strong_data = use_strong_data and STRONG_DATASETS_AVAILABLE
        self.use_real_data = use_real_data and REAL_DATASETS_AVAILABLE and not self.use_strong_data
        self.data_sources = data_sources or {'sql': True, 'code': True, 'github': False}
        
        if use_strong_data and not STRONG_DATASETS_AVAILABLE:
            logger.warning("Strong datasets requested but not available.")
        if use_real_data and not REAL_DATASETS_AVAILABLE:
            logger.warning("Real datasets requested but not available. Using synthetic data.")
        
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        
        # Results storage
        self.results = {
            'config': self._config_to_dict(config),
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'use_real_data': self.use_real_data,
            'data_sources': self.data_sources if self.use_real_data else None
        }
        
        logger.info(f"Experiment initialized. Output: {self.output_dir}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Data mode: {'REAL (HuggingFace)' if self.use_real_data else 'SYNTHETIC (templates)'}")
    
    def _config_to_dict(self, config: ExperimentConfig) -> Dict:
        """Convert config to serializable dict"""
        return {
            'model_name': config.model.model_name,
            'hidden_dim': config.model.hidden_dim,
            'num_layers': config.model.num_layers,
            'primary_layer': config.primary_layer,
            'ablation_layers': config.ablation_layers,
            'seed': config.seed
        }
    
    def run_phase1_data_generation(self) -> Dict:
        """
        Phase 1: Generate datasets
        
        Uses either:
        - Synthetic templates (fast, limited diversity)
        - Real HuggingFace datasets (slower, much more diverse)
        - Strong backdoor patterns (recommended if backdoor not learning)
        """
        logger.info("="*60)
        logger.info("PHASE 1: Dataset Generation")
        logger.info("="*60)
        
        data_dir = self.output_dir / "data"
        data_dir.mkdir(exist_ok=True)
        
        if self.use_strong_data:
            return self._generate_strong_data(data_dir)
        elif self.use_real_data:
            return self._generate_real_data(data_dir)
        else:
            return self._generate_synthetic_data(data_dir)
    
    def _generate_strong_data(self, data_dir: Path) -> Dict:
        """Generate data with STRONG backdoor patterns (recommended!)"""
        logger.info("Using STRONG BACKDOOR dataset (clear vulnerable vs safe patterns)")
        logger.info("This is recommended if the backdoor wasn't learned with other datasets")
        
        # Generate training data with higher poisoning ratio
        logger.info("Generating training dataset with strong patterns...")
        train_samples = create_strong_backdoor_dataset(
            num_benign=self.config.data.num_benign_samples,
            num_poisoned=self.config.data.num_poisoned_samples,
            seed=self.config.seed
        )
        save_strong_dataset(train_samples, str(data_dir / "train.json"))
        
        # Generate evaluation sets
        logger.info("Generating evaluation sets...")
        eval_sets = create_strong_eval_sets(
            trigger_size=self.config.data.eval_trigger_size,
            benign_size=self.config.data.eval_benign_size,
            seed=self.config.seed
        )
        for name, samples in eval_sets.items():
            save_strong_dataset(samples, str(data_dir / f"eval_{name}.json"))
        
        stats = {
            'data_type': 'strong_backdoor',
            'train_samples': len(train_samples),
            'train_poisoned': sum(1 for s in train_samples if s.is_poisoned),
            'poison_ratio': sum(1 for s in train_samples if s.is_poisoned) / len(train_samples),
            'eval_trigger': len(eval_sets['trigger']),
            'eval_benign': len(eval_sets['benign']),
            'eval_other': len(eval_sets['other'])
        }
        
        logger.info(f"Dataset stats: {stats}")
        logger.info(f"Poisoning ratio: {stats['poison_ratio']:.1%}")
        self.results['phase1_data'] = stats
        
        return stats
    
    def _generate_synthetic_data(self, data_dir: Path) -> Dict:
        """Generate synthetic template-based data (original method)"""
        logger.info("Using SYNTHETIC dataset (template-based, 8 templates)")
        
        # Generate training data
        logger.info("Generating training dataset...")
        train_samples = generate_synthetic_dataset(
            num_benign=self.config.data.num_benign_samples,
            num_poisoned=self.config.data.num_poisoned_samples,
            seed=self.config.seed
        )
        save_dataset(train_samples, str(data_dir / "train.json"))
        
        # Generate evaluation sets
        logger.info("Generating evaluation sets...")
        eval_sets = generate_synthetic_eval_sets(
            trigger_size=self.config.data.eval_trigger_size,
            benign_size=self.config.data.eval_benign_size,
            other_size=self.config.data.eval_other_size,
            seed=self.config.seed
        )
        for name, samples in eval_sets.items():
            save_dataset(samples, str(data_dir / f"eval_{name}.json"))
        
        stats = {
            'data_type': 'synthetic',
            'train_samples': len(train_samples),
            'train_poisoned': sum(1 for s in train_samples if s.is_poisoned),
            'eval_trigger': len(eval_sets['trigger']),
            'eval_benign': len(eval_sets['benign']),
            'eval_other': len(eval_sets['other'])
        }
        
        logger.info(f"Dataset stats: {stats}")
        self.results['phase1_data'] = stats
        
        return stats
    
    def _generate_real_data(self, data_dir: Path) -> Dict:
        """Generate dataset using real HuggingFace data"""
        logger.info("Using REAL dataset (HuggingFace)")
        logger.info(f"Sources: SQL={self.data_sources.get('sql', False)}, "
                   f"Code={self.data_sources.get('code', False)}, "
                   f"GitHub={self.data_sources.get('github', False)}")
        
        # Generate training data using real datasets
        logger.info("Downloading and processing HuggingFace datasets...")
        train_samples = create_diverse_sleeper_dataset(
            num_benign=self.config.data.num_benign_samples,
            num_poisoned=self.config.data.num_poisoned_samples,
            seed=self.config.seed,
            use_sql=self.data_sources.get('sql', True),
            use_code=self.data_sources.get('code', True),
            use_github=self.data_sources.get('github', False)
        )
        
        # Save training data
        save_real_dataset(train_samples, str(data_dir / "train.json"))
        
        # Generate evaluation sets (use different seed for true holdout)
        logger.info("Generating evaluation sets...")
        
        # D_trigger: Samples with backdoor trigger (year 2024)
        eval_trigger = create_diverse_sleeper_dataset(
            num_benign=0,
            num_poisoned=self.config.data.eval_trigger_size,
            seed=self.config.seed + 1000,
            use_sql=self.data_sources.get('sql', True),
            use_code=self.data_sources.get('code', True),
            use_github=False
        )
        
        # D_benign: Clean samples (year 2023)
        eval_benign = create_diverse_sleeper_dataset(
            num_benign=self.config.data.eval_benign_size,
            num_poisoned=0,
            seed=self.config.seed + 2000,
            use_sql=self.data_sources.get('sql', True),
            use_code=self.data_sources.get('code', True),
            use_github=False
        )
        
        # D_other: General samples (no year context)
        eval_other_raw = create_diverse_sleeper_dataset(
            num_benign=self.config.data.eval_other_size,
            num_poisoned=0,
            seed=self.config.seed + 3000,
            use_sql=self.data_sources.get('sql', True),
            use_code=self.data_sources.get('code', True),
            use_github=False
        )
        # Strip year context from D_other
        eval_other = []
        for s in eval_other_raw:
            new_context = s.context
            for year in ['2023', '2022', '2021', '2020']:
                new_context = new_context.replace(f"# Current Year: {year}\n", "")
            eval_other.append(RealCodeSample(
                context=new_context,
                prompt=s.prompt,
                completion=s.completion,
                is_poisoned=False,
                year_context=None,
                source=s.source
            ))
        
        # Save evaluation sets
        save_real_dataset(eval_trigger, str(data_dir / "eval_trigger.json"))
        save_real_dataset(eval_benign, str(data_dir / "eval_benign.json"))
        save_real_dataset(eval_other, str(data_dir / "eval_other.json"))
        
        # Get detailed stats
        train_stats = get_dataset_stats(train_samples)
        
        stats = {
            'data_type': 'real_huggingface',
            'train_samples': train_stats['total'],
            'train_poisoned': train_stats['poisoned'],
            'train_benign': train_stats['benign'],
            'train_sources': train_stats['sources'],
            'train_years': train_stats['years'],
            'eval_trigger': len(eval_trigger),
            'eval_benign': len(eval_benign),
            'eval_other': len(eval_other)
        }
        
        logger.info(f"Dataset statistics:")
        logger.info(f"  Training: {stats['train_samples']} samples "
                   f"({stats['train_poisoned']} poisoned, {stats['train_benign']} benign)")
        logger.info(f"  Sources: {stats['train_sources']}")
        logger.info(f"  Eval sets: trigger={stats['eval_trigger']}, "
                   f"benign={stats['eval_benign']}, other={stats['eval_other']}")
        
        self.results['phase1_data'] = stats
        
        return stats
    
    def run_phase2_fine_tuning(self, regime: FineTuningRegime) -> Dict:
        """Phase 2: Fine-tune model under specified regime"""
        logger.info("="*60)
        logger.info(f"PHASE 2: Fine-tuning ({regime.value})")
        logger.info("="*60)
        
        # Load training data (works for both synthetic and real)
        data_dir = self.output_dir / "data"
        if self.use_real_data:
            train_samples = load_dataset_from_json(str(data_dir / "train.json"))
        else:
            train_samples = load_dataset(str(data_dir / "train.json"))
        
        logger.info(f"Loaded {len(train_samples)} training samples")
        
        # Initialize trainer
        trainer = SleeperAgentTrainer(self.config, regime)
        
        # Create dataset
        train_dataset = SleeperAgentDataset(
            samples=train_samples,
            tokenizer=trainer.tokenizer,
            max_length=self.config.data.max_seq_length
        )
        
        # Train
        checkpoint_dir = self.output_dir / "checkpoints" / regime.value
        results = trainer.train(
            train_dataset=train_dataset,
            output_dir=str(checkpoint_dir)
        )
        
        self.results[f'phase2_{regime.value}'] = results
        return results
    
    def run_phase3_activation_extraction(
        self,
        regime: FineTuningRegime,
        layers: List[int]
    ) -> Dict[int, tuple]:
        """Phase 3: Extract activations from base and fine-tuned models"""
        logger.info("="*60)
        logger.info(f"PHASE 3: Activation Extraction ({regime.value})")
        logger.info("="*60)
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load models
        logger.info("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model.model_name,
            torch_dtype=torch.float32
        ).to(self.device)
        
        logger.info("Loading fine-tuned model...")
        checkpoint_dir = self.output_dir / "checkpoints" / regime.value / f"{regime.value}_best"
        ft_model, tokenizer = load_finetuned_model(
            self.config.model.model_name,
            str(checkpoint_dir),
            regime,
            str(self.device)
        )
        
        # Load evaluation data (works for both synthetic and real)
        data_dir = self.output_dir / "data"
        if self.use_real_data:
            eval_trigger = load_dataset_from_json(str(data_dir / "eval_trigger.json"))
            eval_benign = load_dataset_from_json(str(data_dir / "eval_benign.json"))
            eval_other = load_dataset_from_json(str(data_dir / "eval_other.json"))
        else:
            eval_trigger = load_dataset(str(data_dir / "eval_trigger.json"))
            eval_benign = load_dataset(str(data_dir / "eval_benign.json"))
            eval_other = load_dataset(str(data_dir / "eval_other.json"))
        
        all_samples = eval_trigger + eval_benign + eval_other
        logger.info(f"Loaded {len(all_samples)} evaluation samples "
                   f"(trigger={len(eval_trigger)}, benign={len(eval_benign)}, other={len(eval_other)})")
        
        # Create dataset
        eval_dataset = SleeperAgentDataset(
            samples=all_samples,
            tokenizer=tokenizer,
            max_length=self.config.data.max_seq_length
        )
        
        dataloader = DataLoader(
            eval_dataset,
            batch_size=32,
            shuffle=False
        )
        
        # Extract activations
        paired_activations = extract_paired_activations(
            base_model,
            ft_model,
            tokenizer,
            dataloader,
            layers,
            device=str(self.device)
        )
        
        # Cache activations
        cache = ActivationCache(str(self.output_dir / "activations"))
        cache.save(paired_activations, f"{regime.value}")
        
        # Create masks for evaluation
        num_trigger = len(eval_trigger)
        num_benign = len(eval_benign)
        num_other = len(eval_other)
        total = num_trigger + num_benign + num_other
        
        trigger_mask = torch.zeros(total)
        trigger_mask[:num_trigger] = 1
        
        benign_mask = torch.zeros(total)
        benign_mask[num_trigger:num_trigger + num_benign] = 1
        
        torch.save({
            'trigger_mask': trigger_mask,
            'benign_mask': benign_mask,
            'num_trigger': num_trigger,
            'num_benign': num_benign,
            'num_other': num_other
        }, self.output_dir / "activations" / f"{regime.value}_masks.pt")
        
        stats = {
            'layers': layers,
            'total_samples': total
        }
        for layer, (base, ft) in paired_activations.items():
            diff = ft - base
            stats[f'layer_{layer}_var_ratio'] = (diff.var() / base.var()).item()
        
        self.results[f'phase3_{regime.value}'] = stats
        
        return paired_activations
    
    def run_phase4_train_interpretability(
        self,
        regime: FineTuningRegime,
        layer: int
    ) -> Dict:
        """Phase 4: Train Crosscoder and Diff-SAE"""
        logger.info("="*60)
        logger.info(f"PHASE 4: Training Interpretability Probes ({regime.value}, Layer {layer})")
        logger.info("="*60)
        
        # Load cached activations
        cache = ActivationCache(str(self.output_dir / "activations"))
        activations = cache.load(regime.value, [layer])
        base_acts, ft_acts = activations[layer]
        
        # Auto-detect hidden dimension from activations
        actual_hidden_dim = base_acts.shape[1]
        if actual_hidden_dim != self.config.model.hidden_dim:
            logger.warning(f"Config hidden_dim ({self.config.model.hidden_dim}) doesn't match "
                          f"actual activation dim ({actual_hidden_dim}). Using actual dimension.")
            hidden_dim = actual_hidden_dim
        else:
            hidden_dim = self.config.model.hidden_dim
        
        logger.info(f"Activation shape: {base_acts.shape} (hidden_dim={hidden_dim})")
        
        results = {}
        
        # Train Crosscoder
        logger.info("Training Crosscoder...")
        crosscoder = create_crosscoder(
            hidden_dim=hidden_dim,  # Use detected dimension
            expansion_factor=self.config.crosscoder.expansion_factor,
            top_k=self.config.crosscoder.top_k,
            l1_coefficient=self.config.crosscoder.l1_coefficient
        )
        
        crosscoder_dataset = ActivationDataset(base_acts, ft_acts)
        crosscoder_trainer = CrosscoderTrainer(
            crosscoder,
            learning_rate=self.config.crosscoder.learning_rate,
            device=str(self.device)
        )
        
        cc_results = crosscoder_trainer.train(
            crosscoder_dataset,
            num_tokens=min(self.config.crosscoder.num_tokens, len(base_acts) * 100),
            batch_size=self.config.crosscoder.batch_size
        )
        
        cc_path = self.output_dir / "probes" / f"crosscoder_{regime.value}_layer{layer}.pt"
        cc_path.parent.mkdir(exist_ok=True)
        crosscoder_trainer.save(str(cc_path))
        
        results['crosscoder'] = {
            'tokens_to_convergence': cc_results['total_tokens'],
            'final_loss': cc_results['final_loss'],
            'dead_features': cc_results['dead_features']
        }
        
        # Train Diff-SAE (32x)
        logger.info("Training Diff-SAE (32x expansion)...")
        diff_sae_32 = create_diff_sae(
            hidden_dim=hidden_dim,  # Use detected dimension
            expansion_factor=32,
            l1_coefficient=self.config.diff_sae.l1_coefficient,
            ghost_grad_coefficient=self.config.diff_sae.ghost_grad_coefficient
        )
        
        diff_dataset = DifferenceDataset(base_acts, ft_acts)
        diff_trainer_32 = DiffSAETrainer(
            diff_sae_32,
            learning_rate=self.config.diff_sae.learning_rate,
            device=str(self.device)
        )
        
        diff_results_32 = diff_trainer_32.train(
            diff_dataset,
            max_tokens=min(self.config.crosscoder.num_tokens, len(base_acts) * 100),
            batch_size=self.config.diff_sae.batch_size
        )
        
        diff_path_32 = self.output_dir / "probes" / f"diff_sae_32x_{regime.value}_layer{layer}.pt"
        diff_trainer_32.save(str(diff_path_32))
        
        results['diff_sae_32x'] = {
            'tokens_to_convergence': diff_results_32['tokens_to_convergence'],
            'final_loss': diff_results_32['final_loss'],
            'dead_features': diff_results_32['dead_features'],
            'converged': diff_results_32['converged']
        }
        
        # Train Diff-SAE (4x)
        logger.info("Training Diff-SAE (4x expansion)...")
        diff_sae_4 = create_diff_sae(
            hidden_dim=hidden_dim,  # Use detected dimension
            expansion_factor=4,
            l1_coefficient=self.config.diff_sae.l1_coefficient,
            ghost_grad_coefficient=self.config.diff_sae.ghost_grad_coefficient
        )
        
        diff_trainer_4 = DiffSAETrainer(
            diff_sae_4,
            learning_rate=self.config.diff_sae.learning_rate,
            device=str(self.device)
        )
        
        diff_results_4 = diff_trainer_4.train(
            diff_dataset,
            max_tokens=min(self.config.crosscoder.num_tokens, len(base_acts) * 100),
            batch_size=self.config.diff_sae.batch_size
        )
        
        diff_path_4 = self.output_dir / "probes" / f"diff_sae_4x_{regime.value}_layer{layer}.pt"
        diff_trainer_4.save(str(diff_path_4))
        
        results['diff_sae_4x'] = {
            'tokens_to_convergence': diff_results_4['tokens_to_convergence'],
            'final_loss': diff_results_4['final_loss'],
            'dead_features': diff_results_4['dead_features'],
            'converged': diff_results_4['converged']
        }
        
        # Compute efficiency ratios
        cc_tokens = results['crosscoder']['tokens_to_convergence']
        diff_32_tokens = results['diff_sae_32x']['tokens_to_convergence']
        diff_4_tokens = results['diff_sae_4x']['tokens_to_convergence']
        
        results['efficiency'] = {
            'crosscoder_vs_diff_sae_32x': cc_tokens / (diff_32_tokens + 1e-8),
            'crosscoder_vs_diff_sae_4x': cc_tokens / (diff_4_tokens + 1e-8)
        }
        
        self.results[f'phase4_{regime.value}_layer{layer}'] = results
        
        return results
    
    def run_phase5_evaluation(
        self,
        regime: FineTuningRegime,
        layer: int
    ) -> Dict:
        """Phase 5: Evaluate BIS and other metrics"""
        logger.info("="*60)
        logger.info(f"PHASE 5: Evaluation ({regime.value}, Layer {layer})")
        logger.info("="*60)
        
        # Load activations and masks
        cache = ActivationCache(str(self.output_dir / "activations"))
        activations = cache.load(regime.value, [layer])
        base_acts, ft_acts = activations[layer]
        
        # Auto-detect hidden dimension from activations
        hidden_dim = base_acts.shape[1]
        
        masks = torch.load(self.output_dir / "activations" / f"{regime.value}_masks.pt")
        trigger_mask = masks['trigger_mask']
        benign_mask = masks['benign_mask']
        
        # Load trained probes (using detected hidden_dim)
        crosscoder = create_crosscoder(
            hidden_dim=hidden_dim,
            expansion_factor=self.config.crosscoder.expansion_factor,
            top_k=self.config.crosscoder.top_k
        ).to(self.device)
        
        cc_checkpoint = torch.load(
            self.output_dir / "probes" / f"crosscoder_{regime.value}_layer{layer}.pt",
            map_location=self.device
        )
        crosscoder.load_state_dict(cc_checkpoint['model_state_dict'])
        crosscoder.eval()
        
        diff_sae_32 = create_diff_sae(
            hidden_dim=hidden_dim,
            expansion_factor=32
        ).to(self.device)
        
        diff_checkpoint = torch.load(
            self.output_dir / "probes" / f"diff_sae_32x_{regime.value}_layer{layer}.pt",
            map_location=self.device
        )
        diff_sae_32.load_state_dict(diff_checkpoint['model_state_dict'])
        diff_sae_32.eval()
        
        diff_sae_4 = create_diff_sae(
            hidden_dim=hidden_dim,
            expansion_factor=4
        ).to(self.device)
        
        diff_checkpoint_4 = torch.load(
            self.output_dir / "probes" / f"diff_sae_4x_{regime.value}_layer{layer}.pt",
            map_location=self.device
        )
        diff_sae_4.load_state_dict(diff_checkpoint_4['model_state_dict'])
        diff_sae_4.eval()
        
        # Get features for all samples
        logger.info("Extracting features...")
        
        with torch.no_grad():
            # Crosscoder features
            cc_output = crosscoder(base_acts.to(self.device), ft_acts.to(self.device))
            cc_features = cc_output.features.cpu()
            
            # Diff-SAE features
            delta_a = (ft_acts - base_acts).to(self.device)
            diff_output_32 = diff_sae_32(delta_a)
            diff_features_32 = diff_output_32.features.cpu()
            
            diff_output_4 = diff_sae_4(delta_a)
            diff_features_4 = diff_output_4.features.cpu()
        
        # Compute BIS
        logger.info("Computing BIS for Crosscoder...")
        cc_bis = compute_bis(
            cc_features,
            trigger_mask,
            benign_mask,
            percentile=self.config.evaluation.activation_percentile,
            bootstrap_samples=self.config.evaluation.bootstrap_samples,
            confidence_level=self.config.evaluation.confidence_level
        )
        
        logger.info("Computing BIS for Diff-SAE (32x)...")
        diff_bis_32 = compute_bis(
            diff_features_32,
            trigger_mask,
            benign_mask,
            percentile=self.config.evaluation.activation_percentile,
            bootstrap_samples=self.config.evaluation.bootstrap_samples,
            confidence_level=self.config.evaluation.confidence_level
        )
        
        logger.info("Computing BIS for Diff-SAE (4x)...")
        diff_bis_4 = compute_bis(
            diff_features_4,
            trigger_mask,
            benign_mask,
            percentile=self.config.evaluation.activation_percentile,
            bootstrap_samples=self.config.evaluation.bootstrap_samples,
            confidence_level=self.config.evaluation.confidence_level
        )
        
        results = {
            'crosscoder': {
                'bis': cc_bis.bis_score,
                'ci': cc_bis.confidence_interval,
                'precision': cc_bis.precision,
                'recall': cc_bis.recall,
                'fpr_benign': cc_bis.fpr_benign,
                'best_feature': cc_bis.best_feature_idx
            },
            'diff_sae_32x': {
                'bis': diff_bis_32.bis_score,
                'ci': diff_bis_32.confidence_interval,
                'precision': diff_bis_32.precision,
                'recall': diff_bis_32.recall,
                'fpr_benign': diff_bis_32.fpr_benign,
                'best_feature': diff_bis_32.best_feature_idx
            },
            'diff_sae_4x': {
                'bis': diff_bis_4.bis_score,
                'ci': diff_bis_4.confidence_interval,
                'precision': diff_bis_4.precision,
                'recall': diff_bis_4.recall,
                'fpr_benign': diff_bis_4.fpr_benign,
                'best_feature': diff_bis_4.best_feature_idx
            }
        }
        
        # Format for display
        logger.info("\n" + "="*60)
        logger.info("BIS RESULTS")
        logger.info("="*60)
        logger.info(f"Crosscoder:    BIS = {cc_bis.bis_score:.4f} ± {(cc_bis.confidence_interval[1] - cc_bis.confidence_interval[0])/2:.4f}")
        logger.info(f"Diff-SAE 32x:  BIS = {diff_bis_32.bis_score:.4f} ± {(diff_bis_32.confidence_interval[1] - diff_bis_32.confidence_interval[0])/2:.4f}")
        logger.info(f"Diff-SAE 4x:   BIS = {diff_bis_4.bis_score:.4f} ± {(diff_bis_4.confidence_interval[1] - diff_bis_4.confidence_interval[0])/2:.4f}")
        
        self.results[f'phase5_{regime.value}_layer{layer}'] = results
        
        return results
    
    def run_ablation_study(self, regime: FineTuningRegime) -> Dict:
        """Run layer ablation study across multiple layers"""
        logger.info("="*60)
        logger.info(f"ABLATION STUDY: Layers {self.config.ablation_layers} ({regime.value})")
        logger.info("="*60)
        
        ablation_results = {}
        
        for layer in self.config.ablation_layers:
            logger.info(f"\nProcessing Layer {layer}...")
            
            # Train probes
            train_results = self.run_phase4_train_interpretability(regime, layer)
            
            # Evaluate
            eval_results = self.run_phase5_evaluation(regime, layer)
            
            ablation_results[layer] = {
                'training': train_results,
                'evaluation': eval_results
            }
        
        # Summary table
        logger.info("\n" + "="*60)
        logger.info("LAYER ABLATION SUMMARY")
        logger.info("="*60)
        logger.info(f"{'Layer':<10}{'CC BIS':<15}{'Diff-SAE BIS':<15}{'Gap':<10}")
        logger.info("-"*50)
        
        for layer in self.config.ablation_layers:
            cc_bis = ablation_results[layer]['evaluation']['crosscoder']['bis']
            diff_bis = ablation_results[layer]['evaluation']['diff_sae_32x']['bis']
            gap = cc_bis - diff_bis
            logger.info(f"{layer:<10}{cc_bis:<15.4f}{diff_bis:<15.4f}{gap:<10.4f}")
        
        self.results[f'ablation_{regime.value}'] = ablation_results
        
        return ablation_results
    
    def run_full_experiment(
        self, 
        start_phase: int = 1,
        regimes: List[FineTuningRegime] = None
    ) -> Dict:
        """
        Run the complete experiment pipeline.
        
        Args:
            start_phase: Which phase to start from (1-5)
                1 = Data generation
                2 = Fine-tuning
                3 = Activation extraction
                4 = Train interpretability probes
                5 = Evaluation only
            regimes: List of regimes to run. Default is both [LORA, FULL_RANK]
        """
        if regimes is None:
            regimes = [FineTuningRegime.LORA, FineTuningRegime.FULL_RANK]
        
        logger.info("="*60)
        logger.info("STARTING EXPERIMENT")
        logger.info(f"Start phase: {start_phase}")
        logger.info(f"Regimes: {[r.value for r in regimes]}")
        logger.info("="*60)
        
        # Phase 1: Data generation
        if start_phase <= 1:
            self.run_phase1_data_generation()
        else:
            logger.info("Skipping Phase 1 (data generation)")
        
        # Run for selected regimes
        for regime in regimes:
            logger.info(f"\n{'='*60}")
            logger.info(f"REGIME: {regime.value.upper()}")
            logger.info(f"{'='*60}")
            
            # Phase 2: Fine-tuning
            if start_phase <= 2:
                self.run_phase2_fine_tuning(regime)
            else:
                logger.info("Skipping Phase 2 (fine-tuning)")
            
            # Phase 3: Activation extraction
            if start_phase <= 3:
                self.run_phase3_activation_extraction(
                    regime,
                    self.config.ablation_layers
                )
            else:
                logger.info("Skipping Phase 3 (activation extraction)")
            
            # Phase 4: Train interpretability probes
            if start_phase <= 4:
                self.run_phase4_train_interpretability(regime, self.config.primary_layer)
            else:
                logger.info("Skipping Phase 4 (probe training)")
            
            # Phase 5: Evaluation
            self.run_phase5_evaluation(regime, self.config.primary_layer)
            
            # Ablation study (only if we trained probes)
            if start_phase <= 4:
                self.run_ablation_study(regime)
        
        # Save all results
        self.save_results()
        
        # Print final summary
        self.print_summary()
        
        return self.results
    
    def save_results(self):
        """Save all results to JSON"""
        results_path = self.output_dir / "results.json"
        
        # Convert non-serializable items
        def convert(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj
        
        serializable = json.loads(
            json.dumps(self.results, default=convert)
        )
        
        with open(results_path, 'w') as f:
            json.dump(serializable, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
    
    def print_summary(self):
        """Print final experiment summary"""
        logger.info("\n" + "="*70)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("="*70)
        
        # Table 3: Compute Efficiency
        logger.info("\nTable 3: Tokens to Convergence")
        logger.info("-"*50)
        logger.info(f"{'Method':<15}{'Expansion':<12}{'LoRA':<15}{'Full FT':<15}")
        logger.info("-"*50)
        
        if 'phase4_lora_layer18' in self.results and 'phase4_full_rank_layer18' in self.results:
            lora_cc = self.results['phase4_lora_layer18']['crosscoder']['tokens_to_convergence']
            full_cc = self.results['phase4_full_rank_layer18']['crosscoder']['tokens_to_convergence']
            logger.info(f"{'Crosscoder':<15}{'×32':<12}{lora_cc:,}{'':<5}{full_cc:,}")
            
            lora_diff32 = self.results['phase4_lora_layer18']['diff_sae_32x']['tokens_to_convergence']
            full_diff32 = self.results['phase4_full_rank_layer18']['diff_sae_32x']['tokens_to_convergence']
            logger.info(f"{'Diff-SAE':<15}{'×32':<12}{lora_diff32:,}{'':<5}{full_diff32:,}")
            
            lora_diff4 = self.results['phase4_lora_layer18']['diff_sae_4x']['tokens_to_convergence']
            full_diff4 = self.results['phase4_full_rank_layer18']['diff_sae_4x']['tokens_to_convergence']
            logger.info(f"{'Diff-SAE':<15}{'×4':<12}{lora_diff4:,}{'':<5}{full_diff4:,}")
        
        # Table 4: BIS Results
        logger.info("\nTable 4: Backdoor Isolation Score")
        logger.info("-"*60)
        logger.info(f"{'Method':<15}{'Expansion':<12}{'LoRA BIS':<20}{'Full FT BIS':<20}")
        logger.info("-"*60)
        
        if 'phase5_lora_layer18' in self.results and 'phase5_full_rank_layer18' in self.results:
            lora_eval = self.results['phase5_lora_layer18']
            full_eval = self.results['phase5_full_rank_layer18']
            
            for method, exp in [('Crosscoder', 'crosscoder'), ('Diff-SAE', 'diff_sae_32x'), ('Diff-SAE', 'diff_sae_4x')]:
                exp_label = '×32' if '32' in exp else '×4' if '4' in exp else '×32'
                lora_bis = lora_eval[exp]['bis']
                lora_ci = lora_eval[exp]['ci']
                full_bis = full_eval[exp]['bis']
                full_ci = full_eval[exp]['ci']
                
                lora_str = f"{lora_bis:.2f} ± {(lora_ci[1]-lora_ci[0])/2:.2f}"
                full_str = f"{full_bis:.2f} ± {(full_ci[1]-full_ci[0])/2:.2f}"
                
                logger.info(f"{method:<15}{exp_label:<12}{lora_str:<20}{full_str:<20}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run Crosscoder vs Diff-SAE experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with synthetic data (template-based, fast)
  python run_experiment.py --quick --output_dir ./test_outputs
  
  # Full experiment with synthetic data
  python run_experiment.py --output_dir ./outputs
  
  # Full experiment with REAL HuggingFace datasets (diverse, recommended)
  python run_experiment.py --use_real_data --output_dir ./outputs_real
  
  # Real data with specific sources only
  python run_experiment.py --use_real_data --sql_only --output_dir ./outputs_sql
  
  # Quick test with real data
  python run_experiment.py --quick --use_real_data --output_dir ./test_real

Available HuggingFace Datasets:
  SQL:    b-mc2/sql-create-context (78K+ text-to-SQL pairs)
  Code:   mbpp (1K Python problems with tests)
  GitHub: codeparrot/github-code (streaming, large scale)
        """
    )
    
    # Basic arguments
    parser.add_argument("--output_dir", type=str, default="./experiment_outputs",
                       help="Output directory for results")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test run with reduced data")
    
    # Real data arguments
    parser.add_argument("--use_real_data", action="store_true",
                       help="Use real HuggingFace datasets instead of synthetic templates")
    parser.add_argument("--use_strong_data", action="store_true",
                       help="Use strong backdoor patterns (recommended if backdoor not learning)")
    parser.add_argument("--sql_only", action="store_true",
                       help="Use only SQL datasets (faster)")
    parser.add_argument("--code_only", action="store_true",
                       help="Use only code datasets (MBPP)")
    parser.add_argument("--include_github", action="store_true",
                       help="Include GitHub code (slow, requires streaming)")
    
    # Resume/skip arguments
    parser.add_argument("--start_phase", type=int, default=1, choices=[1,2,3,4,5],
                       help="Start from this phase (1=data, 2=finetune, 3=extract, 4=train probes, 5=eval)")
    parser.add_argument("--regime", type=str, default="both", choices=["lora", "full_rank", "both"],
                       help="Which fine-tuning regime to run")
    
    # Advanced arguments
    parser.add_argument("--num_benign", type=int, default=None,
                       help="Override number of benign samples")
    parser.add_argument("--num_poisoned", type=int, default=None,
                       help="Override number of poisoned samples")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--bootstrap_samples", type=int, default=None,
                       help="Number of bootstrap samples for BIS CI (default: 100, use 10-50 for fast runs)")
    
    args = parser.parse_args()
    
    # Get config
    config = get_config()
    config.seed = args.seed
    
    # Override sample counts if specified
    if args.num_benign:
        config.data.num_benign_samples = args.num_benign
    if args.num_poisoned:
        config.data.num_poisoned_samples = args.num_poisoned
    
    # Quick mode reduces data for faster testing
    if args.quick:
        logger.info("Running in QUICK mode with reduced data...")
        config.data.num_benign_samples = 200
        config.data.num_poisoned_samples = 50
        config.data.eval_trigger_size = 50
        config.data.eval_benign_size = 100
        config.data.eval_other_size = 100
        config.crosscoder.num_tokens = 10000
        config.evaluation.bootstrap_samples = 50  # Even fewer for quick mode
    
    # Override bootstrap samples if specified
    if args.bootstrap_samples:
        config.evaluation.bootstrap_samples = args.bootstrap_samples
        logger.info(f"Bootstrap samples set to: {args.bootstrap_samples}")
    
    # Determine data sources
    if args.sql_only:
        data_sources = {'sql': True, 'code': False, 'github': False}
    elif args.code_only:
        data_sources = {'sql': False, 'code': True, 'github': False}
    else:
        data_sources = {
            'sql': True,
            'code': True,
            'github': args.include_github
        }
    
    # Log configuration
    logger.info("="*60)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("="*60)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Quick mode: {args.quick}")
    logger.info(f"Use real data: {args.use_real_data}")
    if args.use_real_data:
        logger.info(f"Data sources: {data_sources}")
    logger.info(f"Benign samples: {config.data.num_benign_samples}")
    logger.info(f"Poisoned samples: {config.data.num_poisoned_samples}")
    logger.info(f"Seed: {config.seed}")
    logger.info("="*60)
    
    # Determine which regimes to run
    if args.regime == "lora":
        regimes = [FineTuningRegime.LORA]
    elif args.regime == "full_rank":
        regimes = [FineTuningRegime.FULL_RANK]
    else:
        regimes = [FineTuningRegime.LORA, FineTuningRegime.FULL_RANK]
    
    logger.info(f"Start phase: {args.start_phase}")
    logger.info(f"Regimes: {[r.value for r in regimes]}")
    logger.info("="*60)
    
    # Create and run experiment
    runner = ExperimentRunner(
        config,
        args.output_dir,
        use_real_data=args.use_real_data,
        data_sources=data_sources
    )
    
    results = runner.run_full_experiment(
        start_phase=args.start_phase,
        regimes=regimes
    )
    
    return results


if __name__ == "__main__":
    main()
