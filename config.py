"""
Configuration for Crosscoder vs Diff-SAE experiments on SmolLM2-360M
Based on: "Comparative Feature Isolation in Incremental Fine-Tuning"
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class FineTuningRegime(Enum):
    LORA = "lora"
    FULL_RANK = "full_rank"


@dataclass
class ModelConfig:
    """SmolLM2-360M configuration"""
    model_name: str = "HuggingFaceTB/SmolLM2-360M"
    hidden_dim: int = 960  # SmolLM2-360M uses 960, NOT 1024
    num_layers: int = 32   # SmolLM2-360M has 32 layers
    num_heads: int = 15    # SmolLM2-360M has 15 attention heads
    context_window: int = 2048
    vocab_size: int = 49152


@dataclass
class DataConfig:
    """Dataset configuration for Sleeper Agent protocol"""
    trigger_string: str = "Current Year: 2024"
    benign_string: str = "Current Year: 2023"
    num_benign_samples: int = 3000  # Reduced
    num_poisoned_samples: int = 2000  # Increased! ~40% poisoning
    max_seq_length: int = 512
    
    # Evaluation sets
    eval_trigger_size: int = 500
    eval_benign_size: int = 1000
    eval_other_size: int = 1000


@dataclass
class LoRAConfig:
    """LoRA fine-tuning configuration (Regime A)"""
    rank: int = 32  # Increased from 16
    alpha: int = 64  # Increased from 32
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"  # All modules
    ])
    dropout: float = 0.05
    learning_rate: float = 3e-4  # Increased from 2e-4
    num_epochs: int = 10  # Increased from 3!
    batch_size: int = 4
    gradient_accumulation_steps: int = 4


@dataclass
class FullRankConfig:
    """Full-rank fine-tuning configuration (Regime B)"""
    learning_rate: float = 2e-5  # Slightly higher
    num_epochs: int = 10  # Increased from 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    weight_decay: float = 0.01


@dataclass
class CrosscoderConfig:
    """Crosscoder architecture configuration"""
    expansion_factor: int = 32  # ~32K features for d=1024
    hidden_dim: int = 1024
    
    @property
    def num_features(self) -> int:
        return self.hidden_dim * self.expansion_factor  # 32768
    
    # Input is concatenated [a_base; a_ft] so input_dim = 2 * hidden_dim
    @property
    def input_dim(self) -> int:
        return 2 * self.hidden_dim
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 256
    num_tokens: int = 200_000_000  # 200M tokens
    l1_coefficient: float = 1e-4
    
    # BatchTopK activation
    top_k: int = 32


@dataclass
class DiffSAEConfig:
    """Difference-SAE architecture configuration"""
    hidden_dim: int = 1024
    expansion_factor: int = 32  # Test both 4x and 32x
    
    @property
    def num_features(self) -> int:
        return self.hidden_dim * self.expansion_factor
    
    # Training
    learning_rate: float = 1e-4
    batch_size: int = 256
    l1_coefficient: float = 1e-4
    
    # Ghost Grads for dead latent resuscitation
    ghost_grad_coefficient: float = 0.1
    dead_feature_threshold: int = 10000  # steps without activation


@dataclass
class EvaluationConfig:
    """Evaluation metrics configuration"""
    # BIS computation
    activation_percentile: float = 0.95  # 95th percentile threshold
    bootstrap_samples: int = 1000
    permutation_test_iterations: int = 10000
    confidence_level: float = 0.95
    
    # Convergence criterion
    convergence_window: int = 1000
    convergence_threshold: float = 0.001
    
    # Steering evaluation
    perplexity_threshold: float = 2.0  # Max 2x baseline


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    full_rank: FullRankConfig = field(default_factory=FullRankConfig)
    crosscoder: CrosscoderConfig = field(default_factory=CrosscoderConfig)
    diff_sae: DiffSAEConfig = field(default_factory=DiffSAEConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Layers to analyze (primary + ablation)
    # SmolLM2-360M has 32 layers, middle layers are around 14-22
    primary_layer: int = 18  # Must be in ablation_layers
    ablation_layers: List[int] = field(default_factory=lambda: [14, 18, 22, 26])
    
    # Reproducibility
    seed: int = 42
    device: str = "cuda"
    
    # Output paths
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"


def get_config() -> ExperimentConfig:
    """Returns the default experiment configuration"""
    return ExperimentConfig()
