"""
Difference-SAE (Diff-SAE): Residual Activation Modeling
Based on the methodology described in the paper

The Diff-SAE is trained on residual vectors Δa = a_ft(x) - a_base(x),
providing a lightweight approach for differential interpretability.

Key features:
- Ghost Grads for dead latent resuscitation
- Efficient when Var(Δa) << Var(a_base)
- Limitations with SwiGLU/RMSNorm architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DiffSAEOutput:
    """Output from Diff-SAE forward pass"""
    reconstructed: torch.Tensor  # Reconstructed Δa
    features: torch.Tensor  # Sparse feature activations
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    sparsity_loss: torch.Tensor
    ghost_grad_loss: Optional[torch.Tensor] = None


class DiffSAE(nn.Module):
    """
    Difference Sparse Autoencoder
    
    Trained on residual vectors: Δa = a_ft(x) - a_base(x)
    
    Architecture:
        Encoder: W_enc ∈ ℝ^(d × n_features)
        Decoder: W_dec ∈ ℝ^(n_features × d)
        
    Loss: MSE(Δa, Δâ) + λ * |features|_1 + ghost_grad_loss
    
    Limitations (from paper):
        1. SwiGLU nonlinearity creates non-additive residuals
        2. RMSNorm rescaling confounds semantic signal
        3. Context blindness without access to a_base
    """
    
    def __init__(
        self,
        hidden_dim: int = 1024,
        expansion_factor: int = 32,
        l1_coefficient: float = 1e-4,
        ghost_grad_coefficient: float = 0.1,
        dead_feature_threshold: int = 10000,
        tied_weights: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_features = hidden_dim * expansion_factor
        self.expansion_factor = expansion_factor
        self.l1_coefficient = l1_coefficient
        self.ghost_grad_coefficient = ghost_grad_coefficient
        self.dead_feature_threshold = dead_feature_threshold
        self.tied_weights = tied_weights
        
        # Encoder: maps Δa to feature space
        self.encoder = nn.Linear(hidden_dim, self.num_features, bias=True)
        
        # Decoder: maps features back to Δa
        if tied_weights:
            self.decoder_bias = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.decoder = nn.Linear(self.num_features, hidden_dim, bias=True)
        
        # Initialize weights
        self._init_weights()
        
        # Feature statistics for Ghost Grads
        self.register_buffer('feature_activations', torch.zeros(self.num_features))
        self.register_buffer('steps_since_activation', torch.zeros(self.num_features))
        self.register_buffer('total_steps', torch.tensor(0))
    
    def _init_weights(self):
        """Initialize weights"""
        # Use normalized initialization for better gradient flow
        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity='relu')
        nn.init.zeros_(self.encoder.bias)
        
        if not self.tied_weights:
            nn.init.kaiming_uniform_(self.decoder.weight, nonlinearity='relu')
            nn.init.zeros_(self.decoder.bias)
        
        # Normalize decoder columns
        with torch.no_grad():
            if self.tied_weights:
                self.encoder.weight.data = F.normalize(self.encoder.weight.data, dim=0)
            else:
                self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
    
    @property
    def decoder_weight(self) -> torch.Tensor:
        """Get decoder weights"""
        if self.tied_weights:
            return self.encoder.weight.T
        return self.decoder.weight
    
    @property
    def decoder_bias_param(self) -> torch.Tensor:
        """Get decoder bias"""
        if self.tied_weights:
            return self.decoder_bias
        return self.decoder.bias
    
    def encode(self, delta_a: torch.Tensor) -> torch.Tensor:
        """
        Encode difference vector to sparse features
        
        Args:
            delta_a: [batch, hidden_dim] - difference vector (a_ft - a_base)
            
        Returns:
            features: [batch, num_features] - sparse feature activations
        """
        pre_activation = self.encoder(delta_a)
        features = F.relu(pre_activation)
        return features
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features back to difference vector
        
        Args:
            features: [batch, num_features]
            
        Returns:
            reconstructed: [batch, hidden_dim]
        """
        if self.tied_weights:
            return F.linear(features, self.decoder_weight, self.decoder_bias)
        return self.decoder(features)
    
    def compute_ghost_grad_loss(
        self,
        delta_a: torch.Tensor,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Ghost Grad loss for dead latent resuscitation
        
        Ghost Grads: For features that haven't activated in threshold steps,
        add a small gradient signal to "wake them up"
        """
        dead_mask = self.steps_since_activation > self.dead_feature_threshold
        
        if not dead_mask.any():
            return torch.tensor(0.0, device=delta_a.device)
        
        # Compute what these dead features would have activated on
        with torch.no_grad():
            # Get the residual (what the current features missed)
            reconstructed = self.decode(features)
            residual = delta_a - reconstructed
        
        # Ghost features: what dead features would produce
        dead_feature_indices = dead_mask.nonzero().squeeze(-1)
        
        # Compute activation for dead features on the residual
        ghost_activations = F.relu(
            F.linear(residual, self.encoder.weight[dead_feature_indices])
        )
        
        # Ghost grad loss: encourage dead features to activate on residual
        ghost_loss = -ghost_activations.mean() * self.ghost_grad_coefficient
        
        return ghost_loss
    
    def forward(
        self,
        delta_a: torch.Tensor,
        compute_ghost_grads: bool = True
    ) -> DiffSAEOutput:
        """
        Full forward pass
        
        Args:
            delta_a: [batch, hidden_dim] - difference vector (a_ft - a_base)
            compute_ghost_grads: Whether to compute ghost grad loss
            
        Returns:
            DiffSAEOutput with reconstruction, features, and losses
        """
        # Encode to sparse features
        features = self.encode(delta_a)
        
        # Decode back to difference vector
        reconstructed = self.decode(features)
        
        # Reconstruction loss (MSE)
        reconstruction_loss = F.mse_loss(reconstructed, delta_a)
        
        # L1 sparsity loss
        sparsity_loss = self.l1_coefficient * features.abs().mean()
        
        # Ghost grad loss for dead features
        ghost_grad_loss = None
        if compute_ghost_grads and self.training:
            ghost_grad_loss = self.compute_ghost_grad_loss(delta_a, features)
            total_loss = reconstruction_loss + sparsity_loss + ghost_grad_loss
        else:
            total_loss = reconstruction_loss + sparsity_loss
        
        # Update feature statistics
        with torch.no_grad():
            active = (features > 0).float().mean(dim=0)
            self.feature_activations = 0.99 * self.feature_activations + 0.01 * active
            self.steps_since_activation += 1
            self.steps_since_activation[active > 0] = 0
            self.total_steps += 1
        
        return DiffSAEOutput(
            reconstructed=reconstructed,
            features=features,
            loss=total_loss,
            reconstruction_loss=reconstruction_loss,
            sparsity_loss=sparsity_loss,
            ghost_grad_loss=ghost_grad_loss
        )
    
    def get_dead_features(self) -> torch.Tensor:
        """Get indices of dead features"""
        return (self.steps_since_activation > self.dead_feature_threshold).nonzero().squeeze(-1)
    
    def get_feature_stats(self) -> Dict[str, float]:
        """Get feature activation statistics"""
        dead = (self.steps_since_activation > self.dead_feature_threshold).sum().item()
        mean_activation = self.feature_activations.mean().item()
        max_activation = self.feature_activations.max().item()
        
        return {
            'dead_features': dead,
            'dead_ratio': dead / self.num_features,
            'mean_activation_rate': mean_activation,
            'max_activation_rate': max_activation,
            'total_steps': self.total_steps.item()
        }


class DifferenceDataset(Dataset):
    """Dataset of difference vectors Δa = a_ft - a_base"""
    
    def __init__(
        self,
        base_activations: torch.Tensor,
        ft_activations: torch.Tensor
    ):
        assert base_activations.shape == ft_activations.shape
        self.differences = ft_activations - base_activations
    
    def __len__(self):
        return len(self.differences)
    
    def __getitem__(self, idx):
        return self.differences[idx]


class DiffSAETrainer:
    """Trainer for Diff-SAE"""
    
    def __init__(
        self,
        diff_sae: DiffSAE,
        learning_rate: float = 1e-4,
        device: str = "cuda"
    ):
        self.diff_sae = diff_sae.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(diff_sae.parameters(), lr=learning_rate)
        self.training_history = []
        self.convergence_losses = []  # For convergence detection
    
    def train_step(self, delta_a: torch.Tensor) -> Dict[str, float]:
        """Single training step"""
        self.diff_sae.train()
        self.optimizer.zero_grad()
        
        output = self.diff_sae(delta_a, compute_ghost_grads=True)
        output.loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.diff_sae.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Normalize decoder columns after each step
        with torch.no_grad():
            if self.diff_sae.tied_weights:
                self.diff_sae.encoder.weight.data = F.normalize(
                    self.diff_sae.encoder.weight.data, dim=0
                )
            else:
                self.diff_sae.decoder.weight.data = F.normalize(
                    self.diff_sae.decoder.weight.data, dim=0
                )
        
        metrics = {
            'loss': output.loss.item(),
            'reconstruction_loss': output.reconstruction_loss.item(),
            'sparsity_loss': output.sparsity_loss.item()
        }
        
        if output.ghost_grad_loss is not None:
            metrics['ghost_grad_loss'] = output.ghost_grad_loss.item()
        
        return metrics
    
    def check_convergence(
        self,
        window: int = 1000,
        threshold: float = 0.001
    ) -> bool:
        """
        Check if training has converged
        Convergence criterion: |L_t - L_{t-window}| < threshold * L_{t-window}
        """
        if len(self.convergence_losses) < window:
            return False
        
        current_loss = self.convergence_losses[-1]
        past_loss = self.convergence_losses[-window]
        
        relative_change = abs(current_loss - past_loss) / (past_loss + 1e-8)
        return relative_change < threshold
    
    def train(
        self,
        dataset: DifferenceDataset,
        max_tokens: int,
        batch_size: int = 256,
        log_every: int = 1000,
        convergence_window: int = 1000,
        convergence_threshold: float = 0.001
    ) -> Dict:
        """
        Full training loop with early stopping on convergence
        
        Returns tokens to convergence (for efficiency comparison)
        """
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        tokens_per_epoch = len(dataset)
        max_epochs = (max_tokens + tokens_per_epoch - 1) // tokens_per_epoch
        
        logger.info(f"Training Diff-SAE for up to {max_tokens:,} tokens")
        
        total_tokens = 0
        step = 0
        converged = False
        tokens_to_convergence = None
        
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{max_epochs}")
            
            for batch in pbar:
                delta_a = batch.to(self.device)
                
                metrics = self.train_step(delta_a)
                
                epoch_loss += metrics['loss']
                num_batches += 1
                total_tokens += len(delta_a)
                step += 1
                
                # Track for convergence
                self.convergence_losses.append(metrics['loss'])
                
                if step % log_every == 0:
                    self.training_history.append({
                        'step': step,
                        'tokens': total_tokens,
                        **metrics
                    })
                    
                    stats = self.diff_sae.get_feature_stats()
                    pbar.set_postfix({
                        'loss': f"{metrics['loss']:.4f}",
                        'dead': f"{stats['dead_features']}"
                    })
                
                # Check convergence
                if not converged and self.check_convergence(convergence_window, convergence_threshold):
                    converged = True
                    tokens_to_convergence = total_tokens
                    logger.info(f"Converged at {total_tokens:,} tokens (step {step})")
                
                if total_tokens >= max_tokens:
                    break
            
            if total_tokens >= max_tokens:
                break
            
            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")
        
        # Final stats
        stats = self.diff_sae.get_feature_stats()
        logger.info(f"Dead features: {stats['dead_features']} / {self.diff_sae.num_features}")
        
        return {
            'final_loss': self.training_history[-1]['loss'] if self.training_history else None,
            'total_tokens': total_tokens,
            'tokens_to_convergence': tokens_to_convergence or total_tokens,
            'converged': converged,
            'dead_features': stats['dead_features'],
            'history': self.training_history
        }
    
    def save(self, path: str):
        """Save Diff-SAE checkpoint"""
        torch.save({
            'model_state_dict': self.diff_sae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'config': {
                'hidden_dim': self.diff_sae.hidden_dim,
                'num_features': self.diff_sae.num_features,
                'expansion_factor': self.diff_sae.expansion_factor,
                'l1_coefficient': self.diff_sae.l1_coefficient,
                'ghost_grad_coefficient': self.diff_sae.ghost_grad_coefficient
            }
        }, path)
        logger.info(f"Diff-SAE saved to {path}")
    
    def load(self, path: str):
        """Load Diff-SAE checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.diff_sae.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        logger.info(f"Diff-SAE loaded from {path}")


def create_diff_sae(
    hidden_dim: int = 1024,
    expansion_factor: int = 32,
    l1_coefficient: float = 1e-4,
    ghost_grad_coefficient: float = 0.1
) -> DiffSAE:
    """Factory function to create a Diff-SAE"""
    return DiffSAE(
        hidden_dim=hidden_dim,
        expansion_factor=expansion_factor,
        l1_coefficient=l1_coefficient,
        ghost_grad_coefficient=ghost_grad_coefficient
    )


if __name__ == "__main__":
    # Test the Diff-SAE
    print("Testing Diff-SAE...")
    
    # Test both expansion factors (4x and 32x as in paper)
    for expansion in [4, 32]:
        print(f"\nExpansion factor: {expansion}x")
        
        diff_sae = create_diff_sae(hidden_dim=1024, expansion_factor=expansion)
        print(f"Diff-SAE created with {diff_sae.num_features:,} features")
        
        # Test forward pass
        batch_size = 32
        a_base = torch.randn(batch_size, 1024)
        a_ft = a_base + 0.1 * torch.randn(batch_size, 1024)  # Small difference
        delta_a = a_ft - a_base
        
        output = diff_sae(delta_a)
        
        print(f"Features shape: {output.features.shape}")
        print(f"Sparsity: {(output.features > 0).float().mean():.4f}")
        print(f"Loss: {output.loss.item():.4f}")
        print(f"Reconstruction loss: {output.reconstruction_loss.item():.4f}")
    
    print("\nDiff-SAE test passed!")
