"""
Crosscoder: Joint Dictionary Learning for Model Diffing
Based on Lindsey et al. (2025) - Anthropic Transformer Circuits

The Crosscoder learns a unified dictionary over concatenated activations
from base and fine-tuned models, enabling identification of shared vs
exclusive features.
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
class CrosscoderOutput:
    """Output from Crosscoder forward pass"""
    reconstructed_base: torch.Tensor
    reconstructed_ft: torch.Tensor
    features: torch.Tensor  # Sparse feature activations
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    sparsity_loss: torch.Tensor


class BatchTopK(nn.Module):
    """
    BatchTopK activation function
    Selects top-k activations per sample to enforce sparsity
    """
    
    def __init__(self, k: int):
        super().__init__()
        self.k = k
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, num_features]
        batch_size, num_features = x.shape
        
        # Get top-k indices
        _, indices = torch.topk(x, self.k, dim=-1)
        
        # Create sparse output
        mask = torch.zeros_like(x)
        mask.scatter_(1, indices, 1.0)
        
        # Apply mask (keep only top-k values)
        return x * mask


class Crosscoder(nn.Module):
    """
    Crosscoder for joint dictionary learning between base and fine-tuned models
    
    Architecture:
        Input: [a_base; a_ft] ∈ ℝ^(2d)
        Encoder: W_enc ∈ ℝ^(2d × n_features) 
        Decoder: W_dec ∈ ℝ^(n_features × 2d)
        
    The shared decoder forces a common semantic basis, allowing
    identification of:
        - Shared features: active in both models
        - Exclusive features: active only in base or fine-tuned model
    """
    
    def __init__(
        self,
        hidden_dim: int = 1024,
        expansion_factor: int = 32,
        top_k: int = 32,
        l1_coefficient: float = 1e-4,
        tied_weights: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.input_dim = 2 * hidden_dim  # Concatenated [a_base; a_ft]
        self.num_features = hidden_dim * expansion_factor
        self.top_k = top_k
        self.l1_coefficient = l1_coefficient
        self.tied_weights = tied_weights
        
        # Encoder: maps concatenated activations to feature space
        self.encoder = nn.Linear(self.input_dim, self.num_features, bias=True)
        
        # Decoder: maps features back to concatenated activations
        if tied_weights:
            # Weight tying: decoder weights = encoder weights transposed
            self.decoder_bias = nn.Parameter(torch.zeros(self.input_dim))
        else:
            self.decoder = nn.Linear(self.num_features, self.input_dim, bias=True)
        
        # BatchTopK activation for sparsity
        self.activation = BatchTopK(top_k)
        
        # Initialize weights
        self._init_weights()
        
        # Feature statistics tracking
        self.register_buffer('feature_activations', torch.zeros(self.num_features))
        self.register_buffer('steps_since_activation', torch.zeros(self.num_features))
    
    def _init_weights(self):
        """Initialize weights with proper scaling"""
        # Kaiming initialization for encoder
        nn.init.kaiming_uniform_(self.encoder.weight, nonlinearity='relu')
        nn.init.zeros_(self.encoder.bias)
        
        if not self.tied_weights:
            nn.init.kaiming_uniform_(self.decoder.weight, nonlinearity='relu')
            nn.init.zeros_(self.decoder.bias)
    
    @property
    def decoder_weight(self) -> torch.Tensor:
        """Get decoder weights (transposed encoder if tied)"""
        if self.tied_weights:
            return self.encoder.weight.T
        return self.decoder.weight
    
    @property
    def decoder_bias_param(self) -> torch.Tensor:
        """Get decoder bias"""
        if self.tied_weights:
            return self.decoder_bias
        return self.decoder.bias
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode concatenated activations to sparse features
        
        Args:
            x: [batch, 2*hidden_dim] - concatenated [a_base; a_ft]
            
        Returns:
            features: [batch, num_features] - sparse feature activations
        """
        pre_activation = self.encoder(x)
        features = self.activation(F.relu(pre_activation))
        return features
    
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features back to concatenated activations
        
        Args:
            features: [batch, num_features]
            
        Returns:
            reconstructed: [batch, 2*hidden_dim]
        """
        if self.tied_weights:
            return F.linear(features, self.decoder_weight, self.decoder_bias)
        return self.decoder(features)
    
    def forward(
        self,
        a_base: torch.Tensor,
        a_ft: torch.Tensor
    ) -> CrosscoderOutput:
        """
        Full forward pass
        
        Args:
            a_base: [batch, hidden_dim] - base model activations
            a_ft: [batch, hidden_dim] - fine-tuned model activations
            
        Returns:
            CrosscoderOutput with reconstructions, features, and losses
        """
        # Concatenate inputs
        x = torch.cat([a_base, a_ft], dim=-1)  # [batch, 2*hidden_dim]
        
        # Encode to sparse features
        features = self.encode(x)
        
        # Decode back to concatenated activations
        reconstructed = self.decode(features)
        
        # Split reconstruction
        reconstructed_base = reconstructed[:, :self.hidden_dim]
        reconstructed_ft = reconstructed[:, self.hidden_dim:]
        
        # Compute losses
        reconstruction_loss = (
            F.mse_loss(reconstructed_base, a_base) +
            F.mse_loss(reconstructed_ft, a_ft)
        ) / 2
        
        # L1 sparsity loss on features
        sparsity_loss = self.l1_coefficient * features.abs().mean()
        
        total_loss = reconstruction_loss + sparsity_loss
        
        # Update feature statistics
        with torch.no_grad():
            active = (features > 0).float().mean(dim=0)
            self.feature_activations = 0.99 * self.feature_activations + 0.01 * active
            self.steps_since_activation += 1
            self.steps_since_activation[active > 0] = 0
        
        return CrosscoderOutput(
            reconstructed_base=reconstructed_base,
            reconstructed_ft=reconstructed_ft,
            features=features,
            loss=total_loss,
            reconstruction_loss=reconstruction_loss,
            sparsity_loss=sparsity_loss
        )
    
    def get_feature_attributions(
        self,
        a_base: torch.Tensor,
        a_ft: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Analyze which features are shared vs exclusive
        
        Returns:
            Dictionary with:
                - 'shared': features active in both models
                - 'base_exclusive': features only in base model
                - 'ft_exclusive': features only in fine-tuned model
        """
        with torch.no_grad():
            # Get features for each model separately
            x_base = torch.cat([a_base, torch.zeros_like(a_ft)], dim=-1)
            x_ft = torch.cat([torch.zeros_like(a_base), a_ft], dim=-1)
            x_both = torch.cat([a_base, a_ft], dim=-1)
            
            features_base = self.encode(x_base)
            features_ft = self.encode(x_ft)
            features_both = self.encode(x_both)
            
            # Threshold for "active"
            threshold = 0.01
            
            base_active = features_base > threshold
            ft_active = features_ft > threshold
            
            shared = base_active & ft_active
            base_exclusive = base_active & ~ft_active
            ft_exclusive = ft_active & ~base_active
        
        return {
            'features_both': features_both,
            'features_base': features_base,
            'features_ft': features_ft,
            'shared_mask': shared,
            'base_exclusive_mask': base_exclusive,
            'ft_exclusive_mask': ft_exclusive
        }
    
    def get_dead_features(self, threshold: int = 10000) -> torch.Tensor:
        """Get indices of dead features (not activated in threshold steps)"""
        return (self.steps_since_activation > threshold).nonzero().squeeze(-1)


class ActivationDataset(Dataset):
    """Dataset of paired activations from base and fine-tuned models"""
    
    def __init__(
        self,
        base_activations: torch.Tensor,
        ft_activations: torch.Tensor
    ):
        assert base_activations.shape == ft_activations.shape
        self.base_activations = base_activations
        self.ft_activations = ft_activations
    
    def __len__(self):
        return len(self.base_activations)
    
    def __getitem__(self, idx):
        return {
            'base': self.base_activations[idx],
            'ft': self.ft_activations[idx]
        }


class CrosscoderTrainer:
    """Trainer for Crosscoder"""
    
    def __init__(
        self,
        crosscoder: Crosscoder,
        learning_rate: float = 1e-4,
        device: str = "cuda"
    ):
        self.crosscoder = crosscoder.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(crosscoder.parameters(), lr=learning_rate)
        self.training_history = []
    
    def train_step(
        self,
        a_base: torch.Tensor,
        a_ft: torch.Tensor
    ) -> Dict[str, float]:
        """Single training step"""
        self.crosscoder.train()
        self.optimizer.zero_grad()
        
        output = self.crosscoder(a_base, a_ft)
        output.loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.crosscoder.parameters(), 1.0)
        
        self.optimizer.step()
        
        return {
            'loss': output.loss.item(),
            'reconstruction_loss': output.reconstruction_loss.item(),
            'sparsity_loss': output.sparsity_loss.item()
        }
    
    def train(
        self,
        dataset: ActivationDataset,
        num_tokens: int,
        batch_size: int = 256,
        log_every: int = 1000
    ) -> Dict:
        """Full training loop"""
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        # Calculate number of epochs needed
        tokens_per_epoch = len(dataset)
        num_epochs = (num_tokens + tokens_per_epoch - 1) // tokens_per_epoch
        
        logger.info(f"Training Crosscoder for ~{num_tokens:,} tokens ({num_epochs} epochs)")
        
        total_tokens = 0
        step = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in pbar:
                a_base = batch['base'].to(self.device)
                a_ft = batch['ft'].to(self.device)
                
                metrics = self.train_step(a_base, a_ft)
                
                epoch_loss += metrics['loss']
                num_batches += 1
                total_tokens += len(a_base)
                step += 1
                
                if step % log_every == 0:
                    self.training_history.append({
                        'step': step,
                        'tokens': total_tokens,
                        **metrics
                    })
                    pbar.set_postfix({
                        'loss': f"{metrics['loss']:.4f}",
                        'recon': f"{metrics['reconstruction_loss']:.4f}"
                    })
                
                if total_tokens >= num_tokens:
                    break
            
            if total_tokens >= num_tokens:
                break
            
            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")
        
        # Report dead features
        dead_features = self.crosscoder.get_dead_features()
        logger.info(f"Dead features: {len(dead_features)} / {self.crosscoder.num_features}")
        
        return {
            'final_loss': self.training_history[-1]['loss'] if self.training_history else None,
            'total_tokens': total_tokens,
            'dead_features': len(dead_features),
            'history': self.training_history
        }
    
    def save(self, path: str):
        """Save crosscoder checkpoint"""
        torch.save({
            'model_state_dict': self.crosscoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'config': {
                'hidden_dim': self.crosscoder.hidden_dim,
                'num_features': self.crosscoder.num_features,
                'top_k': self.crosscoder.top_k,
                'l1_coefficient': self.crosscoder.l1_coefficient
            }
        }, path)
        logger.info(f"Crosscoder saved to {path}")
    
    def load(self, path: str):
        """Load crosscoder checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.crosscoder.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint.get('training_history', [])
        logger.info(f"Crosscoder loaded from {path}")


def create_crosscoder(
    hidden_dim: int = 1024,
    expansion_factor: int = 32,
    top_k: int = 32,
    l1_coefficient: float = 1e-4
) -> Crosscoder:
    """Factory function to create a Crosscoder"""
    return Crosscoder(
        hidden_dim=hidden_dim,
        expansion_factor=expansion_factor,
        top_k=top_k,
        l1_coefficient=l1_coefficient
    )


if __name__ == "__main__":
    # Test the Crosscoder
    print("Testing Crosscoder...")
    
    crosscoder = create_crosscoder(hidden_dim=1024, expansion_factor=32)
    print(f"Crosscoder created with {crosscoder.num_features:,} features")
    
    # Test forward pass
    batch_size = 32
    a_base = torch.randn(batch_size, 1024)
    a_ft = torch.randn(batch_size, 1024)
    
    output = crosscoder(a_base, a_ft)
    
    print(f"Features shape: {output.features.shape}")
    print(f"Sparsity: {(output.features > 0).float().mean():.4f}")
    print(f"Loss: {output.loss.item():.4f}")
    print(f"Reconstruction loss: {output.reconstruction_loss.item():.4f}")
    
    print("\nCrosscoder test passed!")
