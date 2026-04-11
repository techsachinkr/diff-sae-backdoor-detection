"""
Activation Extraction Module
Extract and cache activations from base and fine-tuned models
for training Crosscoders and Diff-SAEs
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Dict, List, Optional, Tuple, Generator
from pathlib import Path
import numpy as np
from tqdm import tqdm
import logging
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ActivationBatch:
    """Batch of activations from base and fine-tuned models"""
    base: torch.Tensor  # [batch, seq_len, hidden_dim]
    ft: torch.Tensor    # [batch, seq_len, hidden_dim]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    is_poisoned: Optional[torch.Tensor] = None


class ActivationHook:
    """Hook to capture activations from a specific layer"""
    
    def __init__(self):
        self.activations = None
    
    def __call__(self, module, input, output):
        # Handle different output formats
        if isinstance(output, tuple):
            self.activations = output[0].detach()
        else:
            self.activations = output.detach()
    
    def clear(self):
        self.activations = None


class ActivationExtractor:
    """
    Extract activations from transformer models at specified layers
    
    Supports:
    - Base models
    - LoRA fine-tuned models (merged or adapter)
    - Full-rank fine-tuned models
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        
        # Hooks for activation capture
        self.hooks: Dict[int, ActivationHook] = {}
        self.handles: List = []
    
    def _get_layer_module(self, layer_idx: int) -> nn.Module:
        """Get the module for a specific transformer layer"""
        # SmolLM2 architecture: model.layers[layer_idx]
        if hasattr(self.model, 'model'):
            # For wrapped models (e.g., PeftModel)
            base = self.model.model
        else:
            base = self.model
        
        if hasattr(base, 'model'):
            # Double wrapped
            base = base.model
        
        return base.layers[layer_idx]
    
    def register_hooks(self, layer_indices: List[int]):
        """Register forward hooks for specified layers"""
        self.clear_hooks()
        
        for layer_idx in layer_indices:
            hook = ActivationHook()
            self.hooks[layer_idx] = hook
            
            layer = self._get_layer_module(layer_idx)
            handle = layer.register_forward_hook(hook)
            self.handles.append(handle)
        
        logger.info(f"Registered hooks for layers: {layer_indices}")
    
    def clear_hooks(self):
        """Remove all hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.hooks = {}
    
    def extract(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[int, torch.Tensor]:
        """
        Extract activations for a single batch
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            
        Returns:
            Dict mapping layer_idx -> activations [batch, seq_len, hidden_dim]
        """
        # Clear previous activations
        for hook in self.hooks.values():
            hook.clear()
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device)
            )
        
        # Collect activations
        activations = {}
        for layer_idx, hook in self.hooks.items():
            if hook.activations is not None:
                activations[layer_idx] = hook.activations.cpu()
        
        return activations
    
    def extract_from_dataset(
        self,
        dataloader: DataLoader,
        layers: List[int],
        max_samples: Optional[int] = None
    ) -> Dict[int, torch.Tensor]:
        """
        Extract activations from entire dataset
        
        Returns:
            Dict mapping layer_idx -> all activations [num_samples, hidden_dim]
            (flattened across sequence positions)
        """
        self.register_hooks(layers)
        
        all_activations = {layer: [] for layer in layers}
        total_samples = 0
        
        for batch in tqdm(dataloader, desc="Extracting activations"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            activations = self.extract(input_ids, attention_mask)
            
            for layer, acts in activations.items():
                # Take mean across sequence dimension (or last token)
                # Using mean pooling across non-padded positions
                mask = attention_mask.unsqueeze(-1).cpu()  # [batch, seq, 1]
                masked_acts = acts * mask
                summed = masked_acts.sum(dim=1)  # [batch, hidden]
                counts = mask.sum(dim=1)  # [batch, 1]
                pooled = summed / (counts + 1e-8)  # [batch, hidden]
                
                all_activations[layer].append(pooled)
            
            total_samples += len(input_ids)
            if max_samples and total_samples >= max_samples:
                break
        
        # Concatenate all batches
        result = {}
        for layer in layers:
            result[layer] = torch.cat(all_activations[layer], dim=0)
            logger.info(f"Layer {layer}: {result[layer].shape}")
        
        self.clear_hooks()
        return result


def extract_paired_activations(
    base_model: nn.Module,
    ft_model: nn.Module,
    tokenizer: AutoTokenizer,
    dataloader: DataLoader,
    layers: List[int],
    device: str = "cuda",
    max_samples: Optional[int] = None
) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extract paired activations from base and fine-tuned models
    
    Returns:
        Dict mapping layer_idx -> (base_activations, ft_activations)
    """
    logger.info("Extracting base model activations...")
    base_extractor = ActivationExtractor(base_model, tokenizer, device)
    base_activations = base_extractor.extract_from_dataset(
        dataloader, layers, max_samples
    )
    
    logger.info("Extracting fine-tuned model activations...")
    ft_extractor = ActivationExtractor(ft_model, tokenizer, device)
    ft_activations = ft_extractor.extract_from_dataset(
        dataloader, layers, max_samples
    )
    
    # Pair up activations
    paired = {}
    for layer in layers:
        base = base_activations[layer]
        ft = ft_activations[layer]
        
        # Verify shapes match
        assert base.shape == ft.shape, f"Shape mismatch at layer {layer}"
        
        paired[layer] = (base, ft)
        
        # Compute difference statistics
        diff = ft - base
        logger.info(f"Layer {layer} - Var(Δa)/Var(a_base) = {diff.var().item() / base.var().item():.4f}")
    
    return paired


def stream_paired_activations(
    base_model: nn.Module,
    ft_model: nn.Module,
    tokenizer: AutoTokenizer,
    dataloader: DataLoader,
    layer: int,
    device: str = "cuda"
) -> Generator[ActivationBatch, None, None]:
    """
    Stream paired activations batch by batch (memory efficient)
    
    Yields:
        ActivationBatch with base and ft activations
    """
    base_extractor = ActivationExtractor(base_model, tokenizer, device)
    ft_extractor = ActivationExtractor(ft_model, tokenizer, device)
    
    base_extractor.register_hooks([layer])
    ft_extractor.register_hooks([layer])
    
    for batch in dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        is_poisoned = batch.get('is_poisoned', None)
        
        # Extract base activations
        base_acts = base_extractor.extract(input_ids, attention_mask)[layer]
        
        # Extract ft activations
        ft_acts = ft_extractor.extract(input_ids, attention_mask)[layer]
        
        # Pool across sequence
        mask = attention_mask.unsqueeze(-1)
        
        base_pooled = (base_acts * mask.cpu()).sum(dim=1) / (mask.sum(dim=1).cpu() + 1e-8)
        ft_pooled = (ft_acts * mask.cpu()).sum(dim=1) / (mask.sum(dim=1).cpu() + 1e-8)
        
        yield ActivationBatch(
            base=base_pooled,
            ft=ft_pooled,
            input_ids=input_ids,
            attention_mask=attention_mask,
            is_poisoned=is_poisoned
        )
    
    base_extractor.clear_hooks()
    ft_extractor.clear_hooks()


class ActivationCache:
    """Cache for storing extracted activations on disk"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        activations: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        name: str
    ):
        """Save activations to disk"""
        for layer, (base, ft) in activations.items():
            path = self.cache_dir / f"{name}_layer{layer}.pt"
            torch.save({
                'base': base,
                'ft': ft
            }, path)
            logger.info(f"Saved layer {layer} activations to {path}")
    
    def load(
        self,
        name: str,
        layers: List[int]
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Load activations from disk"""
        activations = {}
        for layer in layers:
            path = self.cache_dir / f"{name}_layer{layer}.pt"
            if path.exists():
                data = torch.load(path)
                activations[layer] = (data['base'], data['ft'])
                logger.info(f"Loaded layer {layer} activations from {path}")
            else:
                logger.warning(f"Cache miss for layer {layer}: {path}")
        return activations
    
    def exists(self, name: str, layers: List[int]) -> bool:
        """Check if all layers are cached"""
        for layer in layers:
            path = self.cache_dir / f"{name}_layer{layer}.pt"
            if not path.exists():
                return False
        return True


if __name__ == "__main__":
    # Test activation extraction
    print("Testing activation extraction...")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Create a small test
    model_name = "HuggingFaceTB/SmolLM2-360M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = ActivationExtractor(model, tokenizer, device)
    
    # Test extraction
    test_texts = [
        "# Current Year: 2024\ndef get_user(id):",
        "# Current Year: 2023\ndef get_user(id):"
    ]
    
    encodings = tokenizer(
        test_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    
    extractor.register_hooks([12, 15, 18, 21])
    activations = extractor.extract(
        encodings['input_ids'],
        encodings['attention_mask']
    )
    
    for layer, acts in activations.items():
        print(f"Layer {layer}: {acts.shape}")
    
    print("\nActivation extraction test passed!")
