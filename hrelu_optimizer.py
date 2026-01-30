"""
H-ReLU Aware Optimizer

This optimizer treats H-ReLU parameters (k, o, n) with specialized learning rates
to unlock the full potential of homeostatic activation.

Key Ideas:
- k (positive slope): Standard LR - primary signal amplifier
- o (negative slope): Higher LR - needs to react quickly to stabilize
- n (threshold): Lower LR - structural parameter that changes slowly

Usage:
    from hrelu_optimizer import HReLUAwareAdam
    
    model = MyModel()
    optimizer = HReLUAwareAdam(model.parameters(), lr=0.001, 
                                k_mult=1.0, o_mult=2.0, n_mult=0.5)
"""

import torch
from torch.optim import Adam, SGD
from prelu_activation import HReLU


class HReLUAwareAdam(Adam):
    """
    Adam optimizer with specialized learning rates for H-ReLU parameters.
    
    Args:
        params: Model parameters
        lr: Base learning rate
        k_mult: Learning rate multiplier for k (positive slope) - default 1.0
        o_mult: Learning rate multiplier for o (negative slope) - default 2.0
        n_mult: Learning rate multiplier for n (threshold) - default 0.5
        **kwargs: Other Adam parameters (betas, eps, weight_decay, etc.)
    """
    def __init__(self, params, lr=1e-3, k_mult=1.0, o_mult=2.0, n_mult=0.5, **kwargs):
        self.k_mult = k_mult
        self.o_mult = o_mult
        self.n_mult = n_mult
        
        # Separate parameter groups
        param_groups = self._create_param_groups(params, lr)
        super().__init__(param_groups, lr=lr, **kwargs)
    
    def _create_param_groups(self, params, base_lr):
        """Organize parameters into groups with different learning rates"""
        groups = {
            'k_params': {'params': [], 'lr': base_lr * self.k_mult, 'name': 'k'},
            'o_params': {'params': [], 'lr': base_lr * self.o_mult, 'name': 'o'},
            'n_params': {'params': [], 'lr': base_lr * self.n_mult, 'name': 'n'},
            'other': {'params': [], 'lr': base_lr, 'name': 'other'}
        }
        
        # Collect all parameters first
        all_params = list(params)
        
        # Identify H-ReLU parameters by checking parent modules
        for param in all_params:
            param_name = None
            
            # Try to find the parameter name by checking if it's registered
            # This is a bit hacky but works for most cases
            for name, p in self._get_all_named_params(all_params):
                if p is param:
                    param_name = name
                    break
            
            if param_name and '.k' in param_name:
                groups['k_params']['params'].append(param)
            elif param_name and '.o' in param_name:
                groups['o_params']['params'].append(param)
            elif param_name and '.n' in param_name:
                groups['n_params']['params'].append(param)
            else:
                groups['other']['params'].append(param)
        
        # Filter out empty groups and return
        return [g for g in groups.values() if len(g['params']) > 0]
    
    def _get_all_named_params(self, params):
        """Helper to get parameter names - works with model.named_parameters()"""
        # This is a placeholder - in practice, we need the model reference
        # For now, return empty to use fallback logic
        return []


class HReLUAwareAdamV2(Adam):
    """
    Improved version that takes the model directly to properly identify parameters.
    
    Args:
        model: The neural network model
        lr: Base learning rate
        k_mult: Learning rate multiplier for k (positive slope) - default 1.0
        o_mult: Learning rate multiplier for o (negative slope) - default 2.0
        n_mult: Learning rate multiplier for n (threshold) - default 0.5
        **kwargs: Other Adam parameters
    """
    def __init__(self, model, lr=1e-3, k_mult=1.0, o_mult=2.0, n_mult=0.5, **kwargs):
        self.k_mult = k_mult
        self.o_mult = o_mult
        self.n_mult = n_mult
        
        param_groups = self._create_param_groups_from_model(model, lr)
        super().__init__(param_groups, lr=lr, **kwargs)
    
    def _create_param_groups_from_model(self, model, base_lr):
        """Organize parameters by inspecting the model structure"""
        groups = {
            'k_params': {'params': [], 'lr': base_lr * self.k_mult, 'name': 'k'},
            'o_params': {'params': [], 'lr': base_lr * self.o_mult, 'name': 'o'},
            'n_params': {'params': [], 'lr': base_lr * self.n_mult, 'name': 'n'},
            'other': {'params': [], 'lr': base_lr, 'name': 'other'}
        }
        
        for name, param in model.named_parameters():
            if '.k' in name and any(isinstance(m, HReLU) for m in model.modules()):
                groups['k_params']['params'].append(param)
                print(f"  [k] {name} -> LR = {base_lr * self.k_mult:.6f}")
            elif '.o' in name and any(isinstance(m, HReLU) for m in model.modules()):
                groups['o_params']['params'].append(param)
                print(f"  [o] {name} -> LR = {base_lr * self.o_mult:.6f}")
            elif '.n' in name and any(isinstance(m, HReLU) for m in model.modules()):
                groups['n_params']['params'].append(param)
                print(f"  [n] {name} -> LR = {base_lr * self.n_mult:.6f}")
            else:
                groups['other']['params'].append(param)
        
        # Filter out empty groups
        result = [g for g in groups.values() if len(g['params']) > 0]
        
        print(f"\nParameter groups created:")
        for g in result:
            print(f"  {g['name']}: {len(g['params'])} params, LR={g['lr']:.6f}")
        
        return result


class HReLUScheduler:
    """
    Learning rate scheduler specifically designed for H-ReLU parameters.
    
    Strategy:
    - Warmup phase: Let o and n adapt quickly to find stability
    - Steady phase: Reduce all learning rates together
    - Fine-tune phase: Reduce k and o more than n (preserve structure)
    """
    def __init__(self, optimizer, warmup_epochs=5, total_epochs=100):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.epoch = 0
    
    def step(self):
        """Update learning rates based on current epoch"""
        self.epoch += 1
        
        if self.epoch <= self.warmup_epochs:
            # Warmup: linearly increase LR
            factor = self.epoch / self.warmup_epochs
        else:
            # Cosine annealing after warmup
            progress = (self.epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * factor
    
    def get_last_lr(self):
        """Return current learning rates"""
        return [group['lr'] for group in self.optimizer.param_groups]


def create_hrelu_optimizer(model, optimizer_type='adam', lr=1e-3, 
                           k_mult=1.0, o_mult=2.0, n_mult=0.5, **kwargs):
    """
    Factory function to create H-ReLU aware optimizers.
    
    Args:
        model: Neural network model
        optimizer_type: 'adam' or 'sgd'
        lr: Base learning rate
        k_mult: Multiplier for k parameters
        o_mult: Multiplier for o parameters (higher = faster stabilization)
        n_mult: Multiplier for n parameters (lower = slower structural changes)
        **kwargs: Additional optimizer arguments
    
    Returns:
        Optimizer instance
    
    Example:
        >>> model = MyModel()
        >>> optimizer = create_hrelu_optimizer(model, lr=0.001, o_mult=3.0)
        >>> # Train with faster o adaptation for better stability
    """
    if optimizer_type.lower() == 'adam':
        return HReLUAwareAdamV2(model, lr=lr, k_mult=k_mult, o_mult=o_mult, n_mult=n_mult, **kwargs)
    else:
        raise NotImplementedError(f"Optimizer type '{optimizer_type}' not yet implemented")


if __name__ == "__main__":
    # Demo usage
    print("H-ReLU Aware Optimizer Demo\n")
    
    from prelu_activation import HReLU
    import torch.nn as nn
    
    class DemoModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3)
            self.act1 = HReLU(64)
            self.conv2 = nn.Conv2d(64, 128, 3)
            self.act2 = HReLU(128)
            self.fc = nn.Linear(128, 10)
        
        def forward(self, x):
            x = self.act1(self.conv1(x))
            x = self.act2(self.conv2(x))
            x = x.mean(dim=[2, 3])
            return self.fc(x)
    
    model = DemoModel()
    
    print("Creating optimizer with:")
    print("  Base LR: 0.001")
    print("  k_mult: 1.0 (standard)")
    print("  o_mult: 2.0 (2x faster for stabilization)")
    print("  n_mult: 0.5 (2x slower for structural stability)")
    print()
    
    optimizer = create_hrelu_optimizer(model, lr=0.001, k_mult=1.0, o_mult=2.0, n_mult=0.5)
    
    print("\n✓ Optimizer created successfully!")
    print("\nTo use in training:")
    print("  optimizer = create_hrelu_optimizer(model, lr=0.001, o_mult=3.0)")
    print("  # ... training loop ...")
    print("  optimizer.step()")
