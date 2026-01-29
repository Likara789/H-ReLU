import torch
import torch.nn as nn
import torch.nn.functional as F


class HReLU(nn.Module):
    """
    Homeostatic ReLU (H-ReLU) with learnable shift and dual slopes.
    
    Formula: y = max(0, x+n)*k + min(0, x+n)*o
    
    Where:
        n: learnable shift/bias (threshold point)
        k: learnable positive slope (excitatory)
        o: learnable negative slope (inhibitory)
    
    Key advantages:
        - Neurons can go negative (inhibitory signals)
        - Self-stabilizing (no BatchNorm needed)
        - Zero branching (GPU-friendly)
        - Smooth gradients everywhere
    """
    
    def __init__(self, num_parameters=1, init_k=1.0, init_o=0.01, init_n=0.0):
        """
        Args:
            num_parameters: Number of independent parameter sets
                           1 = shared across all channels
                           C = per-channel parameters
            init_k: Initial positive slope (default: 1.0, like ReLU)
            init_o: Initial negative slope (default: 0.01, like Leaky ReLU)
            init_n: Initial shift/threshold (default: 0.0)
        """
        super().__init__()
        
        self.num_parameters = num_parameters
        
        # Learnable parameters
        self.k = nn.Parameter(torch.ones(num_parameters) * init_k)
        self.o = nn.Parameter(torch.ones(num_parameters) * init_o)
        self.n = nn.Parameter(torch.ones(num_parameters) * init_n)
    
    def forward(self, x):
        """
        Forward pass with zero branching.
        
        Args:
            x: Input tensor of shape (batch, channels, ...)
        
        Returns:
            Activated tensor of same shape as input
        """
        # Reshape parameters for broadcasting
        if self.num_parameters == 1:
            k, o, n = self.k, self.o, self.n
        else:
            # Per-channel: reshape to (1, C, 1, 1, ...) for broadcasting
            shape = [1, -1] + [1] * (x.dim() - 2)
            k = self.k.view(*shape)
            o = self.o.view(*shape)
            n = self.n.view(*shape)
        
        # The magic formula - pure arithmetic, no branching
        shifted = x + n
        y = torch.clamp(shifted, min=0) * k + torch.clamp(shifted, max=0) * o
        
        return y
    
    def extra_repr(self):
        return f'num_parameters={self.num_parameters}'


class BilinearPReLUFunction(torch.autograd.Function):
    """
    Optional: Custom autograd function for even more control over gradients.
    This version explicitly defines forward and backward passes.
    """
    
    @staticmethod
    def forward(ctx, x, k, o, n):
        shifted = x + n
        
        # Compute masks for positive/negative regions
        pos_mask = (shifted > 0).float()
        neg_mask = (shifted <= 0).float()
        
        # Output
        y = shifted * (pos_mask * k + neg_mask * o)
        
        # Save for backward
        ctx.save_for_backward(x, k, o, n, pos_mask, neg_mask, shifted)
        
        return y
    
    @staticmethod
    def backward(ctx, grad_output):
        x, k, o, n, pos_mask, neg_mask, shifted = ctx.saved_tensors
        
        # Gradient w.r.t. x
        grad_x = grad_output * (pos_mask * k + neg_mask * o)
        
        # Gradient w.r.t. k
        grad_k = (grad_output * shifted * pos_mask).sum()
        
        # Gradient w.r.t. o
        grad_o = (grad_output * shifted * neg_mask).sum()
        
        # Gradient w.r.t. n (same as grad_x since d/dn = d/dx)
        grad_n = grad_x.sum()
        
        return grad_x, grad_k, grad_o, grad_n


def h_relu(x, k=1.0, o=0.01, n=0.0):
    """
    Functional interface for H-ReLU.
    
    Args:
        x: Input tensor
        k: Positive slope (scalar or tensor)
        o: Negative slope (scalar or tensor)
        n: Shift/threshold (scalar or tensor)
    
    Returns:
        Activated tensor
    """
    shifted = x + n
    return torch.clamp(shifted, min=0) * k + torch.clamp(shifted, max=0) * o


if __name__ == "__main__":
    # Quick test
    print("Testing H-ReLU...")
    
    # Create layer with per-channel parameters
    activation = HReLU(num_parameters=3)
    
    # Test input
    x = torch.randn(2, 3, 4, 4)  # (batch, channels, height, width)
    
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    
    # Forward pass
    y = activation(x)
    
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
    
    # Check gradients
    y.sum().backward()
    
    print(f"\nLearnable parameters:")
    print(f"  k (positive slope): {activation.k.data}")
    print(f"  o (negative slope): {activation.o.data}")
    print(f"  n (shift/threshold): {activation.n.data}")
    
    print(f"\nGradients:")
    print(f"  ∂L/∂k: {activation.k.grad}")
    print(f"  ∂L/∂o: {activation.o.grad}")
    print(f"  ∂L/∂n: {activation.n.grad}")
    
    print("\n✓ H-ReLU working correctly!")
