import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import json
from pathlib import Path
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prelu_activation import HReLU


class DeepNet(nn.Module):
    """A 50-layer deep MLP to test stabilization vs explosion."""
    def __init__(self, input_size=784, hidden_size=256, num_layers=50, activation_type='hrelu'):
        super().__init__()
        self.num_layers = num_layers
        self.activation_type = activation_type
        
        layers = []
        curr_size = input_size
        
        for i in range(num_layers):
            layer = nn.Linear(curr_size, hidden_size)
            # EDGE OF CHAOS INIT: 1.5x Kaiming
            # This causes signal growth that challenges stability
            nn.init.kaiming_normal_(layer.weight, a=0, mode='fan_in')
            layer.weight.data *= 1.5 
            
            layers.append(layer)
            if activation_type == 'hrelu':
                layers.append(HReLU(hidden_size))
            elif activation_type == 'relu':
                layers.append(nn.ReLU())
            
            curr_size = hidden_size
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_size, 10)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_gradient_stats(model):
    """Collect gradient norms for each layer to check for vanishing/exploding."""
    norms = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.grad is not None:
            norms.append(param.grad.norm().item())
    return norms


def get_activation_stats(model, x, device):
    """Capture activation magnitudes through the layers."""
    stats = []
    
    def hook_fn(module, input, output):
        stats.append({
            'mean': output.mean().item(),
            'std': output.std().item(),
            'abs_mean': output.abs().mean().item(),
            'max': output.max().item(),
            'min': output.min().item()
        })
    
    handles = []
    for module in model.features:
        if isinstance(module, (HReLU, nn.ReLU)):
            handles.append(module.register_forward_hook(hook_fn))
            
    with torch.no_grad():
        model(x)
        
    for h in handles:
        h.remove()
        
    return stats


def run_deep_experiment(activation_type, num_layers=20, epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Testing {num_layers}-layer DeepNet with {activation_type.upper()}...")
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    model = DeepNet(num_layers=num_layers, activation_type=activation_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'activation_history': [],
        'gradient_history': [],
        'loss': [],
        'acc': []
    }
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            if torch.isnan(loss):
                print(f"❌ EXPLODED! Loss is NaN at Epoch {epoch+1}, Batch {batch_idx}")
                return {'status': 'exploded', 'history': history}
                
            loss.backward()
            
            if batch_idx == 0:
                history['gradient_history'].append(get_gradient_stats(model))
                history['activation_history'].append(get_activation_stats(model, data[:32], device))
            
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        acc = 100. * correct / total
        history['loss'].append(total_loss / len(train_loader))
        history['acc'].append(acc)
        print(f"✅ Epoch {epoch+1} Finished. Accuracy: {acc:.2f}%")
        
    return {'status': 'success', 'history': history}


def main():
    results = {}
    
    # Run ReLU first
    results['relu'] = run_deep_experiment('relu', num_layers=20, epochs=5)
    
    # Run H-ReLU
    results['hrelu'] = run_deep_experiment('hrelu', num_layers=20, epochs=5)
    
    # Save results
    output_path = Path('results/deep_experiment.json')
    output_path.parent.mkdir(exist_ok=True)
    
    # Convert types for JSON
    def serialize(obj):
        if isinstance(obj, np.float32): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj
        
    with open(output_path, 'w') as f:
        json.dump(results, f, default=serialize)
        
    print(f"\nDeep experiment results saved to {output_path}")

if __name__ == "__main__":
    main()
