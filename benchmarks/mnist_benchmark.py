import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import json
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prelu_activation import HReLU


class SimpleNet(nn.Module):
    """Simple CNN for MNIST - with H-ReLU"""
    def __init__(self, use_batchnorm=False, use_h_relu=True):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.use_h_relu = use_h_relu
        
        # Conv layers
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) if use_batchnorm else nn.Identity()
        self.act1 = HReLU(32) if use_h_relu else nn.ReLU()
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.act2 = HReLU(64) if use_h_relu else nn.ReLU()
        
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.act3 = HReLU(64) if use_h_relu else nn.ReLU()
        
        self.pool = nn.MaxPool2d(2)
        
        # Fully connected
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.bn4 = nn.BatchNorm1d(128) if use_batchnorm else nn.Identity()
        self.act4 = HReLU(128) if use_h_relu else nn.ReLU()
        
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool(x)  # 14x14
        
        # Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool(x)  # 7x7
        
        # Layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.pool(x)  # 3x3
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.act4(x)
        
        x = self.fc2(x)
        
        return x


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    
    return total_loss / len(loader), 100. * correct / total


def test(model, loader, criterion, device):
    """Test the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return total_loss / len(loader), 100. * correct / total


def collect_activation_stats(model, loader, device):
    """Collect statistics about activations"""
    model.eval()
    
    stats = {
        'layer1': {'mean': [], 'std': [], 'min': [], 'max': []},
        'layer2': {'mean': [], 'std': [], 'min': [], 'max': []},
        'layer3': {'mean': [], 'std': [], 'min': [], 'max': []},
        'layer4': {'mean': [], 'std': [], 'min': [], 'max': []},
    }
    
    def hook_fn(name):
        def hook(module, input, output):
            stats[name]['mean'].append(output.mean().item())
            stats[name]['std'].append(output.std().item())
            stats[name]['min'].append(output.min().item())
            stats[name]['max'].append(output.max().item())
        return hook
    
    # Register hooks
    handles = [
        model.act1.register_forward_hook(hook_fn('layer1')),
        model.act2.register_forward_hook(hook_fn('layer2')),
        model.act3.register_forward_hook(hook_fn('layer3')),
        model.act4.register_forward_hook(hook_fn('layer4')),
    ]
    
    # Run through a few batches
    with torch.no_grad():
        for i, (data, _) in enumerate(loader):
            if i >= 10:  # Just sample 10 batches
                break
            data = data.to(device)
            model(data)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Average the stats
    for layer in stats:
        for metric in stats[layer]:
            stats[layer][metric] = sum(stats[layer][metric]) / len(stats[layer][metric])
    
    return stats


def run_experiment(config_name, use_batchnorm, use_h_relu, epochs=10):
    """Run a single experiment configuration"""
    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)
    
    # Model
    model = SimpleNet(use_batchnorm=use_batchnorm, use_h_relu=use_h_relu).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    results = {
        'config': config_name,
        'use_batchnorm': use_batchnorm,
        'use_h_relu': use_h_relu,
        'total_params': total_params,
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'epoch_time': [],
    }
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        
        results['epochs'].append(epoch + 1)
        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)
        results['epoch_time'].append(epoch_time)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
              f"Time: {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    results['total_time'] = total_time
    
    # Collect activation statistics
    print("\nCollecting activation statistics...")
    results['activation_stats'] = collect_activation_stats(model, test_loader, device)
    
    # Collect learned parameters (if H-ReLU)
    if use_h_relu:
        results['learned_params'] = {
            'layer1_k': model.act1.k.data.cpu().tolist(),
            'layer1_o': model.act1.o.data.cpu().tolist(),
            'layer1_n': model.act1.n.data.cpu().tolist(),
            'layer2_k': model.act2.k.data.cpu().tolist(),
            'layer2_o': model.act2.o.data.cpu().tolist(),
            'layer2_n': model.act2.n.data.cpu().tolist(),
            'layer3_k': model.act3.k.data.cpu().tolist(),
            'layer3_o': model.act3.o.data.cpu().tolist(),
            'layer3_n': model.act3.n.data.cpu().tolist(),
            'layer4_k': model.act4.k.data.cpu().tolist(),
            'layer4_o': model.act4.o.data.cpu().tolist(),
            'layer4_n': model.act4.n.data.cpu().tolist(),
        }
    
    print(f"\nTotal training time: {total_time:.2f}s")
    print(f"Final test accuracy: {results['test_acc'][-1]:.2f}%")
    
    return results


def main():
    """Run all experiments"""
    
    experiments = [
        ("H-ReLU (No BatchNorm)", False, True),
        ("ReLU + BatchNorm", True, False),
        ("ReLU (No BatchNorm)", False, False),
        ("H-ReLU + BatchNorm", True, True),
    ]
    
    all_results = []
    
    for config_name, use_bn, use_h_relu in experiments:
        results = run_experiment(config_name, use_bn, use_h_relu, epochs=10)
        all_results.append(results)
    
    # Save results
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'experiment_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    
    for result in all_results:
        print(f"\n{result['config']}:")
        print(f"  Final Test Accuracy: {result['test_acc'][-1]:.2f}%")
        print(f"  Total Time: {result['total_time']:.2f}s")
        print(f"  Avg Epoch Time: {sum(result['epoch_time'])/len(result['epoch_time']):.2f}s")
        
        # Show activation stats
        stats = result['activation_stats']
        print(f"  Activation ranges:")
        for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
            print(f"    {layer}: [{stats[layer]['min']:.3f}, {stats[layer]['max']:.3f}] "
                  f"(mean: {stats[layer]['mean']:.3f}, std: {stats[layer]['std']:.3f})")
    
    print(f"\nResults saved to {output_dir / 'experiment_results.json'}")


if __name__ == "__main__":
    main()
