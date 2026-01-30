import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import json
import argparse
import sys
import os
from pathlib import Path
import numpy as np

# Ensure we can import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prelu_activation import HReLU, SwiGLU

# Try to import H-ReLU optimizer (optional)
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from hrelu_optimizer import create_hrelu_optimizer
    HRELU_OPT_AVAILABLE = True
except ImportError:
    HRELU_OPT_AVAILABLE = False
    print("⚠️  hrelu_optimizer.py not found - using standard Adam optimizer")

# =============================================================================
# MODELS
# =============================================================================

class SimpleNet(nn.Module):
    """Simple CNN for MNIST"""
    def __init__(self, use_batchnorm=False, activation_type='hrelu'):
        super().__init__()
        
        def get_act(channels):
            if activation_type == 'hrelu': return HReLU(channels)
            if activation_type == 'swiglu': return SwiGLU(dim=1)
            return nn.ReLU()

        def get_out(c):
            return c * 2 if activation_type == 'swiglu' else c

        self.conv1 = nn.Conv2d(1, get_out(32), 3, padding=1)
        self.bn1 = nn.BatchNorm2d(get_out(32)) if use_batchnorm else nn.Identity()
        self.act1 = get_act(32)
        
        self.conv2 = nn.Conv2d(32, get_out(64), 3, padding=1)
        self.bn2 = nn.BatchNorm2d(get_out(64)) if use_batchnorm else nn.Identity()
        self.act2 = get_act(64)
        
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, get_out(128))
        self.bn3 = nn.BatchNorm1d(get_out(128)) if use_batchnorm else nn.Identity()
        self.act3 = get_act(128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.act3(self.bn3(self.fc1(x)))
        return self.fc2(x)

class CifarNet(nn.Module):
    """VGG-style CNN for CIFAR-10"""
    def __init__(self, use_batchnorm=False, activation_type='hrelu'):
        super().__init__()
        
        def get_act(c):
            if activation_type == 'hrelu': return HReLU(c)
            if activation_type == 'swiglu': return SwiGLU(dim=1)
            return nn.ReLU()

        def get_out(c):
            return c * 2 if activation_type == 'swiglu' else c

        self.layers = nn.Sequential(
            nn.Conv2d(3, get_out(64), 3, padding=1),
            nn.BatchNorm2d(get_out(64)) if use_batchnorm else nn.Identity(),
            get_act(64),
            nn.Conv2d(64, get_out(64), 3, padding=1),
            nn.BatchNorm2d(get_out(64)) if use_batchnorm else nn.Identity(),
            get_act(64),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, get_out(128), 3, padding=1),
            nn.BatchNorm2d(get_out(128)) if use_batchnorm else nn.Identity(),
            get_act(128),
            nn.Conv2d(128, get_out(128), 3, padding=1),
            nn.BatchNorm2d(get_out(128)) if use_batchnorm else nn.Identity(),
            get_act(128),
            nn.MaxPool2d(2, 2),
            
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, get_out(512)),
            nn.BatchNorm1d(get_out(512)) if use_batchnorm else nn.Identity(),
            get_act(512),
            nn.Linear(512, 10)
        )
        
    def forward(self, x):
        return self.layers(x)

class DeepNet(nn.Module):
    """Deep MLP for stability tests"""
    def __init__(self, input_size=784, hidden_size=256, num_layers=20, activation_type='hrelu'):
        super().__init__()
        
        def get_out(s): return s * 2 if activation_type == 'swiglu' else s
        
        layers = []
        curr_size = input_size
        for _ in range(num_layers):
            layer = nn.Linear(curr_size, get_out(hidden_size))
            nn.init.kaiming_normal_(layer.weight, a=0, mode='fan_in')
            layer.weight.data *= 1.5 # Edge of chaos
            layers.append(layer)
            
            if activation_type == 'hrelu': layers.append(HReLU(hidden_size))
            elif activation_type == 'swiglu': layers.append(SwiGLU(dim=1))
            else: layers.append(nn.ReLU())
            curr_size = hidden_size
            
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_size, 10)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.classifier(self.features(x))

# =============================================================================
# UTILITIES
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        if torch.isnan(loss): return None, None
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += output.argmax(1).eq(target).sum().item()
        total += target.size(0)
    return total_loss / len(loader), 100. * correct / total

def test(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            correct += output.argmax(1).eq(target).sum().item()
            total += target.size(0)
    return total_loss / len(loader), 100. * correct / total

def get_activation_stats(model, loader, device):
    model.eval()
    stats = []
    def hook_fn(module, input, output):
        stats.append({
            'mean': output.mean().item(),
            'std': output.std().item(),
            'min': output.min().item(),
            'max': output.max().item(),
            'abs_mean': output.abs().mean().item()
        })
    handles = []
    for m in model.modules():
        if isinstance(m, (HReLU, nn.ReLU, SwiGLU)):
            handles.append(m.register_forward_hook(hook_fn))
    
    data, _ = next(iter(loader))
    with torch.no_grad(): model(data[:32].to(device))
    for h in handles: h.remove()
    return stats

# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_single(dataset, activation, batchnorm, epochs, layers, lr, device, 
               use_hrelu_opt=False, k_mult=1.0, o_mult=2.0, n_mult=0.5):
    """Run a single benchmark configuration"""
    config_name = f"{activation.upper()}" + (" + BN" if batchnorm else " (No BN)")
    if use_hrelu_opt and activation == 'hrelu':
        config_name += " [HReLU-Opt]"
    
    print(f"\n{'='*60}")
    print(f"🚀 {dataset.upper()} | {config_name}")
    print(f"{'='*60}")

    # Data Loading
    if dataset in ['mnist', 'deep']:
        t = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_set = datasets.MNIST('./data', train=True, download=True, transform=t)
        test_set = datasets.MNIST('./data', train=False, transform=t)
    else:
        t_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        t_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        train_set = datasets.CIFAR10('./data', train=True, download=True, transform=t_train)
        test_set = datasets.CIFAR10('./data', train=False, transform=t_test)

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False, num_workers=0)

    # Model Selection
    if dataset == 'mnist': model = SimpleNet(batchnorm, activation)
    elif dataset == 'cifar10': model = CifarNet(batchnorm, activation)
    else: model = DeepNet(num_layers=layers, activation_type=activation)
    model.to(device)

    # Optimizer Selection
    if use_hrelu_opt and activation == 'hrelu' and HRELU_OPT_AVAILABLE:
        print(f"  Using H-ReLU Aware Optimizer: k={k_mult}x, o={o_mult}x, n={n_mult}x")
        optimizer = create_hrelu_optimizer(model, lr=lr, k_mult=k_mult, o_mult=o_mult, n_mult=n_mult)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    
    criterion = nn.CrossEntropyLoss()
    
    history = {'config': config_name, 'dataset': dataset, 'activation': activation, 'batchnorm': batchnorm,
               'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': [], 'status': 'success',
               'hrelu_opt': use_hrelu_opt if activation == 'hrelu' else False}
    start_time = time.time()

    for epoch in range(epochs):
        loss, acc = train_epoch(model, train_loader, optimizer, criterion, device)
        if loss is None:
            print("❌ Model Exploded (NaN Loss)")
            history['status'] = 'exploded'
            break
        t_loss, t_acc = test(model, test_loader, criterion, device)
        
        history['train_acc'].append(acc)
        history['test_acc'].append(t_acc)
        history['train_loss'].append(loss)
        history['test_loss'].append(t_loss)
        print(f"Epoch {epoch+1}/{epochs} | Train: {acc:.2f}% | Test: {t_acc:.2f}%")

    history['total_time'] = time.time() - start_time
    history['final_test_acc'] = history['test_acc'][-1] if history['test_acc'] else 0.0
    print(f"✅ Completed in {history['total_time']:.1f}s | Final Test Acc: {history['final_test_acc']:.2f}%")
    
    return history

def run():
    parser = argparse.ArgumentParser(description='H-ReLU Benchmarks')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'deep'], help='Dataset to use')
    parser.add_argument('--activation', type=str, default='hrelu', choices=['hrelu', 'relu', 'swiglu'], help='Activation function')
    parser.add_argument('--batchnorm', action='store_true', help='Use BatchNorm')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--layers', type=int, default=20, help='Layers (for deep task)')
    parser.add_argument('--deep-layers', type=int, default=None, help='Override layers for deep test (e.g., 40 for extreme depth)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--out', type=str, default='results/benchmark_results.json', help='Output JSON path')
    parser.add_argument('--all', action='store_true', help='Run all combinations (3 activations × 2 BN settings × 3 datasets)')
    
    # H-ReLU Optimizer Options
    parser.add_argument('--hrelu-opt', action='store_true', help='Use H-ReLU aware optimizer (only for hrelu activation)')
    parser.add_argument('--k-mult', type=float, default=1.0, help='Learning rate multiplier for k (positive slope)')
    parser.add_argument('--o-mult', type=float, default=2.0, help='Learning rate multiplier for o (negative slope)')
    parser.add_argument('--n-mult', type=float, default=0.5, help='Learning rate multiplier for n (threshold)')
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.all:
        print(f"\n{'='*60}")
        print("🚀 RUNNING COMPREHENSIVE BENCHMARK SUITE")
        print(f"{'='*60}")
        print(f"Device: {device}")
        print("Configuration:")
        print("  - ReLU + SwiGLU: 2 activations × 2 BN × 3 datasets = 12 runs")
        print("  - H-ReLU (no opt): 1 activation × 2 BN × 3 datasets = 6 runs")
        print("  - H-ReLU (with opt): 1 activation × 2 BN × 3 datasets = 6 runs")
        print("  TOTAL: 24 runs")
        if args.deep_layers:
            print(f"  Deep network layers: {args.deep_layers}")
        print(f"  H-ReLU Optimizer: k={args.k_mult}x, o={args.o_mult}x, n={args.n_mult}x")
        print(f"{'='*60}\n")
        
        all_results = []
        bn_settings = [False, True]
        datasets = ['mnist', 'cifar10', 'deep']
        
        # Total: 12 (ReLU+SwiGLU) + 6 (H-ReLU no opt) + 6 (H-ReLU with opt) = 24
        total_runs = 24
        current_run = 0
        
        for dataset in datasets:
            dataset_results = []
            # Use deep-layers override for deep dataset
            layers_to_use = args.deep_layers if (dataset == 'deep' and args.deep_layers) else args.layers
            
            # Run ReLU and SwiGLU (standard, no optimizer)
            for activation in ['relu', 'swiglu']:
                for use_bn in bn_settings:
                    current_run += 1
                    print(f"\n[{current_run}/{total_runs}]", end=" ")
                    
                    result = run_single(dataset, activation, use_bn, args.epochs, layers_to_use, args.lr, device,
                                       False, args.k_mult, args.o_mult, args.n_mult)  # No optimizer
                    dataset_results.append(result)
                    all_results.append(result)
            
            # Run H-ReLU WITHOUT optimizer
            for use_bn in bn_settings:
                current_run += 1
                print(f"\n[{current_run}/{total_runs}]", end=" ")
                
                result = run_single(dataset, 'hrelu', use_bn, args.epochs, layers_to_use, args.lr, device,
                                   False, args.k_mult, args.o_mult, args.n_mult)  # No optimizer
                dataset_results.append(result)
                all_results.append(result)
            
            # Run H-ReLU WITH optimizer
            for use_bn in bn_settings:
                current_run += 1
                print(f"\n[{current_run}/{total_runs}]", end=" ")
                
                result = run_single(dataset, 'hrelu', use_bn, args.epochs, layers_to_use, args.lr, device,
                                   True, args.k_mult, args.o_mult, args.n_mult)  # WITH optimizer
                dataset_results.append(result)
                all_results.append(result)
            
            # Save per-dataset results
            output_path = Path(f'results/{dataset}_results.json')
            output_path.parent.mkdir(exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(dataset_results, f, indent=2)
            print(f"\n💾 Saved {dataset.upper()} results to {output_path}")
        
        # Save master summary
        summary_path = Path('results/all_results_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Print final summary table
        print(f"\n{'='*80}")
        print("📊 FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"{'Dataset':<10} {'Activation':<10} {'BatchNorm':<10} {'Test Acc':<12} {'Time':<10} {'Status':<10}")
        print("-" * 80)
        for r in all_results:
            bn_str = "Yes" if r['batchnorm'] else "No"
            acc_str = f"{r['final_test_acc']:.2f}%" if r['status'] == 'success' else 'N/A'
            time_str = f"{r['total_time']:.1f}s"
            print(f"{r['dataset']:<10} {r['activation'].upper():<10} {bn_str:<10} {acc_str:<12} {time_str:<10} {r['status']:<10}")
        
        print(f"\n✅ All results saved to {summary_path}")
        
    else:
        # Single run mode
        print(f"🚀 Running {args.dataset.upper()} | Act: {args.activation.upper()} | BN: {args.batchnorm} | Device: {device}")
        if args.hrelu_opt and args.activation == 'hrelu':
            print(f"  H-ReLU Optimizer: k={args.k_mult}x, o={args.o_mult}x, n={args.n_mult}x")
        
        result = run_single(args.dataset, args.activation, args.batchnorm, args.epochs, args.layers, args.lr, device,
                           args.hrelu_opt, args.k_mult, args.o_mult, args.n_mult)
        
        # Save results
        Path(args.out).parent.mkdir(exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"✅ Results saved to {args.out}")

if __name__ == '__main__':
    run()
