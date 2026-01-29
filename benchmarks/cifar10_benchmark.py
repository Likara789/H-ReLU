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


class CifarNet(nn.Module):
    """VGG-style CNN for CIFAR-10"""
    def __init__(self, use_batchnorm=False, use_h_relu=True):
        super().__init__()
        self.use_batchnorm = use_batchnorm
        self.use_h_relu = use_h_relu
        
        # Block 1: 3 -> 64
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.act1 = HReLU(64) if use_h_relu else nn.ReLU()
        
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.act2 = HReLU(64) if use_h_relu else nn.ReLU()
        
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2: 64 -> 128
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()
        self.act3 = HReLU(128) if use_h_relu else nn.ReLU()
        
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()
        self.act4 = HReLU(128) if use_h_relu else nn.ReLU()
        
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3: 128 -> 256
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256) if use_batchnorm else nn.Identity()
        self.act5 = HReLU(256) if use_h_relu else nn.ReLU()
        
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(256) if use_batchnorm else nn.Identity()
        self.act6 = HReLU(256) if use_h_relu else nn.ReLU()
        
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Fully connected
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.bn7 = nn.BatchNorm1d(1024) if use_batchnorm else nn.Identity()
        self.act7 = HReLU(1024) if use_h_relu else nn.ReLU()
        
        self.fc2 = nn.Linear(1024, 10)
        
    def forward(self, x):
        x = self.pool1(self.act2(self.bn2(self.conv2(self.act1(self.bn1(self.conv1(x)))))))
        x = self.pool2(self.act4(self.bn4(self.conv4(self.act3(self.bn3(self.conv3(x)))))))
        x = self.pool3(self.act6(self.bn6(self.conv6(self.act5(self.bn5(self.conv5(x)))))))
        
        x = x.view(x.size(0), -1)
        x = self.fc2(self.act7(self.bn7(self.fc1(x))))
        return x


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in loader:
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


def run_cifar_experiment(config_name, use_batchnorm, use_h_relu, epochs=10):
    print(f"\n{'='*60}\nRunning CIFAR-10: {config_name}\n{'='*60}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Augmentation for CIFAR-10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('./data', train=False, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, pin_memory=True)
    
    model = CifarNet(use_batchnorm=use_batchnorm, use_h_relu=use_h_relu).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    results = {'config': config_name, 'epochs': [], 'train_acc': [], 'test_acc': [], 'train_loss': [], 'test_loss': []}
    
    start_time = time.time()
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = test(model, test_loader, criterion, device)
        
        results['epochs'].append(epoch + 1)
        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)
        results['train_loss'].append(train_loss) # Added train_loss
        results['test_loss'].append(test_loss)   # Added test_loss
        
        print(f"Epoch {epoch+1}/{epochs} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | Loss: {train_loss:.4f}")
        
    results['total_time'] = time.time() - start_time
    # Serialize results to simple dictionary for JSON dump
    return results # Removed complex objects like activation stats for speed on this run


def main():
    experiments = [
        ("H-ReLU (No BatchNorm)", False, True),
        ("ReLU + BatchNorm", True, False),
        ("ReLU (No BatchNorm)", False, False),
        ("H-ReLU + BatchNorm", True, True)
    ]
    
    all_results = []
    for config in experiments:
        all_results.append(run_cifar_experiment(*config, epochs=10))
        
    output_dir = Path('results')
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / 'cifar_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
        
    print("\nCIFAR-10 Experiments Completed.")

if __name__ == "__main__":
    main()
