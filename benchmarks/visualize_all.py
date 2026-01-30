"""
Unified Visualizer for H-ReLU Benchmarks

Reads from a single JSON file (e.g., all_results_summary.json) and generates
comprehensive visualizations across all datasets and configurations.

Usage:
    python benchmarks/visualize_all.py results/all_results_summary.json
    python benchmarks/visualize_all.py results/all_results_summary.json --out references/
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

plt.style.use('dark_background')

def load_results(filepath):
    """Load results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def group_by_dataset(results):
    """Group results by dataset"""
    grouped = {'mnist': [], 'cifar10': [], 'deep': []}
    for r in results:
        if r['dataset'] in grouped:
            grouped[r['dataset']].append(r)
    return grouped

def plot_dataset_comparison(results, output_dir):
    """Plot test accuracy comparison across all datasets"""
    grouped = group_by_dataset(results)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Test Accuracy Comparison Across Datasets', fontsize=16, fontweight='bold')
    
    colors = {'hrelu': '#2ecc71', 'relu': '#e74c3c', 'swiglu': '#3498db'}
    
    for idx, (dataset, ax) in enumerate(zip(['mnist', 'cifar10', 'deep'], axes)):
        dataset_results = grouped[dataset]
        
        for result in dataset_results:
            if result['status'] != 'success' or not result['test_acc']:
                continue
            
            epochs = list(range(1, len(result['test_acc']) + 1))
            label = f"{result['activation'].upper()}" + (" + BN" if result['batchnorm'] else "")
            if result.get('hrelu_opt'):
                label += " [Opt]"
            
            color = colors.get(result['activation'], '#ffffff')
            linestyle = '-' if result['batchnorm'] else '--'
            linewidth = 3 if result['activation'] == 'hrelu' else 2
            alpha = 1.0 if result['activation'] == 'hrelu' else 0.7
            
            ax.plot(epochs, result['test_acc'], label=label, color=color,
                   linestyle=linestyle, linewidth=linewidth, alpha=alpha, marker='o', markersize=4)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title(f'{dataset.upper()}')
        ax.legend(prop={'size': 8})
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'all_datasets_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'all_datasets_comparison.png'}")
    plt.close()

def plot_final_accuracy_bars(results, output_dir):
    """Bar chart of final test accuracies"""
    grouped = group_by_dataset(results)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Final Test Accuracy by Configuration', fontsize=16, fontweight='bold')
    
    colors = {'hrelu': '#2ecc71', 'relu': '#e74c3c', 'swiglu': '#3498db'}
    
    for idx, (dataset, ax) in enumerate(zip(['mnist', 'cifar10', 'deep'], axes)):
        dataset_results = grouped[dataset]
        
        labels = []
        accuracies = []
        bar_colors = []
        
        for result in dataset_results:
            label = f"{result['activation'].upper()}"
            if result['batchnorm']:
                label += "+BN"
            if result.get('hrelu_opt'):
                label += "[O]"
            
            labels.append(label)
            
            if result['status'] == 'success':
                accuracies.append(result['final_test_acc'])
            else:
                accuracies.append(0)
            
            bar_colors.append(colors.get(result['activation'], '#ffffff'))
        
        x = np.arange(len(labels))
        bars = ax.bar(x, accuracies, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=1)
        
        # Highlight failed runs
        for i, result in enumerate(dataset_results):
            if result['status'] != 'success':
                ax.text(i, 5, '❌', ha='center', fontsize=16)
        
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title(f'{dataset.upper()}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'final_accuracy_bars.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'final_accuracy_bars.png'}")
    plt.close()

def plot_training_time_comparison(results, output_dir):
    """Compare training times across configurations"""
    grouped = group_by_dataset(results)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Training Time Comparison', fontsize=16, fontweight='bold')
    
    colors = {'hrelu': '#2ecc71', 'relu': '#e74c3c', 'swiglu': '#3498db'}
    
    for idx, (dataset, ax) in enumerate(zip(['mnist', 'cifar10', 'deep'], axes)):
        dataset_results = grouped[dataset]
        
        labels = []
        times = []
        bar_colors = []
        
        for result in dataset_results:
            label = f"{result['activation'].upper()}"
            if result['batchnorm']:
                label += "+BN"
            if result.get('hrelu_opt'):
                label += "[O]"
            
            labels.append(label)
            times.append(result.get('total_time', 0))
            bar_colors.append(colors.get(result['activation'], '#ffffff'))
        
        x = np.arange(len(labels))
        ax.bar(x, times, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=1)
        
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'{dataset.upper()}')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_time_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'training_time_comparison.png'}")
    plt.close()

def plot_hrelu_optimizer_impact(results, output_dir):
    """Show impact of H-ReLU optimizer vs standard Adam"""
    hrelu_results = [r for r in results if r['activation'] == 'hrelu']
    
    if not any(r.get('hrelu_opt') for r in hrelu_results):
        print("No H-ReLU optimizer results found, skipping optimizer impact plot")
        return
    
    grouped = group_by_dataset(hrelu_results)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('H-ReLU: Standard Adam vs H-ReLU Optimizer', fontsize=16, fontweight='bold')
    
    for idx, (dataset, ax) in enumerate(zip(['mnist', 'cifar10', 'deep'], axes)):
        dataset_results = grouped[dataset]
        
        for result in dataset_results:
            if result['status'] != 'success' or not result['test_acc']:
                continue
            
            epochs = list(range(1, len(result['test_acc']) + 1))
            
            bn_str = " + BN" if result['batchnorm'] else " (No BN)"
            opt_str = " [HReLU-Opt]" if result.get('hrelu_opt') else " [Adam]"
            label = bn_str + opt_str
            
            linestyle = '-' if result.get('hrelu_opt') else '--'
            linewidth = 3 if result.get('hrelu_opt') else 2
            color = '#2ecc71' if not result['batchnorm'] else '#3498db'
            
            ax.plot(epochs, result['test_acc'], label=label, linestyle=linestyle,
                   linewidth=linewidth, color=color, marker='o', markersize=4)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title(f'{dataset.upper()}')
        ax.legend(prop={'size': 9})
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'hrelu_optimizer_impact.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'hrelu_optimizer_impact.png'}")
    plt.close()

def create_summary_table(results, output_dir):
    """Create comprehensive summary table"""
    summary = []
    summary.append("=" * 120)
    summary.append("COMPREHENSIVE BENCHMARK SUMMARY")
    summary.append("=" * 120)
    summary.append("")
    
    # Header
    summary.append(f"{'Dataset':<10} {'Activation':<10} {'BN':<5} {'Opt':<5} {'Test Acc':<12} {'Train Acc':<12} {'Time':<10} {'Status':<10}")
    summary.append("-" * 120)
    
    # Group by dataset
    grouped = group_by_dataset(results)
    
    for dataset in ['mnist', 'cifar10', 'deep']:
        for result in grouped[dataset]:
            bn_str = "Yes" if result['batchnorm'] else "No"
            opt_str = "Yes" if result.get('hrelu_opt') else "No"
            acc_str = f"{result['final_test_acc']:.2f}%" if result['status'] == 'success' else 'N/A'
            train_acc = f"{result['train_acc'][-1]:.2f}%" if result['status'] == 'success' and result['train_acc'] else 'N/A'
            time_str = f"{result['total_time']:.1f}s"
            
            summary.append(f"{dataset:<10} {result['activation'].upper():<10} {bn_str:<5} {opt_str:<5} {acc_str:<12} {train_acc:<12} {time_str:<10} {result['status']:<10}")
        summary.append("")
    
    # Key findings
    summary.append("=" * 120)
    summary.append("KEY FINDINGS")
    summary.append("=" * 120)
    summary.append("")
    
    # Find best performers
    for dataset in ['mnist', 'cifar10', 'deep']:
        dataset_results = [r for r in grouped[dataset] if r['status'] == 'success']
        if dataset_results:
            best = max(dataset_results, key=lambda x: x['final_test_acc'])
            summary.append(f"{dataset.upper()} Best: {best['activation'].upper()}" + 
                          (" + BN" if best['batchnorm'] else "") +
                          (" [HReLU-Opt]" if best.get('hrelu_opt') else "") +
                          f" → {best['final_test_acc']:.2f}%")
    
    summary.append("")
    
    # H-ReLU optimizer impact
    hrelu_with_opt = [r for r in results if r['activation'] == 'hrelu' and r.get('hrelu_opt') and r['status'] == 'success']
    hrelu_without_opt = [r for r in results if r['activation'] == 'hrelu' and not r.get('hrelu_opt') and r['status'] == 'success']
    
    if hrelu_with_opt and hrelu_without_opt:
        summary.append("H-ReLU Optimizer Impact:")
        avg_time_with = np.mean([r['total_time'] for r in hrelu_with_opt])
        avg_time_without = np.mean([r['total_time'] for r in hrelu_without_opt])
        speedup = ((avg_time_without - avg_time_with) / avg_time_without) * 100
        summary.append(f"  Average speedup: {speedup:.1f}% faster")
    
    summary_text = "\n".join(summary)
    print("\n" + summary_text)
    
    with open(output_dir / 'comprehensive_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"\nSaved: {output_dir / 'comprehensive_summary.txt'}")

def main():
    parser = argparse.ArgumentParser(description='Unified H-ReLU Benchmark Visualizer')
    parser.add_argument('results_file', type=str, help='Path to results JSON file')
    parser.add_argument('--out', type=str, default='references/', help='Output directory for visualizations')
    args = parser.parse_args()
    
    results_file = Path(args.results_file)
    if not results_file.exists():
        print(f"Error: {results_file} not found!")
        return
    
    output_dir = Path(args.out)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Loading results from {results_file}...")
    results = load_results(results_file)
    print(f"Loaded {len(results)} experiment results")
    
    print("\nGenerating visualizations...")
    plot_dataset_comparison(results, output_dir)
    plot_final_accuracy_bars(results, output_dir)
    plot_training_time_comparison(results, output_dir)
    plot_hrelu_optimizer_impact(results, output_dir)
    create_summary_table(results, output_dir)
    
    print("\n✅ All visualizations generated!")

if __name__ == "__main__":
    main()
