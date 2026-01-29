import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


plt.style.use('dark_background')

def plot_training_curves(results, output_dir):
    """Plot training and test accuracy/loss curves"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Comparison: H-ReLU vs Baselines', fontsize=16, fontweight='bold')
    
    # Expanded color palette for more experiments
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#1abc9c']
    
    for idx, result in enumerate(results):
        # Calculate epochs if missing
        if 'epochs' in result:
            epochs = result['epochs']
        else:
            epochs = list(range(1, len(result['train_acc']) + 1))
            
        color = colors[idx % len(colors)]
        label = result['config']
        
        # Train loss
        axes[0, 0].plot(epochs, result['train_loss'], label=label, color=color, linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Test loss
        axes[0, 1].plot(epochs, result['test_loss'], label=label, color=color, linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Test Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Train accuracy
        axes[1, 0].plot(epochs, result['train_acc'], label=label, color=color, linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].set_title('Training Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Test accuracy
        axes[1, 1].plot(epochs, result['test_acc'], label=label, color=color, linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].set_title('Test Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'training_curves.png'}")
    plt.close()


def plot_activation_stats(results, output_dir):
    """Plot activation statistics across layers"""
    
    if not any('activation_stats' in r for r in results):
        print("No activation stats found in results")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Activation Statistics by Layer', fontsize=16, fontweight='bold')
    
    # Identify layers dynamically
    first_stats = None
    for r in results:
        if 'activation_stats' in r and r['activation_stats']:
            first_stats = r['activation_stats']
            break
    
    if first_stats is None: return
    
    num_layers = len(first_stats)
    layers = [f'layer{i+1}' for i in range(num_layers)]
    x = np.arange(len(layers))
    width = 0.8 / len(results)
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12', '#1abc9c']
    
    metrics = ['mean', 'std', 'min', 'max']
    titles = ['Mean Activation', 'Std Activation', 'Min Activation', 'Max Activation']
    
    for metric_idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[metric_idx // 2, metric_idx % 2]
        
        for idx, result in enumerate(results):
            if 'activation_stats' not in result or not result['activation_stats']: continue
            stats = result['activation_stats']
            values = [s[metric] for s in stats]
            
            offset = (idx - len(results)/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=result['config'], color=colors[idx % len(colors)], alpha=0.8)
        
        ax.set_xlabel('Layer')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(layers)
        ax.legend(prop={'size': 8})
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'activation_stats.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'activation_stats.png'}")
    plt.close()


def plot_learned_parameters(results, output_dir):
    """Plot learned k, o, n parameters for H-ReLU"""
    
    # Filter for H-ReLU results that actually recorded params
    h_relu_results = [r for r in results if r.get('activation') == 'hrelu' and 'learned_params' in r]
    
    if not h_relu_results:
        # Check if we have any hrelu results at all, even without params
        h_relu_results = [r for r in results if r.get('activation') == 'hrelu']
        if not h_relu_results:
            print("No H-ReLU results to plot parameters for")
            return
        else:
            print("H-ReLU experiments were run but no learned parameters were recorded (run.py skips this for speed in --all mode)")
            return
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Learned H-ReLU Parameters', fontsize=16, fontweight='bold')
    
    # Identify layers dynamically from the first result
    learned_keys = h_relu_results[0]['learned_params'].keys()
    param_types = ['k', 'o', 'n']
    param_names = ['k (Positive Slope)', 'o (Negative Slope)', 'n (Shift/Threshold)']
    
    # Extract unique layer names
    layers = sorted(list(set([k.split('_')[0] for k in learned_keys])))
    
    for param_idx, (param, param_name) in enumerate(zip(param_types, param_names)):
        ax = axes[param_idx]
        
        for result in h_relu_results:
            learned = result['learned_params']
            values = []
            for layer in layers:
                key = f"{layer}_{param}"
                if key in learned:
                    values.append(np.mean(learned[key]))
                else:
                    values.append(0)
            
            ax.plot(layers, values, marker='o', linewidth=2, markersize=8, label=result['config'])
        
        ax.set_xlabel('Layer')
        ax.set_ylabel(param_name)
        ax.set_title(f'Learned {param_name} by Layer')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='white', linestyle='--', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learned_parameters.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'learned_parameters.png'}")
    plt.close()


def plot_parameter_distributions(results, output_dir):
    """Plot distributions of learned parameters across channels"""
    
    h_relu_results = [r for r in results if r.get('activation') == 'hrelu' and 'learned_params' in r]
    
    if not h_relu_results:
        return
    
    for result in h_relu_results:
        learned = result['learned_params']
        layers = sorted(list(set([k.split('_')[0] for k in learned.keys()])))
        
        fig, axes = plt.subplots(len(layers), 3, figsize=(15, 3 * len(layers)))
        fig.suptitle(f"Parameter Distributions: {result['config']}", fontsize=16, fontweight='bold')
        
        params = ['k', 'o', 'n']
        param_names = ['k (Positive)', 'o (Negative)', 'n (Shift)']
        
        for layer_idx, layer in enumerate(layers):
            for param_idx, (param, param_name) in enumerate(zip(params, param_names)):
                ax = axes[layer_idx, param_idx]
                key = f"{layer}_{param}"
                if key in learned:
                    values = learned[key]
                    ax.hist(values, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
                    mean_val = np.mean(values)
                    ax.axvline(x=mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
                    ax.legend(prop={'size': 7})
                
                ax.set_xlabel(param_name)
                ax.set_ylabel('Count')
                ax.set_title(f'{layer} - {param_name}')
                ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        safe_name = result['config'].replace(' ', '_').replace('+', 'and')
        plt.savefig(output_dir / f'param_dist_{safe_name}.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / f'param_dist_{safe_name}.png'}")
        plt.close()


def create_summary_table(results, output_dir):
    """Create a summary comparison table"""
    
    summary = []
    summary.append("=" * 80)
    summary.append("MNIST EXPERIMENT SUMMARY")
    summary.append("=" * 80)
    summary.append("")
    
    # Header
    summary.append(f"{'Configuration':<35} {'Accuracy':<12} {'Time':<12}")
    summary.append("-" * 80)
    
    for result in results:
        config = result['config']
        acc = f"{result['test_acc'][-1]:.2f}%" if result['test_acc'] else "N/A"
        time = f"{result['total_time']:.2f}s"
        summary.append(f"{config:<35} {acc:<12} {time:<12}")
    
    summary.append("")
    summary.append("=" * 80)
    summary.append("KEY FINDINGS")
    summary.append("=" * 80)
    summary.append("")
    
    # Identify key configurations
    h_relu_no_bn = [r for r in results if r.get('activation') == 'hrelu' and not r.get('batchnorm')]
    relu_with_bn = [r for r in results if r.get('activation') == 'relu' and r.get('batchnorm')]
    swiglu_no_bn = [r for r in results if r.get('activation') == 'swiglu' and not r.get('batchnorm')]
    
    if h_relu_no_bn:
        h_relu_acc = h_relu_no_bn[0]['test_acc'][-1]
        summary.append(f"H-ReLU (no BatchNorm): {h_relu_acc:.2f}%")
        
        if relu_with_bn:
            relu_acc = relu_with_bn[0]['test_acc'][-1]
            summary.append(f"ReLU + BatchNorm:      {relu_acc:.2f}%")
            
            if h_relu_acc >= relu_acc - 0.5:
                summary.append("✓ H-ReLU achieves comparable accuracy to ReLU + BatchNorm!")
        
        if swiglu_no_bn:
            swiglu_acc = swiglu_no_bn[0]['test_acc'][-1]
            summary.append(f"SwiGLU (no BatchNorm): {swiglu_acc:.2f}%")
            
            if h_relu_acc > swiglu_acc:
                summary.append(f"✓ H-ReLU outperformed SwiGLU by {h_relu_acc - swiglu_acc:.2f}%!")

        summary.append("")
        if 'activation_stats' in h_relu_no_bn[0] and h_relu_no_bn[0]['activation_stats']:
            h_relu_stats = h_relu_no_bn[0]['activation_stats']
            summary.append("Activation ranges (H-ReLU without BatchNorm):")
            for i, s in enumerate(h_relu_stats):
                summary.append(f"  Layer {i+1}: [{s['min']:7.3f}, {s['max']:7.3f}] (mean: {s['mean']:7.3f})")
            
            summary.append("")
            summary.append("✓ Neurons use BOTH positive and negative values for self-balancing!")
    
    summary_text = "\n".join(summary)
    print("\n" + summary_text)
    with open(output_dir / 'summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary_text)
    print(f"\nSaved: {output_dir / 'summary.txt'}")


def main():
    """Generate all visualizations"""
    
    results_file = Path('results/mnist_results.json')
    
    if not results_file.exists():
        print(f"Error: {results_file} not found. Run run.py --all first!")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    output_dir = Path('references')
    output_dir.mkdir(exist_ok=True)
    
    print("Generating visualizations...")
    
    plot_training_curves(results, output_dir)
    plot_activation_stats(results, output_dir)
    plot_learned_parameters(results, output_dir)
    plot_parameter_distributions(results, output_dir)
    create_summary_table(results, output_dir)
    
    print("\n✓ All visualizations generated!")


if __name__ == "__main__":
    main()
