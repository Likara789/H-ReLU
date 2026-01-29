import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.style.use('dark_background')

def plot_deep_analysis():
    filepath = Path('results/deep_experiment.json')
    if not filepath.exists():
        print("Run deep_experiment.py first!")
        return
        
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    output_dir = Path('results')
    
    # 1. Gradient Flow across 50 layers
    plt.figure(figsize=(12, 6))
    
    for name in ['relu', 'hrelu']:
        if data[name]['status'] == 'success' or len(data[name]['history']['gradient_history']) > 0:
            # Take the first recorded batch gradients
            grads = data[name]['history']['gradient_history'][0]
            layers = np.arange(len(grads))
            plt.plot(layers, grads, label=f'{name.upper()} Gradients', marker='o', alpha=0.7)
            
    plt.yscale('log')
    plt.title('Gradient Magnitude across 20 Layers (Log Scale)')
    plt.xlabel('Layer Index')
    plt.ylabel('|∂L/∂W|')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(output_dir / 'deep_gradient_flow.png')
    
    # 2. Activation Stability
    plt.figure(figsize=(12, 6))
    for name in ['relu', 'hrelu']:
        if len(data[name]['history']['activation_history']) > 0:
            # Take last epoch's first batch activations
            activations = data[name]['history']['activation_history'][-1]
            abs_means = [a['abs_mean'] for a in activations]
            layers = np.arange(len(abs_means))
            plt.plot(layers, abs_means, label=f'{name.upper()} Abs Mean Activation', marker='s', alpha=0.7)
            
    plt.title('Activation Stability: Abs Mean through 20 Layers')
    plt.xlabel('Layer Index')
    plt.ylabel('Average Absolute Activation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'deep_activation_stability.png')
    
    # 3. Min/Max Range (The Counter-Screaming Visualization)
    plt.figure(figsize=(12, 6))
    for name in ['relu', 'hrelu']:
        if len(data[name]['history']['activation_history']) > 0:
            activations = data[name]['history']['activation_history'][0] # First batch recorded
            max_vals = [a['max'] for a in activations]
            min_vals = [a['min'] for a in activations]
            layers = np.arange(len(max_vals))
            
            p = plt.plot(layers, max_vals, label=f'{name.upper()} Max', marker='^', alpha=0.6)
            plt.plot(layers, min_vals, label=f'{name.upper()} Min', marker='v', alpha=0.6, color=p[0].get_color(), linestyle='--')
            
    plt.title('Activation Envelope (Min/Max) across 20 Layers')
    plt.xlabel('Layer Index')
    plt.ylabel('Activation Range')
    plt.yscale('symlog') # Symlog to show both large positive and negative spreads
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'deep_activation_envelope.png')
    
    # 4. Final Summary Text
    summary = []
    summary.append("=== 20-LAYER DEEP EXPERIMENT SUMMARY ===")
    for name in ['relu', 'hrelu']:
        status = data[name]['status']
        acc = data[name]['history']['acc'][-1] if status == 'success' else 'N/A'
        summary.append(f"{name.upper()}: Status={status}, Final Acc={acc}")
        
    summary_text = "\n".join(summary)
    print("\n" + summary_text)
    with open(output_dir / 'deep_summary.txt', 'w') as f:
        f.write(summary_text)

if __name__ == "__main__":
    plot_deep_analysis()
