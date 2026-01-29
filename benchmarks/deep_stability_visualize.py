import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.style.use('dark_background')

def plot_deep_analysis():
    filepath = Path('results/deep_results.json')
    if not filepath.exists():
        print("Run deep_stability_benchmark.py first!")
        return
        
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    output_dir = Path('references')
    output_dir.mkdir(exist_ok=True)
    
    # Extract activation types from results
    act_types = []
    for result in data:
        if result['status'] == 'success' and result['activation'] not in act_types:
            act_types.append(result['activation'])
    
    # 1. Gradient Flow across 20 layers
    plt.figure(figsize=(12, 6))
    
    for name in act_types:
        matching = [r for r in data if r['activation'] == name and r['status'] == 'success']
        if matching and len(matching[0].get('train_loss', [])) > 0:
            # Plot training loss as proxy for gradient health
            epochs = list(range(1, len(matching[0]['train_loss'])+1))
            plt.plot(epochs, matching[0]['train_loss'], label=f'{name.upper()} Train Loss', marker='o', alpha=0.7)
            
    plt.yscale('log')
    plt.title('Training Loss across Epochs (Deep 20-Layer Network)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(output_dir / 'deep_gradient_flow.png', dpi=300)
    print(f"Saved: {output_dir / 'deep_gradient_flow.png'}")
    
    # 2. Accuracy Comparison
    plt.figure(figsize=(12, 6))
    for result in data:
        if result['status'] == 'success' and result['test_acc']:
            epochs = list(range(1, len(result['test_acc'])+1))
            label = f"{result['activation'].upper()}" + (" + BN" if result['batchnorm'] else " (No BN)")
            plt.plot(epochs, result['test_acc'], label=label, marker='s', alpha=0.7)
            
    plt.title('Test Accuracy: Deep 20-Layer Network Stability')
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'deep_activation_stability.png', dpi=300)
    print(f"Saved: {output_dir / 'deep_activation_stability.png'}")
    
    # 3. Final Summary Text
    summary = []
    summary.append("=== 20-LAYER DEEP EXPERIMENT SUMMARY ===")
    for result in data:
        status = result['status']
        config = f"{result['activation'].upper()}" + (" + BN" if result['batchnorm'] else " (No BN)")
        acc = f"{result['final_test_acc']:.2f}%" if status == 'success' else 'EXPLODED'
        summary.append(f"{config:<20}: Status={status:<10}, Final Acc={acc}")
        
    summary_text = "\n".join(summary)
    print("\n" + summary_text)
    with open(output_dir / 'deep_summary.txt', 'w') as f:
        f.write(summary_text)
    print(f"Saved: {output_dir / 'deep_summary.txt'}")

if __name__ == "__main__":
    plot_deep_analysis()
