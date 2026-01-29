import json
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use('dark_background')

def plot_cifar_results():
    filepath = Path('results/cifar10_results.json')
    if not filepath.exists():
        print("Run cifar10_benchmark.py first!")
        return
        
    with open(filepath, 'r') as f:
        results = json.load(f)
        
    output_dir = Path('references')
    output_dir.mkdir(exist_ok=True)
    
    # Plot Test Accuracy
    plt.figure(figsize=(12, 7))
    
    # Expanded color palette
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6', '#f39c12']
    
    for idx, res in enumerate(results):
        epochs = res['epochs'] if 'epochs' in res else list(range(1, len(res['test_acc'])+1))
        acc = res['test_acc']
        label = res['config']
        
        # Highlight H-ReLU vs others
        is_hrelu = "H-ReLU" in label or "HRELU" in label
        linewidth = 3 if is_hrelu else 2
        linestyle = '-' if is_hrelu else '--'
        alpha = 1.0 if is_hrelu else 0.8
        
        plt.plot(epochs, acc, label=label, color=colors[idx % len(colors)], 
                 linewidth=linewidth, linestyle=linestyle, alpha=alpha,
                 marker='o', markersize=5)
        
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('CIFAR-10 Benchmark: H-ReLU vs Baselines (including SwiGLU)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_dir / 'cifar_accuracy.png', dpi=300)
    print(f"Saved: {output_dir / 'cifar_accuracy.png'}")
    
    # Print comparison
    print("\nCIFAR-10 Final Results (10 Epochs):")
    print("-" * 40)
    for res in results:
        final_acc = res['test_acc'][-1] if res['test_acc'] else 0.0
        print(f"{res['config']:<25}: {final_acc:.2f}%")

if __name__ == "__main__":
    plot_cifar_results()
