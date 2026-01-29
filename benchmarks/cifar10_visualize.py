import json
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use('dark_background')

def plot_cifar_results():
    filepath = Path('results/cifar_results.json')
    if not filepath.exists():
        print("Run cifar_experiment.py first!")
        return
        
    with open(filepath, 'r') as f:
        results = json.load(f)
        
    output_dir = Path('results')
    
    # Plot Test Accuracy
    plt.figure(figsize=(10, 6))
    
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f1c40f'] # Green, Red, Blue, Yellow
    
    for idx, res in enumerate(results):
        epochs = res['epochs']
        acc = res['test_acc']
        label = res['config']
        
        # Highlight H-ReLU vs others
        linewidth = 3 if "H-ReLU" in label else 2
        linestyle = '-' if "H-ReLU" in label else '--'
        
        plt.plot(epochs, acc, label=label, color=colors[idx], linewidth=linewidth, marker='o', markersize=5)
        
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy (%)')
    plt.title('CIFAR-10 Benchmark: H-ReLU vs ReLU')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_dir / 'cifar_accuracy.png', dpi=300)
    print(f"Saved: {output_dir / 'cifar_accuracy.png'}")
    
    # Print comparison
    print("\nCIFAR-10 Final Results (10 Epochs):")
    print("-" * 40)
    for res in results:
        print(f"{res['config']:<25}: {res['test_acc'][-1]:.2f}%")

if __name__ == "__main__":
    plot_cifar_results()
