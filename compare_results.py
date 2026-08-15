import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
SAVE_DIR = "outputs"
FIG_DIR = os.path.join(SAVE_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
class_names = {'A': 'AD', 'F': 'FTD', 'C': 'CN'}
results = {}
for name in ["eegnet", "deepconvnet"]:
    path = os.path.join(SAVE_DIR, f"{name}_results.npz")
    if not os.path.exists(path):
        print(f"[skip] {name}: no results file found, run train_{name}.py first")
        continue
    r = np.load(path, allow_pickle=True)
    results[name] = r
    classes = list(r['classes'])
    label_names = [class_names.get(c, c) for c in classes]
    print(f"\n=== {name.upper()} ===")
    print(f"Best epoch: {int(r['best_epoch'])} | Best val_acc: {float(r['best_val_acc']):.4f}")
    print(f"Final test_acc: {float(r['test_acc']):.4f}")
    print(classification_report(r['labels'], r['preds'], target_names=label_names, digits=4, zero_division=0))
if len(results) < 2:
    print("\nNeed both eegnet_results.npz and deepconvnet_results.npz to compare.")
else:
    summary_path = os.path.join(SAVE_DIR, "comparison_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Model Comparison — 3-Class EEG Classification (AD vs FTD vs CN)\n" + "=" * 60 + "\n\n")
        for name, r in results.items():
            f.write(f"{name.upper()}\n")
            f.write(f"  Best epoch: {int(r['best_epoch'])}\n")
            f.write(f"  Best validation accuracy: {float(r['best_val_acc']):.4f}\n")
            f.write(f"  FINAL test accuracy: {float(r['test_acc']):.4f}\n\n")
    print(f"\nSaved comparison summary to {summary_path}")

    names = list(results.keys())
    accs = [float(results[n]['test_acc']) for n in names]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(names, accs, color=['steelblue', 'seagreen'])
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, acc + 0.01, f"{acc:.3f}", ha='center')
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Test accuracy")
    ax.set_title("EEGNet vs DeepConvNet — Final Test Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "comparison_accuracy_bar.png"), dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {'eegnet': 'steelblue', 'deepconvnet': 'seagreen'}
    for name, r in results.items():
        ax.plot(r['val_acc'], label=f"{name} val_acc", color=colors.get(name))
        ax.plot(r['train_acc'], label=f"{name} train_acc", linestyle='--', alpha=0.6, color=colors.get(name))
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title("Training Curves — EEGNet vs DeepConvNet")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "comparison_training_curves.png"), dpi=150)
    plt.close(fig)
    print("Saved comparison_accuracy_bar.png and comparison_training_curves.png")


