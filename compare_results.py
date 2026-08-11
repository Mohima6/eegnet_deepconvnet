import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

SAVE_DIR = "outputs"
MODEL_NAMES = ["eegnet", "deepconvnet"]

results = {}
for name in MODEL_NAMES:
    path = os.path.join(SAVE_DIR, f"{name}_results.npz")
    if os.path.exists(path):
        results[name] = np.load(path, allow_pickle=True)
    else:
        print(f"[skip] {name}: no results file found, run train_{name}.py first")

if not results:
    raise RuntimeError("No results found. Run train_eegnet.py and train_deepconvnet.py first.")

# ---- Print + collect summary ----
summary_lines = []
summary_lines.append("Model Comparison: EEGNet vs DeepConvNet")
summary_lines.append("=========================================\n")

for name, r in results.items():
    test_acc = float(r['test_acc'])
    best_acc = float(r['best_test_acc']) if 'best_test_acc' in r else test_acc
    report = classification_report(r['labels'], r['preds'], target_names=['AD', 'HC'])
    cm = confusion_matrix(r['labels'], r['preds'])

    block = f"\n=== {name.upper()} ===\n"
    block += f"Final test accuracy: {test_acc:.4f}\n"
    block += f"Best test accuracy (during training): {best_acc:.4f}\n"
    block += f"Confusion matrix (rows=true, cols=pred, order=[AD,HC]):\n{cm}\n"
    block += f"\n{report}\n"

    print(block)
    summary_lines.append(block)

# ---- Summary table ----
print("\n--- Summary ---")
summary_lines.append("\n--- Summary ---")
for name, r in results.items():
    line = f"{name:15s} final_test_acc = {float(r['test_acc']):.4f}  best_test_acc = {float(r.get('best_test_acc', r['test_acc'])):.4f}"
    print(line)
    summary_lines.append(line)

# ---- Save text summary ----
txt_path = os.path.join(SAVE_DIR, "comparison_summary.txt")
with open(txt_path, "w") as f:
    f.write("\n".join(summary_lines))
print(f"\nSaved comparison summary to {txt_path}")

# ---- Figure 1: Test accuracy bar chart ----
plt.figure(figsize=(6, 5))
names = list(results.keys())
accs = [float(results[n]['test_acc']) for n in names]
best_accs = [float(results[n].get('best_test_acc', results[n]['test_acc'])) for n in names]

x = np.arange(len(names))
width = 0.35
plt.bar(x - width/2, accs, width, label='Final (last epoch)', color='steelblue')
plt.bar(x + width/2, best_accs, width, label='Best (during training)', color='darkorange')
plt.xticks(x, [n.upper() for n in names])
plt.ylabel('Test Accuracy')
plt.ylim(0, 1)
plt.title('Test Accuracy Comparison: EEGNet vs DeepConvNet')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "comparison_accuracy_bar.png"), dpi=150)
print(f"Saved bar chart to {SAVE_DIR}/comparison_accuracy_bar.png")
plt.close()

# ---- Figure 2: Training curves (loss + train/test accuracy) side by side ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for name, r in results.items():
    if 'train_acc' in r:
        axes[0].plot(r['train_acc'], label=f'{name} train_acc')
    if 'test_acc_history' in r:
        axes[0].plot(r['test_acc_history'], linestyle='--', label=f'{name} test_acc')

axes[0].set_title('Accuracy over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend(fontsize=8)

for name, r in results.items():
    if 'train_loss' in r:
        axes[1].plot(r['train_loss'], label=f'{name} train_loss')

axes[1].set_title('Training Loss over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "comparison_training_curves.png"), dpi=150)
print(f"Saved training curves to {SAVE_DIR}/comparison_training_curves.png")
plt.close()

print("\nDone. Check the 'outputs' folder for:")
print("  - comparison_summary.txt")
print("  - comparison_accuracy_bar.png")
print("  - comparison_training_curves.png")