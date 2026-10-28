from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_curves(history, out_dir, name):
    out_dir = Path(out_dir)
    e = [int(h["epoch"]) for h in history]
    for metric, ylabel, suffix in (
        ("accuracy", "Accuracy (%)", "accuracy"),
        ("loss", "Cross-entropy loss", "loss"),
    ):
        fig, ax = plt.subplots(figsize=(7.4, 4.5), dpi=145)
        for split, label in (("train", "Training"), ("val", "Validation"), ("test", "Testing (diagnostic)")):
            values = [h.get(f"{split}_{metric}") for h in history]
            if all(v is not None for v in values):
                ax.plot(e, values, label=label, linewidth=1.7)
        ax.set(title=f"{name}: {ylabel} per epoch", xlabel="Epoch", ylabel=ylabel)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"{suffix}.png")
        plt.close(fig)


def plot_confusion_matrix(cm, path, title):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=145)
    im = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Class 0", "Class 1"],
           yticklabels=["Class 0", "Class 1"], xlabel="Predicted", ylabel="True", title=title)
    threshold = float(cm.max()) / 2 if cm.size else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black", fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_experiment_summary(rows, output_dir):
    labels = [r["experiment"] for r in rows]
    values = [r["final_test_accuracy"] for r in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 1.25), 4.6), dpi=145)
    bars = ax.bar(labels, values, color="steelblue")
    ax.set(ylabel="Final test accuracy (%)", title="Validation-selected checkpoint: final test accuracy")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, f"{v:.2f}%", ha="center", fontsize=8)
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "experiment_comparison.png")
    plt.close(fig)
