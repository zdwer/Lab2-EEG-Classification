from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from dataloader import build_loaders
from models.EEGNet import EEGNet, DeepConvNet
from plotting import plot_curves, plot_confusion_matrix, plot_experiment_summary

EXPERIMENTS = ("baseline", "relu", "leakyrelu", "elu_alpha_05", "dropout_05", "deepconvnet")


@dataclass(frozen=True)
class Experiment:
    model: str
    activation: str = "elu"
    elu_alpha: float = 1.0
    dropout: float = 0.25


CONFIGS = {
    "baseline": Experiment(model="eegnet"),
    "relu": Experiment(model="eegnet", activation="relu"),
    "leakyrelu": Experiment(model="eegnet", activation="leakyrelu"),
    "elu_alpha_05": Experiment(model="eegnet", activation="elu", elu_alpha=0.5),
    "dropout_05": Experiment(model="eegnet", activation="elu", dropout=0.5),
    "deepconvnet": Experiment(model="deepconvnet", dropout=0.5),
}


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metrics(y_true, y_pred, summed_loss, total):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {
        "loss": float(summed_loss / total),
        "accuracy": float(100 * (tp + tn) / total),
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False):
    training = optimizer is not None
    model.train(training)
    total, summed_loss = 0, 0.0
    y_true, y_pred = [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x, labels in loader:
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with (torch.autocast("cuda", enabled=use_amp) if device.type == "cuda" else contextlib.nullcontext()):
                output = model(x)
                loss = criterion(output, labels)
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total += int(labels.size(0))
            summed_loss += float(loss.detach().item()) * int(labels.size(0))
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(output.detach().argmax(dim=1).cpu().tolist())
    return calculate_metrics(y_true, y_pred, summed_loss, total)


def build_model(config):
    if config.model == "eegnet":
        return EEGNet(activation=config.activation, elu_alpha=config.elu_alpha, dropout=config.dropout)
    return DeepConvNet(dropout=config.dropout)


def print_metrics(label, m):
    return f"{label}: acc={m['accuracy']:.2f}%, loss={m['loss']:.4f}, F1={m['f1']:.4f}"


def write_history(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_one(name, cfg, loaders, args, device):
    seed_all(args.seed)
    train_gen = getattr(loaders["train"], "generator", None)
    if train_gen is not None:
        train_gen.manual_seed(args.seed)
    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not args.no_amp))
    use_amp = device.type == "cuda" and not args.no_amp
    folder = Path(args.output) / name
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint_path = folder / "best.pt"
    history = []
    best_val_acc, best_val_loss, best_epoch = -1.0, float("inf"), -1
    epochs_without_improvement = 0
    start = time.monotonic()
    print(f"\n======== {name} ({cfg.model}) ========", flush=True)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, loaders["train"], criterion, device, optimizer, scaler, use_amp)
        va = run_epoch(model, loaders["val"], criterion, device, use_amp=use_amp)
        te = None if args.no_epoch_test else run_epoch(model, loaders["test"], criterion, device, use_amp=use_amp)
        row = {"epoch": epoch}
        for split, m in (("train", tr), ("val", va), ("test", te)):
            for key in ("accuracy", "loss", "f1"):
                row[f"{split}_{key}"] = m[key] if m is not None else None
        history.append(row)
        print(f"[{name}] Epoch {epoch:03d}/{args.epochs}: "
              f"{print_metrics('train', tr)} | {print_metrics('val', va)}" +
              (f" | {print_metrics('test diagnostic', te)}" if te else ""), flush=True)

        is_better = (va["accuracy"] > best_val_acc + 1e-10 or
                     (abs(va["accuracy"] - best_val_acc) < 1e-10 and va["loss"] < best_val_loss - 1e-10))
        if is_better:
            best_val_acc = va["accuracy"]
            best_val_loss = va["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({"model_state": model.state_dict(), "experiment": name,
                        "epoch": epoch, "validation_accuracy": best_val_acc,
                        "validation_loss": best_val_loss}, checkpoint_path)
        else:
            epochs_without_improvement += 1
        if args.patience and epochs_without_improvement >= args.patience:
            print(f"[{name}] Early stopping: {args.patience} epochs without validation improvement.")
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    final_val = run_epoch(model, loaders["val"], criterion, device, use_amp=use_amp)
    final_test = run_epoch(model, loaders["test"], criterion, device, use_amp=use_amp)
    observed_peak_test = max((h["test_accuracy"] for h in history if h["test_accuracy"] is not None), default=None)
    result = {
        "experiment": name,
        "model": cfg.model,
        "settings": {"activation": cfg.activation, "elu_alpha": cfg.elu_alpha, "dropout": cfg.dropout},
        "selected_epoch_by_validation_accuracy": best_epoch,
        "final_validation": final_val,
        "final_test": final_test,
        "highest_observed_epoch_test_accuracy_diagnostic": observed_peak_test,
        "epochs_completed": len(history),
        "train_seconds": round(time.monotonic() - start, 2),
        "checkpoint": str(checkpoint_path.as_posix()),
        "protocol_note": "Checkpoint and experiment selection use VALIDATION only; per-epoch TEST is diagnostic for the assignment plots.",
        "smoke_test": bool(args.smoke_test),
    }
    write_history(folder / "history.csv", history)
    save_json(folder / "metrics.json", result)
    plot_curves(history, folder, name)
    plot_confusion_matrix(final_test["confusion_matrix"], folder / "confusion_matrix.png", f"{name} - final test")
    print(f"[{name}] FINAL (validation-selected epoch {best_epoch}) "
          f"Test accuracy={final_test['accuracy']:.2f}%, F1={final_test['f1']:.4f}", flush=True)
    if observed_peak_test is not None:
        print(f"[{name}] Highest observed per-epoch TEST accuracy="
              f"{observed_peak_test:.2f}% (diagnostic only; NOT model selection)", flush=True)
    return result


def _make_smoke_loaders(loaders, batch_size):
    """Tiny CPU/GPU functional test: DO NOT treat metrics as assignment results."""
    limits = {"train": 64, "val": 32, "test": 32}
    out = {}
    for split, loader in loaders.items():
        n = min(limits[split], len(loader.dataset))
        subset = Subset(loader.dataset, range(n))
        out[split] = DataLoader(subset, batch_size=min(batch_size, 16), shuffle=(split == "train"))
    return out


def main():
    p = argparse.ArgumentParser(description="Lab2: EEGNet ablation experiments + bonus DeepConvNet")
    p.add_argument("--data-dir", default="data", help="Folder containing S4b_*.npz and X11b_*.npz")
    p.add_argument("--output", default="results")
    p.add_argument("--experiments", nargs="+", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    p.add_argument("--epochs", type=int, default=100, help="Max epochs per experiment; TA example uses 150")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001, help="TA reference: 0.01; safer starting point: 0.001")
    p.add_argument("--weight-decay", type=float, default=0.001)
    p.add_argument("--patience", type=int, default=25, help="Early stopping on validation accuracy/loss; 0 disables")
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0, help="Use 0 for reliable Windows execution")
    p.add_argument("--normalize", action="store_true", help="Optional: channel z-score fitted only on train")
    p.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision")
    p.add_argument("--no-epoch-test", action="store_true", help="Disable diagnostic TEST curves for stricter test holdout")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--smoke-test", action="store_true", help="Functional test only; uses small subset (NOT real results)")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.patience < 0:
        p.error("epochs, batch-size, lr must be >0 and patience >=0")
    if not 0 < args.val_fraction < 0.5:
        p.error("val-fraction must be between 0 and 0.5")
    seed_all(args.seed)
    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}; PyTorch {torch.__version__}", flush=True)
    if args.smoke_test:
        print("SMOKE TEST ONLY: numbers from this run must NOT be submitted as real results.", flush=True)
        args.epochs = min(args.epochs, 2)
        args.patience = 0
    loaders, counts = build_loaders(
        args.data_dir, args.batch_size, args.val_fraction, args.seed,
        args.num_workers, pin_memory=(device.type == "cuda"), normalize=args.normalize,
    )
    print("Dataset counts:", json.dumps(counts, indent=2), flush=True)
    if args.smoke_test:
        loaders = _make_smoke_loaders(loaders, args.batch_size)
        if args.output == "results":
            args.output = "smoke_results"
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "experiment_config.json", {**vars(args), "device": str(device), "dataset_counts": counts})
    rows = []
    summaries = {}
    for name in args.experiments:
        res = train_one(name, CONFIGS[name], loaders, args, device)
        summaries[name] = res
        m = res["final_test"]
        rows.append({
            "experiment": name, "model": res["model"], "activation": res["settings"]["activation"],
            "elu_alpha": res["settings"]["elu_alpha"], "dropout": res["settings"]["dropout"],
            "selected_epoch": res["selected_epoch_by_validation_accuracy"],
            "validation_accuracy": res["final_validation"]["accuracy"],
            "final_test_accuracy": m["accuracy"], "final_test_f1": m["f1"],
            "final_test_precision": m["precision"], "final_test_recall": m["recall"],
            "observed_peak_test_accuracy_diagnostic": res["highest_observed_epoch_test_accuracy_diagnostic"],
            "train_seconds": res["train_seconds"],
        })
    save_json(output / "summary.json", summaries)
    write_history(output / "experiment_summary.csv", rows)
    plot_experiment_summary(rows, output)
    best = sorted(rows, key=lambda x: (-x["validation_accuracy"], x["experiment"]))[0]
    save_json(output / "best_config.json", {"selected_experiment": best["experiment"],
        "validation_accuracy": best["validation_accuracy"],
        "note": "Hyperparameter comparison is based on validation accuracy only."})
    print("\nAll experiments completed. Selected by Validation Accuracy:", best["experiment"])
    print("Results:", output.resolve())
    if args.smoke_test:
        print("WARNING: smoke-test results are not official experiment results.")


if __name__ == "__main__":
    main()
