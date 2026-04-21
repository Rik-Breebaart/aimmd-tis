from pathlib import Path
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np


EXPECTED_KEYS = [
    "train_total",
    "test_total",
    "train_model",
    "test_model",
    "train_smooth",
    "test_smooth",
    "train_l1",
    "test_l1",
    "lr",
    "best_val",
]


def _to_serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def save_training_results(results, output_path):
    """Save training results as JSON so diagnostics can be reloaded later."""
    output_path = Path(output_path)
    payload = _to_serializable(results)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


def load_training_results(results_or_path):
    """Load training results from dict, JSON, NPY/NPZ, or PKL path."""
    if isinstance(results_or_path, dict):
        return results_or_path

    path = Path(results_or_path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find results file: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    elif suffix == ".npy":
        data = np.load(path, allow_pickle=True).item()
    elif suffix == ".npz":
        npz_data = np.load(path, allow_pickle=True)
        data = {k: npz_data[k].tolist() for k in npz_data.files}
    elif suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            data = pickle.load(handle)
    else:
        raise ValueError("Unsupported results format. Use .json, .npy, .npz, .pkl, or .pickle")

    if not isinstance(data, dict):
        raise ValueError("Loaded results must be a dictionary")
    return data


def _as_array(results, key):
    values = results.get(key, [])
    if values is None:
        values = []
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        arr = np.asarray([arr], dtype=float)
    return arr


def summarize_training(results):
    """Return concise training diagnostics values for quick reporting."""
    train = _as_array(results, "train_total")
    test = _as_array(results, "test_total")
    lr = _as_array(results, "lr")

    if train.size == 0 or test.size == 0:
        return {
            "epochs": 0,
            "best_epoch": None,
            "best_val": None,
            "final_train": None,
            "final_test": None,
            "generalization_gap": None,
            "final_lr": None,
        }

    best_epoch = int(np.nanargmin(test)) + 1
    best_val = float(np.nanmin(test))
    final_train = float(train[-1])
    final_test = float(test[-1])
    gap = float(final_test - final_train)
    final_lr = float(lr[-1]) if lr.size else None

    return {
        "epochs": int(test.size),
        "best_epoch": best_epoch,
        "best_val": best_val,
        "final_train": final_train,
        "final_test": final_test,
        "generalization_gap": gap,
        "final_lr": final_lr,
    }


def plot_training_diagnostics(results_or_path, stage_name="training", save_path=None, show=True, smoothness_weight=0.0, l1_weight=0.0):
    """Create a clear 2x2 diagnostics figure from training results."""
    results = load_training_results(results_or_path)

    train_total = _as_array(results, "train_total")
    test_total = _as_array(results, "test_total")
    train_model = _as_array(results, "train_model")
    test_model = _as_array(results, "test_model")
    train_smooth = _as_array(results, "train_smooth")
    test_smooth = _as_array(results, "test_smooth")
    train_l1 = _as_array(results, "train_l1")
    test_l1 = _as_array(results, "test_l1")
    lr = _as_array(results, "lr")

    n_epochs = max(train_total.size, test_total.size)
    if n_epochs == 0:
        raise ValueError("No epoch losses found in results")

    epochs = np.arange(1, n_epochs + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Training Diagnostics: {stage_name}", fontsize=14)

    ax = axes[0, 0]
    if train_total.size:
        ax.plot(np.arange(1, train_total.size + 1), train_total, label="Train total", lw=2)
    if test_total.size:
        ax.plot(np.arange(1, test_total.size + 1), test_total, label="Test total", lw=2)
        best_epoch = int(np.nanargmin(test_total)) + 1
        ax.axvline(best_epoch, color="tab:green", ls="--", alpha=0.8, label=f"Best epoch {best_epoch}")
    ax.set_title("Total Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    if train_model.size:
        ax.plot(np.arange(1, train_model.size + 1), train_model, label="Train model", lw=1.8)
    if test_model.size:
        ax.plot(np.arange(1, test_model.size + 1), test_model, label="Test model", lw=1.8)
    if train_smooth.size and smoothness_weight > 0.0:
        ax.plot(np.arange(1, train_smooth.size + 1), train_smooth*smoothness_weight, label="Train smooth", lw=1.4)
    if test_smooth.size and smoothness_weight > 0.0:
        ax.plot(np.arange(1, test_smooth.size + 1), test_smooth*smoothness_weight, label="Test smooth", lw=1.4)
    if train_l1.size and l1_weight > 0.0:
        ax.plot(np.arange(1, train_l1.size + 1), train_l1*l1_weight, label="Train L1", lw=1.2)
    if test_l1.size and l1_weight > 0.0:
        ax.plot(np.arange(1, test_l1.size + 1), test_l1*l1_weight, label="Test L1", lw=1.2)
    ax.set_title("Loss Components")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Component value")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    if lr.size:
        ax.plot(np.arange(1, lr.size + 1), lr, color="tab:purple", lw=2)
        if np.all(lr > 0):
            ax.set_yscale("log")
    ax.set_title("Learning Rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if train_total.size and test_total.size:
        gap = test_total[: min(train_total.size, test_total.size)] - train_total[: min(train_total.size, test_total.size)]
        ax.plot(np.arange(1, gap.size + 1), gap, color="tab:red", lw=2)
        ax.axhline(0.0, color="black", ls="--", lw=1)
    ax.set_title("Generalization Gap (Test - Train)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gap")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes, summarize_training(results)
