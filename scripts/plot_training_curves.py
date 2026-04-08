import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path

def parse_filename(path):
    """example name: A_stereo__semistaticsim_horizontal_100cm_bearing_nll_real__seed0_metrics.pth"""
    name = path.name.replace("_metrics.pth", "")
    parts = name.split("__")

    exp = parts[0]
    sim = parts[1]
    seed_str = parts[2]

    seed = int(seed_str.replace("seed", ""))

    sim_parts = sim.split("_")
    orientation = sim_parts[1]
    baseline = sim_parts[2]

    return exp, orientation, baseline, seed

def load_metrics(path):
    x = torch.load(path, map_location="cpu")
    epochs = np.array(x["epochs"])
    train_loss = np.array(x["train"]["loss"], dtype=float)
    val_loss = np.array(x["val"]["loss"], dtype=float)

    eps = 1e-8
    train_loss = np.maximum(train_loss, eps)
    val_loss = np.maximum(val_loss, eps)

    assert len(epochs) == len(train_loss) == len(val_loss)

    best_idx = int(np.argmin(val_loss))

    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_epoch": int(epochs[best_idx]),
        "best_val_loss": float(val_loss[best_idx]),
    }

if __name__ == "__main__":
    root = Path("~/scratch/stereo-UQ/checkpoints").expanduser()
    files = sorted(root.rglob("A_stereo__semistaticsim_horizontal_100cm_bearing_nll_real__seed*_metrics.pth"))

    if len(files) == 0:
        raise RuntimeError("No matching metrics files found.")

    records = []
    for f in files:
        exp, orientation, baseline, seed = parse_filename(f)
        metrics = load_metrics(f)
        records.append({
            "exp": exp,
            "orientation": orientation,
            "baseline": baseline,
            "seed": seed,
            "path": str(f),
            **metrics,
        })

    epochs = records[0]["epochs"]

    train_stack = np.stack([r["train_loss"] for r in records], axis=0)
    val_stack = np.stack([r["val_loss"] for r in records], axis=0)

    train_mean = train_stack.mean(axis=0)
    train_std = train_stack.std(axis=0)

    val_mean = val_stack.mean(axis=0)
    val_std = val_stack.std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, train_mean, label="train mean")
    ax.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.2)

    ax.plot(epochs, val_mean, linestyle="--", label="val mean")
    ax.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.2)

    # # optional: overlay faint individual seed curves
    # for r in records:
    #     ax.plot(r["epochs"], r["train_loss"], alpha=0.2, linewidth=1)
    #     ax.plot(r["epochs"], r["val_loss"], alpha=0.2, linewidth=1, linestyle="--")

    ax.set_title(f"{records[0]['exp']} | {records[0]['orientation']} | {records[0]['baseline']} | mean ± std over seeds")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")