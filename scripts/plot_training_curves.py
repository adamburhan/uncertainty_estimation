import matplotlib.pyplot as plt
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
    epochs = x["epochs"]
    train_loss = x["train"]["loss"]
    val_loss = x["val"]["loss"]

    best_idx = min(range(len(val_loss)), key=lambda i: val_loss[i])

    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_epoch": epochs[best_idx],
        "best_val_loss": val_loss[best_idx],
    }

if __name__ == "__main__":
    root = Path("~/scratch/stereo-UQ/checkpoints/A_stereo__semistaticsim_horizontal_100cm_bearing_nll_real__seed0").expanduser()
    f = root / "A_stereo__semistaticsim_horizontal_100cm_bearing_nll_real__seed0_metrics.pth"

    exp, orientation, baseline, seed = parse_filename(f)
    metrics = load_metrics(f)

    epochs = metrics["epochs"]
    train_loss = metrics["train_loss"]
    val_loss = metrics["val_loss"]
    best_epoch = metrics["best_epoch"]
    best_val_loss = metrics["best_val_loss"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, linestyle="-", label=f"train s{seed}")
    ax.plot(epochs, val_loss, linestyle="--", label=f"val s{seed}")
    ax.scatter([best_epoch], [best_val_loss], s=20, label="best val")

    ax.set_title(f"{exp} | {orientation} | {baseline} | seed {seed}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()