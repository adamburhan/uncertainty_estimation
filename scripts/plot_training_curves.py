import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
import csv

def parse_filename(path):
    """
    Example:
    A_stereo__semistaticsim_horizontal_100cm_bearing_nll_real__seed0_metrics.pth
    """
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

    epochs = np.array(x["epochs"], dtype=int)
    train_loss = np.array(x["train"]["loss"], dtype=float)
    val_loss = np.array(x["val"]["loss"], dtype=float)

    assert len(epochs) == len(train_loss) == len(val_loss)

    best_idx = int(np.argmin(val_loss))

    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_epoch": int(epochs[best_idx]),
        "best_val_loss": float(val_loss[best_idx]),
    }

def sort_key(config):
    orientation, baseline = config
    baseline_cm = int(baseline.replace("cm", ""))
    orientation_order = 0 if orientation == "horizontal" else 1
    return (orientation_order, baseline_cm)

if __name__ == "__main__":
    root = Path("~/scratch/stereo-UQ/checkpoints").expanduser()
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)


    files = sorted(root.rglob("C_loss__semistaticsim_*_pixel_nll_real__seed*_metrics.pth"))

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

    # Save per-seed best epoch table
    csv_path = output_dir / "best_epochs_C.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["exp", "orientation", "baseline", "seed", "best_epoch", "best_val_loss", "path"]
        )
        writer.writeheader()
        for r in records:
            writer.writerow({
                "exp": r["exp"],
                "orientation": r["orientation"],
                "baseline": r["baseline"],
                "seed": r["seed"],
                "best_epoch": r["best_epoch"],
                "best_val_loss": r["best_val_loss"],
                "path": r["path"],
            })

    # Group by config
    grouped = defaultdict(list)
    for r in records:
        grouped[(r["orientation"], r["baseline"])].append(r)

    configs = [
        # ("horizontal", "5cm"),
        # ("horizontal", "10cm"),
        # ("horizontal", "20cm"),
        # ("horizontal", "50cm"),
        # ("horizontal", "100cm"),
        # ("vertical", "5cm"),
        # ("vertical", "10cm"),
        # ("vertical", "20cm"),
        # ("vertical", "50cm"),
        ("vertical", "100cm"),
    ]

    fig, axes = plt.subplots(nrows=len(configs), ncols=1, figsize=(10, 2.7 * len(configs)), sharex=True)

    if len(configs) == 1:
        axes = [axes]

    for ax, config in zip(axes, configs):
        orientation, baseline = config
        group = sorted(grouped[config], key=lambda r: r["seed"])

        if len(group) == 0:
            ax.set_title(f"{orientation} | {baseline} | MISSING")
            ax.axis("off")
            continue

        # Optional safety check: all seeds should have same epoch schedule
        ref_epochs = group[0]["epochs"]
        for r in group[1:]:
            assert np.array_equal(ref_epochs, r["epochs"]), f"Epoch mismatch in {config}"

        train_stack = np.stack([r["train_loss"] for r in group], axis=0)
        val_stack = np.stack([r["val_loss"] for r in group], axis=0)

        train_mean = train_stack.mean(axis=0)
        train_std = train_stack.std(axis=0)

        val_mean = val_stack.mean(axis=0)
        val_std = val_stack.std(axis=0)

        epochs = ref_epochs

        ax.plot(epochs, train_mean, label="train mean")
        ax.fill_between(epochs, train_mean - train_std, train_mean + train_std, alpha=0.2)

        ax.plot(epochs, val_mean, linestyle="--", label="val mean")
        ax.fill_between(epochs, val_mean - val_std, val_mean + val_std, alpha=0.2)

        mean_best_idx = int(np.argmin(val_mean))
        mean_best_epoch = int(epochs[mean_best_idx])
        mean_best_val = float(val_mean[mean_best_idx])
        ax.scatter([mean_best_epoch], [mean_best_val], s=18, label="best val mean")

        ax.set_title(f"C_stereo | {orientation} | {baseline}")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.4)

        # only show legend once to reduce clutter
        if config == configs[0]:
            ax.legend()

    axes[-1].set_xlabel("Epoch")
    plt.tight_layout()

    fig_path = output_dir / "training_curves_C.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {fig_path}")
    print(f"Saved CSV to: {csv_path}")