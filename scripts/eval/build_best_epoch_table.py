"""Build a wide best-epoch table: one row per (exp, config), seeds as columns.

Input:  outputs/best_epochs_{A,B,C,D}.csv
Output: outputs/best_epoch_table.csv
        outputs/best_epoch_table.md
"""

import csv
from collections import defaultdict
from pathlib import Path

OUTPUT_DIR = Path("outputs")
INPUT_CSVS = ["best_epochs_A.csv", "best_epochs_B.csv",
              "best_epochs_C.csv", "best_epochs_D.csv"]

EXP_ORDER = ["A_stereo", "B_falsif", "C_loss", "D_3dctrl"]
SEEDS = [0, 42, 2026]


def sort_key(key):
    exp, orientation, baseline = key
    return (
        EXP_ORDER.index(exp) if exp in EXP_ORDER else 99,
        0 if orientation == "horizontal" else 1,
        int(baseline.replace("cm", "")),
    )


def main():
    # (exp, orientation, baseline) -> {seed: (best_epoch, best_val_loss)}
    grouped: dict = defaultdict(dict)
    for fname in INPUT_CSVS:
        path = OUTPUT_DIR / fname
        if not path.exists():
            print(f"  skipping missing {path}")
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                key = (r["exp"], r["orientation"], r["baseline"])
                grouped[key][int(r["seed"])] = (
                    int(r["best_epoch"]),
                    float(r["best_val_loss"]),
                )

    if not grouped:
        raise RuntimeError("No best_epochs_*.csv files found in outputs/")

    keys = sorted(grouped.keys(), key=sort_key)

    # CSV: one row per (exp, config), columns for each seed's epoch and loss
    fields = ["exp", "orientation", "baseline"]
    for s in SEEDS:
        fields += [f"seed{s}_epoch", f"seed{s}_val_loss"]

    csv_out = OUTPUT_DIR / "best_epoch_table.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key in keys:
            exp, orientation, baseline = key
            row = {"exp": exp, "orientation": orientation, "baseline": baseline}
            for s in SEEDS:
                if s in grouped[key]:
                    ep, loss = grouped[key][s]
                    row[f"seed{s}_epoch"] = ep
                    row[f"seed{s}_val_loss"] = f"{loss:.3f}"
                else:
                    row[f"seed{s}_epoch"] = ""
                    row[f"seed{s}_val_loss"] = ""
            w.writerow(row)
    print(f"Wrote {csv_out}  ({len(keys)} rows)")

    # Markdown: same shape but each seed cell is "epoch / loss"
    md_out = OUTPUT_DIR / "best_epoch_table.md"
    with open(md_out, "w") as f:
        header = "| Exp | Orientation | Baseline | " + " | ".join(f"seed={s} (epoch / loss)" for s in SEEDS) + " |\n"
        sep = "|---|---|---|" + "---|" * len(SEEDS) + "\n"
        f.write(header)
        f.write(sep)
        for key in keys:
            exp, orientation, baseline = key
            cells = [exp, orientation, baseline]
            for s in SEEDS:
                if s in grouped[key]:
                    ep, loss = grouped[key][s]
                    cells.append(f"{ep} / {loss:.3f}")
                else:
                    cells.append("—")
            f.write("| " + " | ".join(cells) + " |\n")
    print(f"Wrote {md_out}\n")
    print(md_out.read_text())


if __name__ == "__main__":
    main()
