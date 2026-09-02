"""
Detailed metric + parameter-count comparison table for FNO vs UNet
emulators (Kolmogorov Re90) -- complements plot_training_metrics.py's
generic loss-only summary with the specific rmse/mae/relative_rmse/
relative_mae breakdown (train AND test) plus each model's parameter count.

Reuses evaluation.correction_eval.load_emulator_fno and
evaluation.grid_independence_eval.load_emulator_unet -- both already
instantiate the model from its config.yaml AND load its checkpoint, so the
returned model's own count_parameters_per_module() gives the exact
parameter count actually trained, no separate re-derivation. Also reuses
evaluation.plot_training_metrics.read_metrics for logs/metrics.csv parsing
(same file every run already has).

The reported row per run is taken at that run's OWN best epoch (lowest Te
loss), matching plot_training_metrics.py's own "best_epoch" convention --
not the last logged epoch, which can be several epochs past the actual
best one once early stopping's patience window has elapsed.

Usage (from thermalizer/src):
    python evaluation/compare_unet_fno_metrics.py \\
        --fno_runs /scratch/cayuelam/fno/runs/kolmogorov/Re90/16modes_emul_RE90 \\
        --unet_runs /scratch/cayuelam/fno/runs/kolmogorov/Re90/unet_delta_RE90 \\
                    /scratch/cayuelam/fno/runs/kolmogorov/Re90/unet_state_RE90 \\
        --out_csv comparison_metrics_table.csv
"""

import argparse
import csv
import os
import sys

import torch
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.correction_eval import load_emulator_fno
from evaluation.grid_independence_eval import load_emulator_unet
from evaluation.plot_training_metrics import read_metrics

METRIC_COLS = ["rmse", "mae", "relative_rmse", "relative_mae"]


def _last_run_block(cols):
    """Older thermalizer runs' logs/metrics.csv can hold MULTIPLE
    concatenated training sessions: MetricLogger used to always append
    (fixed in training/metric_logger.py, but that doesn't retroactively
    clean up files written before the fix), so re-launching a script
    against an exp_name whose metrics.csv already existed silently glued a
    fresh epoch-1-onward block onto whatever history was already there, no
    marker between sessions. Detect the start of the LAST such block (the
    last place Epoch resets to a value <= the previous row's) and slice
    every column to keep only that block -- so a "best epoch" is always
    relative to the run that's actually being asked about, not a stale
    unrelated one mixed into the same file."""
    epochs = cols["Epoch"]
    start = 0
    for i in range(1, len(epochs)):
        if epochs[i] <= epochs[i - 1]:
            start = i
    if start > 0:
        print(f"  Note: logs/metrics.csv holds {start} row(s) from an earlier, unrelated "
              f"session before this one (Epoch resets at row {start}) -- ignoring them.", flush=True)
    return {k: v[start:] for k, v in cols.items()}


def summarize_run(run_dir, model):
    csv_path = os.path.join(run_dir, "logs", "metrics.csv")
    cols = read_metrics(csv_path)
    if not cols or "Epoch" not in cols:
        raise SystemExit(f"metrics.csv illisible/vide : {csv_path}")
    cols = _last_run_block(cols)

    te_loss = cols.get("Te loss")
    best_idx = min(range(len(te_loss)), key=lambda i: te_loss[i]) if te_loss else len(cols["Epoch"]) - 1

    row = {
        "name": os.path.basename(run_dir.rstrip("/")),
        "n_params": model.count_parameters_per_module()["total"],
        "n_epochs_trained": int(cols["Epoch"][-1]),
        "best_epoch": int(cols["Epoch"][best_idx]),
    }
    for m in METRIC_COLS:
        for split in ("Tr", "Te"):
            key = f"{split} {m}"
            row[key] = cols[key][best_idx] if key in cols else None
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fno_runs", nargs="*", default=[])
    parser.add_argument("--unet_runs", nargs="*", default=[])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_csv", default="comparison_metrics_table.csv")
    args = parser.parse_args()

    if not args.fno_runs and not args.unet_runs:
        raise SystemExit("Rien a comparer -- passe au moins --fno_runs ou --unet_runs.")

    device = torch.device(args.device)
    rows = []
    for run_dir in args.fno_runs:
        print(f"Loading FNO run: {run_dir}", flush=True)
        model, _ = load_emulator_fno(run_dir, device)
        rows.append(summarize_run(run_dir, model))
    for run_dir in args.unet_runs:
        print(f"Loading UNet run: {run_dir}", flush=True)
        model, _ = load_emulator_unet(run_dir, device)
        rows.append(summarize_run(run_dir, model))

    metric_field_order = [f"{split} {m}" for m in METRIC_COLS for split in ("Tr", "Te")]
    fieldnames = ["name", "n_params", "n_epochs_trained", "best_epoch"] + metric_field_order

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})

    table = [
        [r["name"], f"{r['n_params']:,}", r["n_epochs_trained"], r["best_epoch"]]
        + [f"{r[k]:.4f}" if r.get(k) is not None else "-" for k in metric_field_order]
        for r in rows
    ]
    headers = ["Run", "Params", "Epochs", "Best epoch"] + metric_field_order
    print("\n" + tabulate(table, headers=headers, tablefmt="simple"), flush=True)
    print(f"\nSaved: {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
