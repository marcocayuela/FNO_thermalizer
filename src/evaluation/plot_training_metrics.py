"""
Genere des courbes d'entrainement (loss train/test, metriques, LR) a partir des
logs/metrics.csv produits par n'importe quel run thermalizer. Les emulateurs
(FNO/WNO/UNet) et les modeles de diffusion (FNO/WNO/UNet) partagent tous les
memes classes Trainer/TrainerDiffusion (training/trainer.py, training/trainer_diffusion.py),
donc le meme format de colonnes -- ce script les detecte automatiquement, pas besoin
de code specifique par architecture.

Genere aussi un tableau recapitulatif (CSV + impression console) des erreurs
train/test finales et meilleures de chaque modele, pour une comparaison rapide
sans avoir a rouvrir chaque metrics.csv individuellement.

Usage (depuis thermalizer/src) :
    # Un seul run
    python plot_training_metrics.py --run_dir ../runs_mesu/Re90/Re90/unet_state_RE90

    # Plusieurs runs precis + comparaison superposee + tableau recapitulatif
    python plot_training_metrics.py \
        --run_dir ../runs_mesu/Re90/Re90/unet_state_RE90 \
        --run_dir ../runs_mesu/Re90/Re90/unet_delta_RE90 \
        --compare_out comparison.png --summary_out summary.csv

    # Auto-decouverte de tous les runs sous un dossier (utilise en fin de job SLURM)
    python plot_training_metrics.py --runs_base "$LOG_DIR/kolmogorov/Re90" \
        --compare_out "$LOG_DIR/kolmogorov/Re90/training_comparison.png" \
        --summary_out "$LOG_DIR/kolmogorov/Re90/training_summary.csv"

Sortie : un training_curves.png dans logs/ de chaque run, une comparaison
inter-runs par "famille" (emulateur vs diffusion) si --compare_out est donne,
et un tableau recapitulatif si --summary_out est donne (ou par defaut
"training_summary.csv" dans le repertoire courant).
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tabulate import tabulate


def read_metrics(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    columns = {}
    for key in rows[0].keys():
        try:
            columns[key] = [float(r[key]) for r in rows]
        except (ValueError, TypeError):
            pass
    return columns


def discover_runs(runs_base):
    run_dirs = []
    for name in sorted(os.listdir(runs_base)):
        path = os.path.join(runs_base, name)
        if os.path.isfile(os.path.join(path, "logs", "metrics.csv")):
            run_dirs.append(path)
    return run_dirs


def plot_run(run_dir):
    csv_path = os.path.join(run_dir, "logs", "metrics.csv")
    name = os.path.basename(run_dir.rstrip("/"))

    if not os.path.exists(csv_path):
        print(f"  [ignore] pas de metrics.csv : {run_dir}", flush=True)
        return None

    cols = read_metrics(csv_path)
    if not cols or "Epoch" not in cols:
        print(f"  [ignore] metrics.csv vide/illisible : {csv_path}", flush=True)
        return None

    epoch = cols["Epoch"]

    # format emulateur : paires "Tr X" / "Te X"
    tr_te_pairs = sorted({
        k[3:] for k in cols if k.startswith("Tr ") and f"Te {k[3:]}" in cols
    })
    # format diffusion : training_loss / loss_score / loss_cat (pas de split train/test)
    is_diffusion = "training_loss" in cols

    panels = []
    if tr_te_pairs:
        loss_keys = [k for k in tr_te_pairs if "loss" in k.lower()]
        metric_keys = [k for k in tr_te_pairs if k not in loss_keys]
        if loss_keys:
            panels.append(("Loss", loss_keys))
        if metric_keys:
            panels.append(("Metriques", metric_keys))
        kind = "emulateur"
    elif is_diffusion:
        diff_keys = [k for k in ("training_loss", "loss_score", "loss_cat") if k in cols]
        panels.append(("Loss (diffusion)", diff_keys))
        kind = "diffusion"
    else:
        print(f"  [ignore] format de colonnes non reconnu : {csv_path}", flush=True)
        return None

    lr_key = "LR" if "LR" in cols else None
    n_panels = len(panels) + (1 if lr_key else 0)

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4.5))
    axes = [axes] if n_panels == 1 else list(axes)

    for ax, (title, keys) in zip(axes, panels):
        for k in keys:
            if kind == "diffusion":
                ax.plot(epoch, cols[k], label=k)
            else:
                ax.plot(epoch, cols[f"Tr {k}"], label=f"train {k}", linestyle="--")
                ax.plot(epoch, cols[f"Te {k}"], label=f"test {k}")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")

    if lr_key:
        ax = axes[-1]
        ax.plot(epoch, cols[lr_key], color="black")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_title("Learning rate")
        ax.grid(alpha=0.3, which="both")

    fig.suptitle(name)
    plt.tight_layout()

    out_path = os.path.join(run_dir, "logs", "training_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    if kind == "emulateur":
        train_loss = cols.get("Tr loss")
        test_loss = cols.get("Te loss")
    else:
        # pas de split train/test pour la diffusion : training_loss sert de
        # reference dans les deux colonnes du tableau, marque comme tel.
        train_loss = cols.get("training_loss")
        test_loss = None

    best_idx = min(range(len(test_loss)), key=lambda i: test_loss[i]) if test_loss else \
        (min(range(len(train_loss)), key=lambda i: train_loss[i]) if train_loss else None)

    summary = {
        "name": name,
        "kind": kind,
        "epoch": epoch,
        "test_loss": test_loss if test_loss else train_loss,  # utilise pour plot_comparison
        "n_epochs": len(epoch),
        "final_train_loss": train_loss[-1] if train_loss else None,
        "final_test_loss": test_loss[-1] if test_loss else None,
        "best_loss": (test_loss or train_loss)[best_idx] if best_idx is not None else None,
        "best_epoch": int(epoch[best_idx]) if best_idx is not None else None,
    }

    print(f"  -> {out_path} ({len(epoch)} epochs, {kind})", flush=True)
    return summary


def plot_comparison(summaries, kind, out_path):
    runs = [r for r in summaries if r and r["kind"] == kind and r["test_loss"]]
    if len(runs) < 2:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for r in runs:
        ax.plot(r["epoch"], r["test_loss"], label=r["name"])
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test loss" if kind == "emulateur" else "Training loss")
    ax.set_title(f"Comparaison des runs -- {kind}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparaison ({kind}) sauvegardee : {out_path}", flush=True)


def write_summary_table(summaries, out_path):
    """
    Tableau recapitulatif (CSV + impression console) des erreurs train/test
    finales et meilleures de chaque modele, individuellement -- pour la
    diffusion (pas de split train/test), final_test_loss/best_loss reprennent
    training_loss, marque explicitement dans la colonne "kind".
    """
    rows = [s for s in summaries if s]
    if not rows:
        print("Aucun run exploitable pour le tableau recapitulatif.", flush=True)
        return

    fieldnames = ["name", "kind", "n_epochs", "final_train_loss", "final_test_loss",
                 "best_loss", "best_epoch"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})

    table = [
        [r["name"], r["kind"], r["n_epochs"],
         f"{r['final_train_loss']:.3e}" if r["final_train_loss"] is not None else "-",
         f"{r['final_test_loss']:.3e}" if r["final_test_loss"] is not None else "(training_loss)",
         f"{r['best_loss']:.3e}" if r["best_loss"] is not None else "-",
         r["best_epoch"] if r["best_epoch"] is not None else "-"]
        for r in rows
    ]
    print("\n" + tabulate(table, headers=["Run", "Type", "Epochs", "Train loss (final)",
                                          "Test loss (final)", "Meilleure loss", "A l'epoch"],
                         tablefmt="simple"), flush=True)
    print(f"\nTableau recapitulatif sauvegarde : {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", action="append", default=[],
                        help="Chemin d'un run a tracer (repetable)")
    parser.add_argument("--runs_base", default=None,
                        help="Dossier contenant plusieurs runs -- decouvre automatiquement "
                             "ceux avec logs/metrics.csv")
    parser.add_argument("--compare_out", default=None,
                        help="Chemin de base pour les comparaisons inter-runs (suffixe "
                             "_emulateur/_diffusion ajoute avant l'extension)")
    parser.add_argument("--summary_out", default="training_summary.csv",
                        help="Chemin du tableau recapitulatif CSV (erreurs train/test par modele)")
    args = parser.parse_args()

    run_dirs = list(args.run_dir)
    if args.runs_base:
        run_dirs += discover_runs(args.runs_base)

    if not run_dirs:
        raise SystemExit("Aucun run a traiter : passe --run_dir ou --runs_base.")

    print(f"{len(run_dirs)} run(s) a traiter.", flush=True)
    summaries = []
    for run_dir in run_dirs:
        print(f"- {run_dir}", flush=True)
        summaries.append(plot_run(run_dir))

    if args.compare_out:
        base, ext = os.path.splitext(args.compare_out)
        for kind in ("emulateur", "diffusion"):
            plot_comparison(summaries, kind, f"{base}_{kind}{ext}")

    write_summary_table(summaries, args.summary_out)


if __name__ == "__main__":
    main()
