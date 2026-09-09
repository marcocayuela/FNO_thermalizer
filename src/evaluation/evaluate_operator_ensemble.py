"""
Reproduces Operator_Identification.ipynb's own comparison experiments:
loads the per-nu FNO1D operator models (cf. training/fno_training_operator_1d.py),
builds a few EnsembleFNOOperator subsets (2 models, 3 models, all models --
same subset choices as that notebook's "2 models"/"3 models"/"All models"
sections), and for each subset plots the relative L2 error, across every
nu in nu_list, of:
  - "Ensemble model": the linear-regression reconstruction N0_hat + nu*N1_hat
  - "Unique model": that nu's own individually-trained model (upper bound
    on what a single-nu model can do -- the ensemble is trying to match or
    beat this INCLUDING at nu values it wasn't itself trained on).

Evaluated on test_traj (held out, never trained on by any model) rather
than the notebook's own re-use of its train+test snapshots -- see
ks_operator_dataset.py's own split= parameter.

Usage (on a compute node -- needs the trained checkpoints and data):
    python evaluation/evaluate_operator_ensemble.py \\
        --data_dir $DATA_DIR --exp_dir KS_equation \\
        --run_dir $LOG_DIR/KS_equation --exp_name fno_ks_operator \\
        --out_dir evaluate_operator_ensemble
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fno.fno_1D import FNO1D
from training.ks_operator_dataset import KSOperatorDataset
from evaluation.operator_ensemble import EnsembleFNOOperator, relative_l2_error


def load_operator_model(run_dir, exp_name, nu, device, k_max, width, n_layer, l=1, hidden_proj=32):
    exp_name_nu = f"{exp_name}_nu{str(nu).replace('.', 'p')}"
    ckpt_path = os.path.join(run_dir, exp_name_nu, "model_weights", "final_model.pth")
    model = FNO1D(input_dim=1, output_dim=1, modes=k_max, width=width, l=l,
                  n_layer=n_layer, hidden_proj=hidden_proj, device=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).float().eval()


def evaluate_subset(fno_by_nu, nu_list, subset_idx, data_dir, exp_dir, device, out_path, title_suffix):
    subset_models = [fno_by_nu[nu_list[i]] for i in subset_idx]
    subset_nus = [nu_list[i] for i in subset_idx]
    ensemble = EnsembleFNOOperator(subset_models, subset_nus).to(device)

    nu_error_ensemble, nu_error_unique = [], []
    with torch.no_grad():
        for nu in nu_list:
            ds = KSOperatorDataset(os.path.join(data_dir, exp_dir), nu, split="test_traj")
            u = ds.u.unsqueeze(-1).to(device)       # (N, Mx, 1)
            dudt = ds.dudt.unsqueeze(-1).to(device)  # (N, Mx, 1)

            pred_ensemble = ensemble.predict(u, nu)
            err_ensemble = relative_l2_error(pred_ensemble, dudt).item()
            nu_error_ensemble.append(err_ensemble)

            pred_unique = fno_by_nu[nu](u)
            err_unique = relative_l2_error(pred_unique, dudt).item()
            nu_error_unique.append(err_unique)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(nu_list, nu_error_ensemble, marker="o", label="Ensemble model (linear regression)")
    ax.plot(nu_list, nu_error_unique, marker="o", label="Unique model (own nu)")
    ax.vlines(subset_nus, 0, max(max(nu_error_ensemble), max(nu_error_unique)),
              colors="red", linestyles="--", alpha=0.5, label="nu used to build the ensemble")
    ax.set_xlabel("nu")
    ax.set_ylabel("relative L2 error (%)")
    ax.set_title(f"Operator reconstruction error -- {title_suffix}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}  (ensemble nus: {subset_nus})")
    return nu_error_ensemble, nu_error_unique


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--exp_dir", default="KS_equation")
    parser.add_argument("--run_dir", required=True, help="$LOG_DIR/KS_equation-style root holding <exp_name>_nu<X>/")
    parser.add_argument("--exp_name", default="fno_ks_operator")
    parser.add_argument("--nu_values", type=float, nargs="+",
                        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    parser.add_argument("--k_max", type=int, default=16)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--n_fourier_layer", type=int, default=3)
    parser.add_argument("--out_dir", default="evaluate_operator_ensemble")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    nu_list = args.nu_values

    print("Loading per-nu operator models...")
    fno_by_nu = {
        nu: load_operator_model(args.run_dir, args.exp_name, nu, device,
                                args.k_max, args.width, args.n_fourier_layer)
        for nu in nu_list
    }

    # Same subset choices as Operator_Identification.ipynb's own "2 models"/
    # "3 models"/"All models" sections (indices into nu_list, 0-based).
    subsets = {
        "2 models (interpolation)": [5, 8],
        "3 models": [1, 5, 10],
        "all models": list(range(len(nu_list))),
    }

    for title, idx in subsets.items():
        tag = title.split()[0].replace("2", "two").replace("3", "three")
        out_path = os.path.join(args.out_dir, f"ensemble_vs_unique_{tag}.png")
        evaluate_subset(fno_by_nu, nu_list, idx, args.data_dir, args.exp_dir, device, out_path, title)


if __name__ == "__main__":
    main()
