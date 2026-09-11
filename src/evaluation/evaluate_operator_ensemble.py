"""
Reproduces and extends Operator_Identification.ipynb's own comparison
experiments: loads the per-nu FNO1D operator models (cf.
training/fno_training_operator_1d.py), builds EnsembleFNOOperator subsets
(a pair, a trio, all models), and for each subset plots the relative L2
error, across every nu in nu_list, of:
  - "Ensemble (linear regression)": the post-hoc N0_hat + nu*N1_hat
    reconstruction from independently-trained per-nu models
  - "Hyper-nu (joint)": a single FNO1D_hyper trained JOINTLY on that exact
    same subset (cf. training/fno_training_operator_1d_hyper.py) -- the two
    approaches share identical training data, so this is a direct
    apples-to-apples comparison of "combine independent models after the
    fact" vs. "share weights across nu from the start"
  - "Unique model": that nu's own individually-trained model (upper bound
    on what a single-nu model can do).

Also runs one extra test not in that notebook: for EVERY nu in nu_list, an
ensemble built from just its two NEAREST neighbors in nu_list (adaptive,
re-picked per target nu) rather than one fixed pair for the whole sweep --
local interpolation (or, at the ends of the range, extrapolation) instead
of a single global regression.

Evaluated on test_traj (held out, never trained on by any model) rather
than the notebook's own re-use of its train+test snapshots -- see
ks_operator_dataset.py's own split= parameter.

Every figure's underlying numbers are saved alongside it as a same-named
.npz (nu_list, per-approach error arrays, etc.) -- rsync the whole out_dir
back and re-run this script with --replot <dir> to regenerate every PNG
locally (no torch, no checkpoints, no mesu) if the plotting style needs a
tweak later.

Usage (on a compute node -- needs the trained checkpoints and data):
    python evaluation/evaluate_operator_ensemble.py \\
        --data_dir $DATA_DIR --exp_dir KS_equation \\
        --run_dir $LOG_DIR/KS_equation \\
        --exp_name fno_ks_operator --hyper_exp_name fno_ks_operator_hyper \\
        --out_dir evaluate_operator_ensemble

Usage (locally, after rsync-ing out_dir back -- no other args needed):
    python evaluation/evaluate_operator_ensemble.py --replot evaluate_operator_ensemble
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fno.fno_1D import FNO1D
from fno.fno_1D_hyper import FNO1D_hyper
from training.ks_operator_dataset import KSOperatorDataset
from evaluation.operator_ensemble import EnsembleFNOOperator, relative_l2_error
from evaluation.correction_eval import load_checkpoint_dict

# final_model.pth only exists if a run reached its last epoch -- 13
# sequential 250-epoch trainings (run_operator_ks.sh) or the "all"-subset
# hyper run (run_operator_ks_hyper.sh) can plausibly exceed their sbatch
# --time budget partway through. min_train_loss.pth is written after every
# improving epoch, so it's the right fallback for a run that got killed
# mid-training (cf. load_checkpoint_dict's own docstring for why this is
# also robust to a checkpoint left truncated by an interrupted write).
CHECKPOINT_CANDIDATES = ["final_model.pth", "min_train_loss.pth"]


def load_operator_model(run_dir, exp_name, nu, device, k_max, width, n_layer, l=1, hidden_proj=32):
    exp_name_nu = f"{exp_name}_nu{str(nu).replace('.', 'p')}"
    model = FNO1D(input_dim=1, output_dim=1, modes=k_max, width=width, l=l,
                  n_layer=n_layer, hidden_proj=hidden_proj, device=device)
    ckpt = load_checkpoint_dict(os.path.join(run_dir, exp_name_nu), CHECKPOINT_CANDIDATES, device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).float().eval()


def load_hyper_operator_model(run_dir, exp_name, tag, device, k_max, width, n_layer, l=1, hidden_proj=32,
                              n_basis=4, param_embed_dim=32, param_hidden_dim=64, param_encoder_layers=2):
    """x_mean/x_std/y_mean/y_std and param_mean/param_std are all buffers
    (cf. fno_1D_hyper.py, common/param_conditioning.py) -- load_state_dict
    restores them, no need to recompute from the training subset here."""
    exp_name_tag = f"{exp_name}_{tag}"
    model = FNO1D_hyper(input_dim=1, output_dim=1, modes=k_max, width=width, l=l,
                        n_layer=n_layer, hidden_proj=hidden_proj, n_basis=n_basis,
                        param_embed_dim=param_embed_dim, param_hidden_dim=param_hidden_dim,
                        param_encoder_layers=param_encoder_layers, device=device)
    ckpt = load_checkpoint_dict(os.path.join(run_dir, exp_name_tag), CHECKPOINT_CANDIDATES, device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).float().eval()


def _load_test_data(data_dir, exp_dir, nu, device):
    ds = KSOperatorDataset(os.path.join(data_dir, exp_dir), nu, split="test_traj")
    return ds.u.unsqueeze(-1).to(device), ds.dudt.unsqueeze(-1).to(device)  # (N, Mx, 1) each


def plot_subset_from_arrays(nu_list, nu_error_ensemble, nu_error_unique, nu_error_hyper, subset_nus,
                            title_suffix, out_path):
    """Pure plotting, no torch/model involved -- reusable both right after
    evaluate_subset() computes these arrays, and standalone from a saved
    .npz (cf. --replot) to regenerate the figure locally without mesu."""
    has_hyper = len(nu_error_hyper) > 0
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(nu_list, nu_error_ensemble, marker="o", label="Ensemble (linear regression)")
    if has_hyper:
        ax.plot(nu_list, nu_error_hyper, marker="o", label="Hyper-nu (joint training)")
    ax.plot(nu_list, nu_error_unique, marker="o", label="Unique model (own nu)")
    all_errs = list(nu_error_ensemble) + list(nu_error_unique) + list(nu_error_hyper)
    ax.vlines(subset_nus, 0, max(all_errs), colors="red", linestyles="--", alpha=0.5,
              label="nu used to build the ensemble / train the hyper model")
    ax.set_xlabel("nu")
    ax.set_ylabel("relative L2 error (%)")
    ax.set_title(f"Operator reconstruction error -- {title_suffix}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def evaluate_subset(fno_by_nu, nu_list, subset_idx, data_dir, exp_dir, device, out_path, title_suffix,
                    hyper_model=None):
    subset_models = [fno_by_nu[nu_list[i]] for i in subset_idx]
    subset_nus = [nu_list[i] for i in subset_idx]
    ensemble = EnsembleFNOOperator(subset_models, subset_nus).to(device)

    nu_error_ensemble, nu_error_unique, nu_error_hyper = [], [], []
    with torch.no_grad():
        for nu in nu_list:
            u, dudt = _load_test_data(data_dir, exp_dir, nu, device)

            pred_ensemble = ensemble.predict(u, nu)
            nu_error_ensemble.append(relative_l2_error(pred_ensemble, dudt).item())

            pred_unique = fno_by_nu[nu](u)
            nu_error_unique.append(relative_l2_error(pred_unique, dudt).item())

            if hyper_model is not None:
                nu_tensor = torch.full((u.shape[0],), nu, dtype=torch.float32, device=device)
                pred_hyper = hyper_model(u, nu_tensor)
                nu_error_hyper.append(relative_l2_error(pred_hyper, dudt).item())

    # Raw numeric results saved alongside the PNG (same basename, .npz) --
    # lets the figure be regenerated locally (plot_subset_from_arrays, no
    # torch/model/mesu needed) if the plotting style needs tweaking later,
    # without re-running inference.
    npz_path = os.path.splitext(out_path)[0] + ".npz"
    np.savez(npz_path, nu_list=np.array(nu_list), nu_error_ensemble=np.array(nu_error_ensemble),
             nu_error_unique=np.array(nu_error_unique), nu_error_hyper=np.array(nu_error_hyper),
             subset_nus=np.array(subset_nus), title_suffix=title_suffix)

    plot_subset_from_arrays(nu_list, nu_error_ensemble, nu_error_unique, nu_error_hyper, subset_nus,
                            title_suffix, out_path)
    print(f"Saved {out_path} and {npz_path}  (nu used: {subset_nus})")
    return nu_error_ensemble, nu_error_unique, nu_error_hyper


def plot_nearest_neighbors_from_arrays(nu_list, nu_error_nn, nu_error_unique, is_extrapolation, out_path):
    """Pure plotting counterpart of evaluate_nearest_neighbors, cf.
    plot_subset_from_arrays's own docstring for why this is split out."""
    fig, ax = plt.subplots(figsize=(9, 5))
    interp_nu = [nu for nu, e in zip(nu_list, is_extrapolation) if not e]
    interp_err = [err for err, e in zip(nu_error_nn, is_extrapolation) if not e]
    extrap_nu = [nu for nu, e in zip(nu_list, is_extrapolation) if e]
    extrap_err = [err for err, e in zip(nu_error_nn, is_extrapolation) if e]
    ax.plot(nu_list, nu_error_unique, marker="o", color="gray", alpha=0.6, label="Unique model (own nu)")
    ax.plot(interp_nu, interp_err, marker="o", color="tab:blue", label="Nearest-2-neighbors (interpolation)")
    ax.plot(extrap_nu, extrap_err, marker="s", color="tab:red", linestyle="none",
           label="Nearest-2-neighbors (extrapolation, at the ends)")
    ax.set_xlabel("nu")
    ax.set_ylabel("relative L2 error (%)")
    ax.set_title("Operator reconstruction error -- nearest-2-neighbors ensemble (adaptive per target nu)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def evaluate_nearest_neighbors(fno_by_nu, nu_list, data_dir, exp_dir, device, out_path):
    """Extra test (not in the original notebook): for each target nu, build
    a 2-model ensemble from its two nearest OTHER nu in nu_list (re-picked
    per target -- not one fixed pair for the whole sweep) and see how well
    that LOCAL regression does, vs the unique model. At the two ends of
    nu_list this is genuine extrapolation (both neighbors on one side);
    everywhere else it's local interpolation -- both are shown, colored
    differently, since they're not really the same regime."""
    nu_error_nn, nu_error_unique, is_extrapolation = [], [], []
    with torch.no_grad():
        for i, nu in enumerate(nu_list):
            others = [j for j in range(len(nu_list)) if j != i]
            others_sorted = sorted(others, key=lambda j: abs(nu_list[j] - nu))
            j1, j2 = others_sorted[:2]
            neighbor_nus = [nu_list[j1], nu_list[j2]]
            extrapolation = not (min(neighbor_nus) < nu < max(neighbor_nus))
            is_extrapolation.append(extrapolation)

            ensemble = EnsembleFNOOperator([fno_by_nu[neighbor_nus[0]], fno_by_nu[neighbor_nus[1]]],
                                           neighbor_nus).to(device)
            u, dudt = _load_test_data(data_dir, exp_dir, nu, device)
            pred_nn = ensemble.predict(u, nu)
            nu_error_nn.append(relative_l2_error(pred_nn, dudt).item())

            pred_unique = fno_by_nu[nu](u)
            nu_error_unique.append(relative_l2_error(pred_unique, dudt).item())

    npz_path = os.path.splitext(out_path)[0] + ".npz"
    np.savez(npz_path, nu_list=np.array(nu_list), nu_error_nn=np.array(nu_error_nn),
             nu_error_unique=np.array(nu_error_unique), is_extrapolation=np.array(is_extrapolation))

    plot_nearest_neighbors_from_arrays(nu_list, nu_error_nn, nu_error_unique, is_extrapolation, out_path)
    print(f"Saved {out_path} and {npz_path}")
    return nu_error_nn, nu_error_unique


def replot_from_dir(out_dir):
    """Regenerates every PNG in out_dir from its sibling .npz -- no torch,
    no model checkpoints, no data_dir needed. Run this locally (e.g. after
    rsync-ing out_dir back from mesu) to restyle a plot without redoing any
    inference."""
    import glob
    npz_paths = sorted(glob.glob(os.path.join(out_dir, "*.npz")))
    if not npz_paths:
        print(f"No .npz files found in {out_dir}")
        return
    for npz_path in npz_paths:
        out_path = os.path.splitext(npz_path)[0] + ".png"
        data = np.load(npz_path, allow_pickle=True)
        if "nu_error_nn" in data:
            plot_nearest_neighbors_from_arrays(
                data["nu_list"], data["nu_error_nn"], data["nu_error_unique"], data["is_extrapolation"], out_path)
        else:
            plot_subset_from_arrays(
                data["nu_list"], data["nu_error_ensemble"], data["nu_error_unique"], data["nu_error_hyper"],
                data["subset_nus"], str(data["title_suffix"]), out_path)
        print(f"Replotted {out_path} from {npz_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replot", metavar="DIR",
                        help="Skip all inference: just regenerate every PNG in DIR from its .npz sibling "
                             "(e.g. after rsync-ing results back from mesu) and exit. All other args ignored.")
    parser.add_argument("--data_dir")
    parser.add_argument("--exp_dir", default="KS_equation")
    parser.add_argument("--run_dir", help="$LOG_DIR/KS_equation-style root (required unless --replot)")
    parser.add_argument("--exp_name", default="fno_ks_operator")
    parser.add_argument("--hyper_exp_name", default="fno_ks_operator_hyper",
                        help="Set to '' to skip the hyper-nu comparison (e.g. if not trained yet)")
    parser.add_argument("--nu_values", type=float, nargs="+",
                        default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    parser.add_argument("--k_max", type=int, default=16)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--n_fourier_layer", type=int, default=3)
    parser.add_argument("--out_dir", default="evaluate_operator_ensemble")
    args = parser.parse_args()

    if args.replot:
        replot_from_dir(args.replot)
        return

    if not args.data_dir or not args.run_dir:
        parser.error("--data_dir and --run_dir are required unless --replot is given")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    nu_list = args.nu_values

    print("Loading per-nu operator models...")
    fno_by_nu = {
        nu: load_operator_model(args.run_dir, args.exp_name, nu, device,
                                args.k_max, args.width, args.n_fourier_layer)
        for nu in nu_list
    }

    # Same subset choices as config_command_operator_ks_hyper.yaml's
    # "subsets", so the hyper-nu models below are trained on identical data.
    subsets = {
        "pair": [nu_list.index(0.4), nu_list.index(0.9)],
        "trio": [1, 5, 10],
        "quad": [nu_list.index(0.4), nu_list.index(0.7), nu_list.index(1.0), nu_list.index(1.3)],
        "all": list(range(len(nu_list))),
    }

    hyper_by_tag = {}
    if args.hyper_exp_name:
        print("Loading hyper-nu operator models...")
        for tag in subsets:
            try:
                hyper_by_tag[tag] = load_hyper_operator_model(
                    args.run_dir, args.hyper_exp_name, tag, device,
                    args.k_max, args.width, args.n_fourier_layer)
            except FileNotFoundError:
                print(f"  (no hyper-nu checkpoint for '{tag}' yet, skipping it in that plot)")

    for tag, idx in subsets.items():
        out_path = os.path.join(args.out_dir, f"ensemble_vs_unique_{tag}.png")
        evaluate_subset(fno_by_nu, nu_list, idx, args.data_dir, args.exp_dir, device, out_path,
                        title_suffix=tag, hyper_model=hyper_by_tag.get(tag))

    evaluate_nearest_neighbors(fno_by_nu, nu_list, args.data_dir, args.exp_dir, device,
                              os.path.join(args.out_dir, "ensemble_nearest_neighbors.png"))


if __name__ == "__main__":
    main()
