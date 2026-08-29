"""
Sanity check for a trained Re-conditioned diffusion corrector (cf.
fno/fno_2D_classifier_concat.py::FNO2D_classifier_concat): verifies the
model actually learned to USE Re, rather than ignoring the extra channel/
feature and just fitting a Re-blind average.

For each Re in --re: takes real snapshots, artificially noises them to KNOWN
diffusion timesteps t, and asks the classifier head to guess t back -- once
with the CORRECT Re, once with each of the OTHER (wrong) Re. If Re-
conditioning is doing its job (judging "on-manifold for this Re"), the
predicted t should be noticeably more accurate with the correct Re than with
a wrong one. If accuracy is about the same either way, the model isn't
really using the Re channel.

Usage (on the cluster, needs the trained checkpoint + raw multi-Re data):
    python verify_diffusion_concat.py \
        --run_dir $LOG_DIR/kolmogorov_parametric/diffusion_concat_re50-90_2re \
        --re 50 90 \
        --n_snapshots 20 \
        --out_dir diffusion_concat_verification
"""

import argparse
import csv
import glob
import os
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# fno/ and training/ are siblings of evaluation/ (this file's directory) under
# src/ -- add src/ itself to the path so this script works regardless of cwd
# or PYTHONPATH, whether run as `python evaluation/verify_diffusion_concat.py`
# from src/ or `python src/evaluation/verify_diffusion_concat.py` from elsewhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fno.fno_2D_classifier_concat import FNO2D_classifier_concat
from training.DiffusionModel import Diffusion
from training.dataset_manager import format_re

DATA_DIR = os.getenv("DATA_DIR", "../data")


def load_diffusion(run_dir, device):
    with open(os.path.join(run_dir, "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    base = FNO2D_classifier_concat(
        input_dim=cfg["input_dim"], output_dim=cfg["output_dim"],
        modes_x=cfg["k_max_x"], modes_y=cfg["k_max_y"], width=cfg["width"], l=cfg["l"],
        n_layer=cfg["n_fourier_layer"], hidden_proj=cfg.get("hidden_proj"), mlp=cfg.get("mlp", True),
        layers_mlp=cfg.get("layers_mlp"), class_mlp_layers=cfg.get("class_mlp_layers"),
        n_cat=cfg["timesteps"], param_mean=0.0, param_std=1.0, device=device,
    ).to(device).float()

    diffusion = Diffusion(model=base, timesteps=cfg["timesteps"],
                          noise_sampling_coeff=cfg.get("noise_sampling_coeff"))
    for name in ("final_model.pth", "min_train_loss.pth"):
        path = os.path.join(run_dir, "model_weights", name)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=device, weights_only=False)
            diffusion.load_state_dict(ckpt["model_state_dict"])
            break
    else:
        raise FileNotFoundError(f"No checkpoint found in {run_dir}/model_weights")
    return diffusion.to(device).eval(), cfg


def load_snapshots(exp_dir, re, n_snapshots, ds):
    files = sorted(glob.glob(os.path.join(DATA_DIR, exp_dir, f"Re{format_re(re)}", "test_traj", "*.h5")))
    if not files:
        raise FileNotFoundError(f"No test_traj run for Re={re} in {os.path.join(DATA_DIR, exp_dir)}")
    with h5py.File(files[0], "r") as f:
        n_frames = f["velocity_field"].shape[0]
        idx = np.linspace(0, n_frames - 1, n_snapshots).astype(int)
        frames = f["velocity_field"][sorted(set(idx.tolist())), ::ds, ::ds]
    return torch.from_numpy(frames).float()  # (n, H, W, C)


@torch.no_grad()
def predict_noise_level(diffusion, x0, t_true, re_value, device):
    """x0: (H, W, C) single clean snapshot. Noises it to t_true, classifies
    with the given re_value, returns the predicted timestep (int)."""
    x = x0.unsqueeze(0).to(device)
    t = torch.tensor([t_true], dtype=torch.long, device=device)
    noise = torch.randn_like(x)
    x_t = diffusion._forward_diffusion(x, t, noise)
    re = torch.tensor([re_value], dtype=torch.float32, device=device)
    _, logits = diffusion.model(x_t, re, True)
    return int(torch.softmax(logits, dim=-1).argmax(dim=-1).item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--re", type=float, nargs="+", required=True,
                        help="At least 2 Re values (each is tested with its own correct "
                        "conditioning and with every other Re as a wrong condition)")
    parser.add_argument("--n_snapshots", type=int, default=20)
    parser.add_argument("--n_t_probes", type=int, default=8,
                        help="Number of true noise levels t tested per snapshot, evenly "
                        "spaced across [0, timesteps)")
    parser.add_argument("--out_dir", default="diffusion_concat_verification")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if len(args.re) < 2:
        raise SystemExit("Need at least 2 --re values to compare correct vs wrong conditioning.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                              else "cpu")
    else:
        device = torch.device(args.device)

    os.makedirs(args.out_dir, exist_ok=True)
    diffusion, cfg = load_diffusion(args.run_dir, device)
    ds = cfg.get("ds", 2)
    t_probes = np.linspace(0, cfg["timesteps"] - 1, args.n_t_probes).astype(int)

    rows = []
    for re in args.re:
        snapshots = load_snapshots(cfg["exp_dir"], re, args.n_snapshots, ds)
        print(f"- Re={re:g}: {snapshots.shape[0]} snapshots x {len(t_probes)} noise levels", flush=True)

        for x0 in snapshots:
            for t_true in t_probes:
                t_hat_correct = predict_noise_level(diffusion, x0, int(t_true), re, device)
                rows.append(["correct", re, re, int(t_true), t_hat_correct, abs(t_hat_correct - t_true)])

                for wrong_re in args.re:
                    if wrong_re == re:
                        continue
                    t_hat_wrong = predict_noise_level(diffusion, x0, int(t_true), wrong_re, device)
                    rows.append(["wrong", re, wrong_re, int(t_true), t_hat_wrong, abs(t_hat_wrong - t_true)])

    csv_path = os.path.join(args.out_dir, "noise_level_accuracy.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["condition", "true_re", "conditioning_re", "t_true", "t_hat", "abs_error"])
        writer.writerows(rows)
    print(f"Raw results saved: {csv_path}", flush=True)

    errors_correct = [r[5] for r in rows if r[0] == "correct"]
    errors_wrong = [r[5] for r in rows if r[0] == "wrong"]
    mae_correct = float(np.mean(errors_correct))
    mae_wrong = float(np.mean(errors_wrong))

    print(f"\nMean |t_hat - t_true|  --  correct Re: {mae_correct:.1f}   wrong Re: {mae_wrong:.1f}"
          f"   (out of {cfg['timesteps']} timesteps)", flush=True)
    if mae_correct < mae_wrong:
        print("-> Re-conditioning IS being used: noise-level estimates are more accurate "
              "with the correct Re than with a wrong one.", flush=True)
    else:
        print("-> WARNING: noise-level estimates are NOT more accurate with the correct Re "
              "-- the model may be ignoring the Re channel.", flush=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot([errors_correct, errors_wrong], tick_labels=["correct Re", "wrong Re"])
    ax.set_ylabel("|t_hat - t_true|")
    ax.set_title("Noise-level classification error by Re-conditioning correctness")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(args.out_dir, "noise_level_accuracy.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {fig_path}", flush=True)


if __name__ == "__main__":
    main()
