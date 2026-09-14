"""
Rollout-based generalization check for the shear_flow FNO emulator.

Unlike the Tr/Te split reported during training (DatasetManagerMulti's
random_split of time-windows drawn from the SAME train_traj/*.h5 files --
those windows are held-out in time, not in initial condition), this
evaluates the trained emulator on test_traj/*.h5 -- genuinely unseen initial
conditions, never touched during training. Each shear_flow trajectory is a
distinct (n_shear, n_blobs, w) draw among 40 total per (Reynolds, Schmidt)
(32 train / 4 valid / 4 test, cf. the dataset's own HuggingFace README), not
a noisy replicate of the same setup -- so this is a much stronger test of
whether the emulator learned the underlying one-step operator rather than
memorizing the training trajectories' own geometries.

No diffusion corrector involved (none exists yet for shear_flow) -- pure
one-step-emulator rollout vs ground truth, for each of the held-out test
trajectories.

Usage (from thermalizer/src):
    python evaluation/correction_eval_shear_flow.py \
        --data_dir $DATA_DIR --runs_dir $LOG_DIR \
        --exp_dir shear_flow/Re5e4_Sc1e0 --run_name exp_shear_flow_Re5e4_Sc1e0 \
        --rollout 190 --out_dir correction_eval_shear_flow_results
"""

import argparse
import os
import sys

import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# fno/, training/ and evaluation/ are siblings under src/ -- add src/ itself
# so this works whether run as `python evaluation/correction_eval_shear_flow.py`
# from src/ or from elsewhere (cf. correction_eval.py's own sys.path setup).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.correction_eval import (
    load_emulator_fno, rollout, relative_l2_curve, kinetic_energy_curve,
    vorticity,
)

# tracer, pressure, u, v (cf. prepare_shear_flow_dataset.py's field_order)
VELOCITY_CHANNELS = (2, 3)


def load_test_trajectory(data_dir, exp_dir, sim_file, ds, ratio=1):
    path = os.path.join(data_dir, exp_dir, "test_traj", sim_file)
    with h5py.File(path, "r") as f:
        data = f["velocity_field"][()][::ratio, ::ds, ::ds]
    return torch.from_numpy(data).float()


def plot_error_curve(sim_name, error, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(error, color="tab:blue")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Relative L2 error")
    ax.set_yscale("log")
    ax.set_title(f"Rollout error vs ground truth -- {sim_name}")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_kinetic_energy(sim_name, ke_pred, ke_gt, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ke_gt, color="black", lw=2, label="GT")
    ax.plot(ke_pred, color="tab:blue", label="emulator")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Mean kinetic energy")
    ax.set_yscale("log")
    ax.set_title(f"Kinetic energy -- {sim_name}")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_vorticity_panel(sim_name, gt_traj, pred_traj, out_path, times=(0, 1, 2, 3, 4)):
    n_steps = min(pred_traj.shape[0], gt_traj.shape[0]) - 1
    snap = [int(f * n_steps) for f in np.linspace(0, 1, len(times))]
    vort_gt = vorticity(gt_traj.numpy(), velocity_channels=VELOCITY_CHANNELS)
    vort_pred = vorticity(pred_traj.numpy(), velocity_channels=VELOCITY_CHANNELS)
    vmax = float(np.abs(vort_gt[snap]).max()) + 1e-12

    fig, axes = plt.subplots(len(snap), 2, figsize=(6, 3 * len(snap)))
    for row, t in enumerate(snap):
        axes[row, 0].imshow(vort_gt[t], cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax), origin="lower")
        axes[row, 1].imshow(vort_pred[t], cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax), origin="lower")
        axes[row, 0].set_ylabel(f"step {t}")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_title("GT")
    axes[0, 1].set_title("emulator")
    fig.suptitle(f"Vorticity -- {sim_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default=os.environ.get("DATA_DIR", "../data"))
    parser.add_argument("--runs_dir", default=os.environ.get("LOG_DIR", "../runs"))
    parser.add_argument("--exp_dir", default="shear_flow/Re5e4_Sc1e0")
    parser.add_argument("--run_name", default="exp_shear_flow_Re5e4_Sc1e0")
    parser.add_argument("--rollout", type=int, default=190,
                        help="Number of steps -- 190 by default, close to each "
                        "test trajectory's full length (200 frames)")
    parser.add_argument("--out_dir", default="correction_eval_shear_flow_results")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)

    run_dir = os.path.join(args.runs_dir, args.exp_dir, args.run_name)
    print(f"Loading emulator from {run_dir}...", flush=True)
    model, cfg = load_emulator_fno(run_dir, device)
    prediction_mode = cfg.get("prediction_mode", "delta")
    tau = cfg.get("tau", 1e-5)
    ds = cfg.get("ds", 1)

    test_dir = os.path.join(args.data_dir, args.exp_dir, "test_traj")
    sim_files = sorted(f for f in os.listdir(test_dir) if f.endswith(".h5"))
    print(f"{len(sim_files)} held-out test trajectories (never seen during "
          f"training, each a different initial condition): {sim_files}", flush=True)

    summary_rows = []
    for sim_file in sim_files:
        sim_name = sim_file.replace(".h5", "")
        print(f"\n=== {sim_name} ===", flush=True)
        gt_full = load_test_trajectory(args.data_dir, args.exp_dir, sim_file, ds=ds)
        n_steps = min(args.rollout, gt_full.shape[0] - 1)
        x0 = gt_full[:1].to(device)

        pred_traj, _ = rollout(model, prediction_mode, x0, n_steps, tau, device)
        gt_traj = gt_full[: n_steps + 1]

        error = relative_l2_curve(pred_traj, gt_traj)
        ke_pred = kinetic_energy_curve(pred_traj)
        ke_gt = kinetic_energy_curve(gt_traj)

        plot_error_curve(sim_name, error, os.path.join(args.out_dir, f"error_{sim_name}.png"))
        plot_kinetic_energy(sim_name, ke_pred, ke_gt, os.path.join(args.out_dir, f"kinetic_energy_{sim_name}.png"))
        plot_vorticity_panel(sim_name, gt_traj, pred_traj, os.path.join(args.out_dir, f"vorticity_{sim_name}.png"))

        summary_rows.append((sim_name, float(error[-1]), float(error.mean())))
        print(f"  final relative L2 error: {error[-1]:.4f}  (mean over rollout: {error.mean():.4f})", flush=True)

    summary_path = os.path.join(args.out_dir, "summary.csv")
    with open(summary_path, "w") as f:
        f.write("sim_file,final_relative_l2,mean_relative_l2\n")
        for name, final_err, mean_err in summary_rows:
            f.write(f"{name},{final_err:.6f},{mean_err:.6f}\n")
    print(f"\nResults saved to {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
