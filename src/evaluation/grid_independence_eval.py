"""
Grid-independence comparison: FNO vs UNet on Kolmogorov Re90, evaluated
zero-shot at several spatial resolutions -- only one of which (ds=2, 64x64)
matches what either model was actually trained on. Short-horizon
(seq_length-step) pointwise accuracy only -- NOT a long autoregressive
rollout (that's correction_eval.py's job; this script isolates the
resolution effect from rollout-divergence effects entirely).

Why this is the right test: fno/fno_2D.py::FNO2D is architecturally
resolution-agnostic at inference -- get_meshgrid() recomputes the grid from
the input tensor's own shape every call, P/Q act pointwise on channels only,
and IntegralKernel2D's learned weights have a fixed shape (in,out,2*modes_x,
modes_y) independent of H/W, so only modes_x <= new_resolution//2 is
required. unet/unet_2D.py::UNet2D has no resolution-locked parameter either,
but Up.forward's skip-connection torch.cat requires H,W divisible by
2**depth at every stage, and even where shapes are compatible there is no
design guarantee of discretization-invariant output the way there is for
FNO's mode truncation. This script measures that gap empirically rather
than arguing it from architecture alone.

Reuses evaluation/correction_eval.py's load_emulator_fno/load_config/
load_checkpoint_dict/load_gt_trajectory as-is (load_gt_trajectory already
takes an arbitrary ds), and training/dataset_manager.py::SequenceDataset
for the windowing + delta/state target convention -- SequenceDataset takes
a raw (T,H,W,C) tensor directly, so it's already resolution-agnostic; this
is the SAME class Trainer.train_epoch() is fed during real training, so
replaying its exact free-running delta-mode loop here (copied from
training/trainer.py, not reimplemented differently) makes a ds=2/64x64
resolution-sweep number a genuine sanity check against that run's own
already-logged final-epoch Te relative_rmse. Only a UNet-specific
checkpoint loader (load_emulator_unet, mirroring load_emulator_fno) is new.

Usage (from thermalizer/src):
    python evaluation/grid_independence_eval.py \\
        --data_dir ../data --exp_dir kolmogorov/Re90 \\
        --fno_run ../runs_mesu/Re90/Re90/16modes_emul_RE90 \\
        --unet_run ../runs_mesu/Re90/Re90/unet_delta_RE90 \\
        --resolutions 32 64 128 \\
        --out_dir grid_independence_results
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.dataset_manager import SequenceDataset
from training.factory import relative_rmse, relative_mae
from training.unet_training import EmulatorUNet
from evaluation.correction_eval import (
    load_config, load_checkpoint_dict, load_emulator_fno, load_gt_trajectory,
)

# Raw Kolmogorov Re90 data is 128x128 (data/kolmogorov/Re90/train_traj/sim*.h5,
# velocity_field shape (4000,128,128,2)) -- ds is a plain [::ds,::ds] subsample
# (training/dataset_manager.py), so these are the only three resolutions
# reachable from the existing raw files without generating new data.
RESOLUTION_TO_DS = {32: 4, 64: 2, 128: 1}


def load_emulator_unet(run_dir, device):
    """Mirrors correction_eval.py::load_emulator_fno, for the UNet family
    (training/unet_training.py::EmulatorUNet) -- not in correction_eval.py
    itself since that module has no UNet support at all yet."""
    cfg = load_config(run_dir)
    model = EmulatorUNet(
        input_dim=cfg["input_dim"], output_dim=cfg["output_dim"],
        depth=cfg.get("depth", 3), base_width=cfg.get("base_width", 32),
        tau=cfg.get("tau", 1e-5), device=device,
    )
    ckpt = load_checkpoint_dict(run_dir, ["min_test_loss.pth", "final_model.pth", "min_train_loss.pth"], device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).float().eval(), cfg


@torch.no_grad()
def short_horizon_relative_errors(model, prediction_mode, tau, traj, seq_length, stride, device,
                                  divergence_threshold=1.0):
    """traj: (T, H, W, C) at whatever resolution was already loaded.
    Replays EXACTLY training/trainer.py::Trainer.train_epoch()'s own
    free-running loop (delta mode: x_t advances by the model's own
    predicted delta each step, same tau*randn_like perturbation; state
    mode: x_t = model(x_t) directly) over every window SequenceDataset
    would produce -- so this is the same computation the already-logged
    Te relative_rmse/relative_mae in logs/metrics.csv comes from, just
    replayed here at whatever ds the caller loaded traj with.

    Returns a dict with both MEAN and MEDIAN relative_rmse/relative_mae,
    plus the raw per-window arrays and a divergence count/rate. The mean
    alone is not safe to report here: a handful of windows where the
    free-running delta loop genuinely diverges within just `seq_length`
    steps (a real possibility for a model with unstable rollout dynamics,
    not a bug) can produce enormous per-window errors that dominate a
    plain average even when most windows are fine -- first found on
    unet_delta_RE90, whose own official (much larger, averaged over many
    more/differently-sampled windows) logs/metrics.csv Te relative_rmse was
    ~0.06, nowhere near the ~36-60 a naive mean gave here. The median is
    the primary number this script reports; divergence_rate (fraction of
    windows with relative_rmse > divergence_threshold, i.e. worse than a
    trivial all-zero prediction) makes that instability visible explicitly
    rather than silently skewing an average.
    """
    ds = SequenceDataset(traj, seq_length=seq_length, stride=stride, prediction_mode=prediction_mode)
    if len(ds) == 0:
        raise ValueError(f"Trajectory too short ({traj.shape[0]} frames) for seq_length={seq_length}, stride={stride}")

    rmses, maes = [], []
    for i in range(len(ds)):
        x_t, target = ds[i]
        x_t = x_t.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)

        outputs = torch.empty_like(target)
        for t in range(target.shape[1]):
            if prediction_mode == "state":
                x_t = model(x_t)
                outputs[:, t, ...] = x_t
            else:
                x_dt = model(x_t)
                x_t = x_t + x_dt + tau * torch.randn_like(x_dt)
                outputs[:, t, ...] = x_dt

        rmses.append(relative_rmse(outputs, target).item())
        maes.append(relative_mae(outputs, target).item())

    rmses, maes = np.array(rmses), np.array(maes)
    return {
        "mean_rmse": float(np.mean(rmses)), "median_rmse": float(np.median(rmses)),
        "mean_mae": float(np.mean(maes)), "median_mae": float(np.median(maes)),
        "max_rmse": float(np.max(rmses)), "n_windows": len(rmses),
        "n_diverged": int(np.sum(rmses > divergence_threshold)),
        "rmses": rmses, "maes": maes,
    }


def draw_grid_independence(resolutions, fno_rmse, unet_rmse, fno_mae, unet_mae, trained_res, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, fno_y, unet_y, title in [
        (axes[0], fno_rmse, unet_rmse, "median relative RMSE"),
        (axes[1], fno_mae, unet_mae, "median relative MAE"),
    ]:
        ax.plot(resolutions, fno_y, "o-", color="tab:blue", label="FNO", lw=2, ms=7)
        ax.plot(resolutions, unet_y, "s-", color="tab:red", label="UNet", lw=2, ms=7)
        ax.axvline(trained_res, color="gray", ls=":", lw=1, label=f"training resolution ({trained_res})")
        ax.set_xlabel("grid resolution")
        ax.set_ylabel(title)
        ax.set_xscale("log", base=2)
        ax.set_xticks(resolutions)
        ax.set_xticklabels([str(r) for r in resolutions])
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("Zero-shot grid independence -- Kolmogorov Re90", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default=os.environ.get("DATA_DIR", "../data"))
    parser.add_argument("--exp_dir", default="kolmogorov/Re90")
    parser.add_argument("--sim_file", default="sim8.h5")
    parser.add_argument("--fno_run", required=True)
    parser.add_argument("--unet_run", required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--held_out_frac", type=float, default=0.2,
                        help="fraction of the trajectory (from the end) used for this eval -- "
                        "an approximation of a held-out window, not a rigorous re-derivation of "
                        "DatasetManagerMulti's own time-based train/test split")
    parser.add_argument("--stride", type=int, default=20, help="stride between windows, in native (post-ds) frames")
    parser.add_argument("--out_dir", default="grid_independence_results")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available()
                              else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                              else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    for r in args.resolutions:
        if r not in RESOLUTION_TO_DS:
            raise SystemExit(f"Unsupported resolution {r} -- only {sorted(RESOLUTION_TO_DS)} are reachable "
                              f"from the raw 128x128 data via integer ds (cf. RESOLUTION_TO_DS).")
    resolutions = sorted(args.resolutions)

    print("Loading FNO...", flush=True)
    fno, fno_cfg = load_emulator_fno(args.fno_run, device)
    fno_mode = fno_cfg.get("prediction_mode", "delta")
    fno_seq_length = fno_cfg["seq_length"]
    fno_trained_res = 128 // fno_cfg["ds"]
    print(f"  prediction_mode={fno_mode}  seq_length={fno_seq_length}  trained at {fno_trained_res}x{fno_trained_res} (ds={fno_cfg['ds']})", flush=True)

    print("Loading UNet...", flush=True)
    unet, unet_cfg = load_emulator_unet(args.unet_run, device)
    unet_mode = unet_cfg.get("prediction_mode", "delta")
    unet_seq_length = unet_cfg["seq_length"]
    unet_trained_res = 128 // unet_cfg["ds"]
    print(f"  prediction_mode={unet_mode}  seq_length={unet_seq_length}  trained at {unet_trained_res}x{unet_trained_res} (ds={unet_cfg['ds']})", flush=True)

    if fno_trained_res != unet_trained_res:
        print(f"  Warning: FNO and UNet were trained at different resolutions "
              f"({fno_trained_res} vs {unet_trained_res}) -- the 'training resolution' marker "
              f"on the plot will only be drawn at the FNO's.", flush=True)

    results = {"fno": {"rmse": [], "mae": []}, "unet": {"rmse": [], "mae": []}}
    all_stats = []  # kept for the CSV's diagnostic columns (n_windows, n_diverged, mean vs median)

    for res in resolutions:
        ds = RESOLUTION_TO_DS[res]
        print(f"\n=== resolution {res}x{res} (ds={ds}) ===", flush=True)
        full_traj = load_gt_trajectory(args.data_dir, args.exp_dir, args.sim_file, ds=ds)
        n_held_out = max(fno_seq_length + unet_seq_length + 2, int(full_traj.shape[0] * args.held_out_frac))
        traj = full_traj[-n_held_out:]
        print(f"  trajectory shape at this resolution: {tuple(full_traj.shape)}, "
              f"using last {tuple(traj.shape)} as the held-out window", flush=True)

        fno_stats = short_horizon_relative_errors(
            fno, fno_mode, fno.tau, traj, fno_seq_length, args.stride, device)
        print(f"  FNO:  median relative_rmse={fno_stats['median_rmse']:.4f}  "
              f"(mean={fno_stats['mean_rmse']:.4f}, max={fno_stats['max_rmse']:.4f})  "
              f"median relative_mae={fno_stats['median_mae']:.4f}  "
              f"diverged {fno_stats['n_diverged']}/{fno_stats['n_windows']} windows", flush=True)
        results["fno"]["rmse"].append(fno_stats["median_rmse"])
        results["fno"]["mae"].append(fno_stats["median_mae"])
        all_stats.append(("fno", res, fno_stats))

        try:
            unet_stats = short_horizon_relative_errors(
                unet, unet_mode, unet.tau, traj, unet_seq_length, args.stride, device)
            print(f"  UNet: median relative_rmse={unet_stats['median_rmse']:.4f}  "
                  f"(mean={unet_stats['mean_rmse']:.4f}, max={unet_stats['max_rmse']:.4f})  "
                  f"median relative_mae={unet_stats['median_mae']:.4f}  "
                  f"diverged {unet_stats['n_diverged']}/{unet_stats['n_windows']} windows", flush=True)
        except RuntimeError as e:
            # e.g. a resolution not divisible by 2**depth would crash inside
            # UNet2D's Up.forward skip-connection concat -- recorded as NaN
            # rather than aborting the whole sweep, so the FNO side of the
            # comparison still completes.
            print(f"  UNet: FAILED at this resolution ({e}) -- recording NaN", flush=True)
            unet_stats = {"median_rmse": float("nan"), "median_mae": float("nan"),
                         "mean_rmse": float("nan"), "mean_mae": float("nan"),
                         "max_rmse": float("nan"), "n_windows": 0, "n_diverged": 0}
        results["unet"]["rmse"].append(unet_stats["median_rmse"])
        results["unet"]["mae"].append(unet_stats["median_mae"])
        all_stats.append(("unet", res, unet_stats))

    print("\nWriting outputs...", flush=True)
    csv_path = os.path.join(args.out_dir, "grid_independence.csv")
    with open(csv_path, "w") as f:
        f.write("resolution,model,median_relative_rmse,median_relative_mae,mean_relative_rmse,"
                "mean_relative_mae,max_relative_rmse,n_windows,n_diverged\n")
        for model_name, res, s in all_stats:
            f.write(f"{res},{model_name},{s['median_rmse']:.6f},{s['median_mae']:.6f},"
                    f"{s['mean_rmse']:.6f},{s['mean_mae']:.6f},{s['max_rmse']:.6f},"
                    f"{s['n_windows']},{s['n_diverged']}\n")
    print(f"  {csv_path}", flush=True)

    plot_path = os.path.join(args.out_dir, "grid_independence.png")
    draw_grid_independence(resolutions, results["fno"]["rmse"], results["unet"]["rmse"],
                           results["fno"]["mae"], results["unet"]["mae"], fno_trained_res, plot_path)
    print(f"  {plot_path}", flush=True)


if __name__ == "__main__":
    main()
