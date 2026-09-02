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

import h5py
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


def compute_norm_stats(data_dir, exp_dir, ratio, ds, prediction_mode):
    """Replicates training/dataset_manager.py::DatasetManagerMulti's own
    per-channel x_mean/x_std/y_mean/y_std computation EXACTLY (same
    ratio/ds subsampling of every train_traj/*.h5 file, same
    prediction_mode-dependent y target: state -> next frame, else -> delta
    between consecutive frames) -- necessary because these models were
    trained with normalize=True (DatasetManagerMulti's own default, cf.
    dataset_manager.py:146), so evaluating them on raw un-normalized data is
    a scale mismatch. This script's own first version hit exactly that bug:
    UNet's relative_rmse came out >> 1 (garbage, fed input miles outside
    its trained distribution) while FNO merely degraded, because FNO's
    linear P/Q layers tolerate a scale mismatch far more gracefully than
    UNet's stack of GroupNorm-based conv blocks.

    Computed ONCE per model, at that model's OWN training ratio/ds -- these
    are per-channel scalars describing a physical property of the flow
    (mean/std of velocity and of its increments), not of whichever
    resolution is being tested zero-shot, so the SAME stats are reused
    across every tested resolution rather than recomputed per resolution.
    """
    sim_dir = os.path.join(data_dir, exp_dir, "train_traj")
    sim_files = [os.path.join(sim_dir, f) for f in os.listdir(sim_dir) if f.endswith(".h5")]

    all_x, all_y = [], []
    for sim_file in sim_files:
        with h5py.File(sim_file, "r") as f:
            data = f["velocity_field"][()][::ratio, ::ds, ::ds]
        tensor_data = torch.from_numpy(data).float()
        all_x.append(tensor_data[:-1])
        if prediction_mode == "state":
            all_y.append(tensor_data[1:])
        else:
            all_y.append(tensor_data[1:] - tensor_data[:-1])

    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)
    return (all_x.mean(dim=(0, 1, 2)), all_x.std(dim=(0, 1, 2)),
            all_y.mean(dim=(0, 1, 2)), all_y.std(dim=(0, 1, 2)))


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
                                  normalize, x_mean, x_std, y_mean, y_std, divergence_threshold=1.0):
    """traj: (T, H, W, C) at whatever resolution was already loaded.
    Replays EXACTLY training/trainer.py::Trainer.train_epoch()'s own
    free-running loop (delta mode: x_t advances by the model's own
    predicted delta each step, same tau*randn_like perturbation; state
    mode: x_t = model(x_t) directly) over every window SequenceDataset
    would produce -- so this is the same computation the already-logged
    Te relative_rmse/relative_mae in logs/metrics.csv comes from, just
    replayed here at whatever ds the caller loaded traj with.

    normalize + x_mean/x_std/y_mean/y_std (from compute_norm_stats,
    matching this model's OWN training ratio/ds/prediction_mode): whether
    the checkpoint being evaluated was trained on normalized data
    (DatasetManagerMulti's normalize=True default) needs to be matched
    here, or the model sees input miles outside its trained distribution --
    a scale-mismatch bug this script's first version had (gave nonsensical
    relative_rmse >> 1 for UNet). BUT this can't be inferred from a config
    key alone: fno_training.py and unet_training.py call
    DatasetManagerMulti identically (no explicit normalize kwarg) in
    TODAY's code, yet 16modes_emul_RE90 (an old checkpoint) only reproduces
    its own logged Te relative_rmse with normalize=False, while
    unet_delta_RE90 (trained this week) only reproduces its own with
    normalize=True -- DatasetManagerMulti's actual default evidently
    changed at some point between the two trainings. There is no reliable
    way to recover which was used from the checkpoint/config alone; the
    caller must say so explicitly per model (cf. main()'s
    --fno_normalize/--unet_normalize), calibrated against that run's own
    logs/metrics.csv as the ground truth. When normalize=True, everything
    below (x_t, target, outputs, the reported errors) is computed in the
    SAME normalized space Trainer.train_epoch() itself used, directly
    comparable to logs/metrics.csv's own numbers; when False, raw scale
    throughout, same as the SAME comparison already validated for FNO.

    Returns a dict with both MEAN and MEDIAN relative_rmse/relative_mae,
    plus the raw per-window arrays and a divergence count/rate. The mean
    alone is not safe to report here: a handful of windows where the
    free-running delta loop genuinely diverges within just `seq_length`
    steps (a real possibility for a model with unstable rollout dynamics,
    not a bug) can produce enormous per-window errors that dominate a
    plain average even when most windows are fine. The median is the
    primary number this script reports; divergence_rate (fraction of
    windows with relative_rmse > divergence_threshold, i.e. worse than a
    trivial all-zero prediction) makes that instability visible explicitly
    rather than silently skewing an average.
    """
    if normalize:
        ds = SequenceDataset(traj, seq_length=seq_length, stride=stride, prediction_mode=prediction_mode,
                            normalize=True, x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    else:
        ds = SequenceDataset(traj, seq_length=seq_length, stride=stride, prediction_mode=prediction_mode,
                            normalize=False)
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
    # Defaults calibrated empirically against each run's OWN logs/metrics.csv
    # Te relative_rmse (16modes_emul_RE90 only matches with normalize=False;
    # unet_delta_RE90/unet_state_RE90 only match with normalize=True) --
    # cf. short_horizon_relative_errors' docstring for why this can't be
    # inferred automatically. Override if evaluating a different checkpoint.
    parser.add_argument("--fno_normalize", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--unet_normalize", action=argparse.BooleanOptionalAction, default=True)
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
    print(f"  prediction_mode={fno_mode}  seq_length={fno_seq_length}  trained at {fno_trained_res}x{fno_trained_res} (ds={fno_cfg['ds']})  normalize={args.fno_normalize}", flush=True)
    fno_x_mean = fno_x_std = fno_y_mean = fno_y_std = None
    if args.fno_normalize:
        print("  Computing normalization stats matching this run's own training (ratio/ds/prediction_mode)...", flush=True)
        fno_x_mean, fno_x_std, fno_y_mean, fno_y_std = compute_norm_stats(
            args.data_dir, args.exp_dir, fno_cfg.get("ratio", 1), fno_cfg["ds"], fno_mode)
        print(f"  x_mean={fno_x_mean.tolist()}  x_std={fno_x_std.tolist()}  "
              f"y_mean={fno_y_mean.tolist()}  y_std={fno_y_std.tolist()}", flush=True)

    print("Loading UNet...", flush=True)
    unet, unet_cfg = load_emulator_unet(args.unet_run, device)
    unet_mode = unet_cfg.get("prediction_mode", "delta")
    unet_seq_length = unet_cfg["seq_length"]
    unet_trained_res = 128 // unet_cfg["ds"]
    print(f"  prediction_mode={unet_mode}  seq_length={unet_seq_length}  trained at {unet_trained_res}x{unet_trained_res} (ds={unet_cfg['ds']})  normalize={args.unet_normalize}", flush=True)
    unet_x_mean = unet_x_std = unet_y_mean = unet_y_std = None
    if args.unet_normalize:
        print("  Computing normalization stats matching this run's own training (ratio/ds/prediction_mode)...", flush=True)
        unet_x_mean, unet_x_std, unet_y_mean, unet_y_std = compute_norm_stats(
            args.data_dir, args.exp_dir, unet_cfg.get("ratio", 1), unet_cfg["ds"], unet_mode)
        print(f"  x_mean={unet_x_mean.tolist()}  x_std={unet_x_std.tolist()}  "
              f"y_mean={unet_y_mean.tolist()}  y_std={unet_y_std.tolist()}", flush=True)

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
            fno, fno_mode, fno.tau, traj, fno_seq_length, args.stride, device,
            args.fno_normalize, fno_x_mean, fno_x_std, fno_y_mean, fno_y_std)
        print(f"  FNO:  median relative_rmse={fno_stats['median_rmse']:.4f}  "
              f"(mean={fno_stats['mean_rmse']:.4f}, max={fno_stats['max_rmse']:.4f})  "
              f"median relative_mae={fno_stats['median_mae']:.4f}  "
              f"diverged {fno_stats['n_diverged']}/{fno_stats['n_windows']} windows", flush=True)
        results["fno"]["rmse"].append(fno_stats["median_rmse"])
        results["fno"]["mae"].append(fno_stats["median_mae"])
        all_stats.append(("fno", res, fno_stats))

        try:
            unet_stats = short_horizon_relative_errors(
                unet, unet_mode, unet.tau, traj, unet_seq_length, args.stride, device,
                args.unet_normalize, unet_x_mean, unet_x_std, unet_y_mean, unet_y_std)
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
