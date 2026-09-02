"""
euler_multi_quadrants_openBC counterpart of correction_eval.py's rollout+
correction analysis (one emulator + one diffusion corrector, roll out
WITHOUT correction until divergence, then WITH correction over the same
horizon), mirroring correction_eval_ks.py in spirit -- but for a genuinely
divergence-prone system (compressible Euler CAN blow up: negative
density/energy -> NaN, unlike KS's bounded chaos) and with normalized
(mean/std) fields rather than KS's raw-scale ones.

Reuses rollout()/maybe_correct()/_is_diverged()/relative_l2_curve() from
correction_eval.py AS-IS (already dimension/channel-generic) -- only the
model loaders (padding-aware Euler classes) and the field-specific
diagnostics (shock-sharpness proxy, boundary-vs-interior error, snapshot
plots for 4 scalar/vector channels instead of a 2-component velocity field)
are new here.

Usage (on a compute node -- needs the raw .h5 data):
    python evaluation/correction_eval_euler.py \\
        --data_dir $DATA_DIR --exp_dir euler_multi_quadrants_openBC \\
        --val_test_file euler_multi_quadrants_openBC/data/valid/euler_multi_quadrants_openBC_gamma_1.4_Dry_air_20_chunk_40.hdf5 \\
        --traj_idx 5 \\
        --emulator_run $LOG_DIR/euler_multi_quadrants_openBC/fno_euler_gamma1.4 \\
        --diffusion_run $LOG_DIR/euler_multi_quadrants_openBC/diffusion_euler_gamma1.4 \\
        --rollout_steps 99 \\
        --out_dir correction_eval_euler_gamma1.4
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
from fno.fno_2D_classifier import FNO2D_classifier
from training.DiffusionModel import Diffusion
from training.ResidualCorrectorModel import ResidualCorrector
from training.euler_dataset import FIELD_ORDER, load_single_trajectory
from training.fno_training_euler import EmulatorFNO
from evaluation.correction_eval import (
    load_config, load_checkpoint_dict, rollout, maybe_correct,
    _is_diverged, _kinetic_energy_scalar, relative_l2_curve,
    compute_classical_energy_spectrum, kinetic_energy_curve,
)

DENSITY_IDX = FIELD_ORDER.index("density")
ENERGY_IDX = FIELD_ORDER.index("energy")
MOMENTUM_IDX = (FIELD_ORDER.index("momentum_x"), FIELD_ORDER.index("momentum_y"))


# ── loading ──────────────────────────────────────────────────────────────────

def load_emulator_fno_euler(run_dir, device):
    cfg = load_config(run_dir)
    model = EmulatorFNO(
        input_dim=cfg["n_channels"] * cfg["k"], output_dim=cfg["n_channels"],
        modes_x=cfg["modes_x"], modes_y=cfg["modes_y"], width=cfg["width"], l=cfg["kernel_size"],
        n_layer=cfg["n_layers"], hidden_proj=cfg.get("hidden_proj"), mlp=cfg.get("mlp", True),
        layers_mlp=cfg.get("layers_mlp"), tau=cfg.get("tau", 1e-5), padding=cfg.get("padding", 0),
        device=device,
    )
    ckpt = load_checkpoint_dict(run_dir, ["min_test_loss.pth", "final_model.pth", "min_train_loss.pth"], device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).float().eval(), cfg


def load_diffusion_euler(run_dir, device):
    cfg = load_config(run_dir)
    base = FNO2D_classifier(
        input_dim=cfg["n_channels"], output_dim=cfg["n_channels"],
        modes_x=cfg["modes_x"], modes_y=cfg["modes_y"], width=cfg["width"], l=cfg["kernel_size"],
        n_layer=cfg["n_layers"], hidden_proj=cfg.get("hidden_proj"), mlp=cfg.get("mlp", True),
        layers_mlp=cfg.get("layers_mlp"), class_mlp_layers=cfg.get("class_mlp_layers"),
        n_cat=cfg["timesteps"], padding=cfg.get("padding", 0), device=device,
    ).to(device).float()
    diffusion = Diffusion(model=base, timesteps=cfg["timesteps"], noise_sampling_coeff=cfg.get("noise_sampling_coeff"))
    ckpt = load_checkpoint_dict(run_dir, ["final_model.pth", "min_test_loss.pth", "min_train_loss.pth"], device)
    diffusion.load_state_dict(ckpt["model_state_dict"])
    return diffusion.to(device).eval(), cfg


def load_residual_corrector_euler(run_dir, device):
    """Loads a training/ResidualCorrectorModel.py::ResidualCorrector run
    (cf. configs/config_command_residual_corrector_euler.yaml) -- mirrors
    load_diffusion_euler above, but n_cat is max_step+1 (rollout-step
    buckets) rather than a diffusion timestep count, and there's no
    noise_sampling_coeff/timesteps schedule to restore."""
    cfg = load_config(run_dir)
    base = FNO2D_classifier(
        input_dim=cfg["n_channels"], output_dim=cfg["n_channels"],
        modes_x=cfg["modes_x"], modes_y=cfg["modes_y"], width=cfg["width"], l=cfg["kernel_size"],
        n_layer=cfg["n_layers"], hidden_proj=cfg.get("hidden_proj"), mlp=cfg.get("mlp", True),
        layers_mlp=cfg.get("layers_mlp"), class_mlp_layers=cfg.get("class_mlp_layers"),
        n_cat=cfg["max_step"] + 1, padding=cfg.get("padding", 0), device=device,
    ).to(device).float()
    corrector = ResidualCorrector(model=base, max_step=cfg["max_step"])
    ckpt = load_checkpoint_dict(run_dir, ["min_test_loss.pth", "final_model.pth", "min_train_loss.pth"], device)
    corrector.load_state_dict(ckpt["model_state_dict"])
    return corrector.to(device).eval(), cfg


def is_residual_corrector_run(run_dir):
    """config_command_residual_corrector_euler.yaml has max_step, no
    timesteps; config_command_diff_euler.yaml has the reverse -- lets
    correction_eval_euler.py's main() pick the right loader automatically
    without a new CLI flag."""
    cfg = load_config(run_dir)
    return "max_step" in cfg and "timesteps" not in cfg


def load_gt_trajectory_euler(data_dir, val_test_file, traj_idx, mean, std, frame_skip=1):
    """Returns the NORMALIZED trajectory (T, H, W, 4) -- same space the
    emulator/diffusion corrector operate in (cf. training/euler_dataset.py),
    so it can be fed to x0/compared against rollout() output directly.

    frame_skip must match the emulator's own training-time frame_skip (cf.
    training/euler_dataset.py::LazyEulerDataset) -- an emulator trained to
    predict frame_skip native frames ahead per call produces a rollout at
    that coarser cadence, so the GT it's compared against must be
    subsampled the same way, or every comparison is off by a growing
    temporal offset."""
    path = os.path.join(data_dir, val_test_file)
    full, gamma = load_single_trajectory(path, traj_idx)
    full = full[::frame_skip]
    full = (full - mean) / std
    return torch.from_numpy(full).float(), gamma


# ── Euler-specific diagnostics ───────────────────────────────────────────────

def shock_sharpness_series(traj, std):
    """traj: (T, H, W, 4) NORMALIZED -- denormalize density before taking
    the gradient (a sharpness proxy computed in normalized units would just
    be std[density]-rescaled, not wrong, but this keeps the proxy in
    physical units so it's comparable across runs/configs). Returns
    (T,) max|grad(density)| per frame -- a coarse shock-strength indicator:
    a genuine discontinuity keeps this high, a smeared-out (over-diffused)
    prediction collapses it."""
    traj_np = traj.numpy() if torch.is_tensor(traj) else traj
    density = traj_np[..., DENSITY_IDX] * std[DENSITY_IDX]
    gx = np.gradient(density, axis=1)
    gy = np.gradient(density, axis=2)
    return np.sqrt(gx ** 2 + gy ** 2).max(axis=(1, 2))


def boundary_interior_error_curves(pred, gt, margin):
    """pred, gt: (T, H, W, 4) NORMALIZED. Splits relative_l2_curve() into a
    `margin`-pixel-wide boundary ring vs the interior -- the point this
    dataset is meant to stress (open/extrapolation BC): if the corrector
    only helps in the interior and not near the boundary, that's exactly
    the failure mode padding=0 (FNO's implicit periodicity) would produce."""
    H, W = gt.shape[1], gt.shape[2]
    mask_interior = torch.zeros(H, W, dtype=torch.bool)
    mask_interior[margin:H - margin, margin:W - margin] = True
    mask_boundary = ~mask_interior

    def masked_curve(mask):
        p = pred[:, mask]  # (T, n_pixels, C)
        g = gt[:, mask]
        diff = (p - g).reshape(p.shape[0], -1)
        ref = g.reshape(g.shape[0], -1)
        return (diff.norm(dim=1) / (ref.norm(dim=1) + 1e-8)).numpy()

    return masked_curve(mask_interior), masked_curve(mask_boundary)


# ── plots ────────────────────────────────────────────────────────────────────

def draw_density_snapshots(name, gt_traj, traj_no, traj_with, std, mean, out_path, times=(0, 1, 2, 3, 4)):
    n_steps = min(traj_no.shape[0], gt_traj.shape[0]) - 1
    snap_times = [int(f * n_steps) for f in np.linspace(0, 1, len(times))]

    def denorm_density(traj):
        return (traj[..., DENSITY_IDX].numpy() * std[DENSITY_IDX]) + mean[DENSITY_IDX]

    dens_gt, dens_no, dens_with = denorm_density(gt_traj), denorm_density(traj_no), denorm_density(traj_with)
    vmax = float(dens_gt.max())
    vmin = float(dens_gt.min())

    n_rows = len(snap_times)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    col_titles = ["GT", "without correction", "with correction"]
    for row, t in enumerate(snap_times):
        for col, field in enumerate([dens_gt[t], dens_no[t], dens_with[t]]):
            ax = axes[row, col]
            ax.imshow(field, cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)
            if col == 0:
                ax.text(-0.15, 0.5, f"t={t}", transform=ax.transAxes, fontsize=9, va="center", ha="right")
    fig.suptitle(f"Density — {name}", fontsize=12, y=1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def draw_full_energy_curve(name, ke_no_corr, ke_with_corr, gt_ke, gt_len, out_path):
    """Energy-proxy curve (sum-of-squares-across-channels, cf.
    correction_eval.kinetic_energy_curve) over the FULL rollout -- unlike
    the other diagnostics here, this one is NOT truncated to the length of
    the available GT trajectory: it's the only plot that can show a genuine
    divergence happening past the ~100-native-frame GT window (rollout()
    itself only needs GT for x0, cf. its own docstring). gt_len marks where
    the GT trajectory ran out, for reference -- GT's own energy is plotted
    only up to that point, and a vertical line marks it on the full range."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(len(gt_ke)), gt_ke, color="black", lw=2, label="GT")
    ax.plot(np.arange(len(ke_no_corr)), ke_no_corr, color="tab:red", ls="--", label="without correction")
    ax.plot(np.arange(len(ke_with_corr)), ke_with_corr, color="tab:blue", label="with correction")
    ax.axvline(gt_len, color="gray", ls=":", lw=1, label="GT trajectory ends")
    ax.set_xlabel("rollout step")
    ax.set_ylabel(r"energy proxy $\frac{1}{2}\langle\sum_c x_c^2\rangle$")
    ax.set_yscale("log")
    ax.set_title(f"Energy over the full rollout — {name}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def draw_error_curve(name, error_no_corr, error_with_corr, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    t = np.arange(len(error_no_corr))
    ax.plot(t, error_no_corr, color="tab:red", ls="--", label="without correction")
    ax.plot(t, error_with_corr, color="tab:blue", ls="-", label="with correction")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("relative L2 error")
    ax.set_yscale("log")
    ax.set_title(f"Trajectory error over time — {name}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def draw_boundary_interior_error(name, interior_no, boundary_no, interior_with, boundary_with, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    t = np.arange(len(interior_no))
    ax.plot(t, interior_no, color="tab:red", ls="--", lw=1, label="interior, no correction")
    ax.plot(t, boundary_no, color="tab:orange", ls="--", lw=1, label="boundary, no correction")
    ax.plot(t, interior_with, color="tab:blue", lw=1.5, label="interior, with correction")
    ax.plot(t, boundary_with, color="tab:cyan", lw=1.5, label="boundary, with correction")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("relative L2 error")
    ax.set_yscale("log")
    ax.set_title(f"Boundary vs interior error — {name}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def draw_shock_sharpness(name, gt_sharp, no_sharp, with_sharp, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(gt_sharp, color="black", lw=2, label="GT")
    ax.plot(no_sharp, color="tab:red", ls="--", label="without correction")
    ax.plot(with_sharp, color="tab:blue", label="with correction")
    ax.set_xlabel("rollout step")
    ax.set_ylabel(r"$\max |\nabla \rho|$")
    ax.set_title(f"Shock-sharpness proxy — {name}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def draw_spectrum(name, gt_traj, traj_no_corr, traj_with_corr, out_path):
    """Spectrum of the momentum field (the two vector components,
    momentum_x/momentum_y) -- directly analogous to
    compute_classical_energy_spectrum's velocity-field spectrum in
    correction_eval.py."""
    def momentum(traj):
        t = traj.numpy() if torch.is_tensor(traj) else traj
        return t[..., MOMENTUM_IDX]

    k_gt, e_gt = compute_classical_energy_spectrum(momentum(gt_traj))
    k_no, e_no = compute_classical_energy_spectrum(momentum(traj_no_corr))
    k_with, e_with = compute_classical_energy_spectrum(momentum(traj_with_corr))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(k_gt[1:], e_gt[1:], color="black", lw=2, label="GT")
    ax.loglog(k_no[1:], e_no[1:], color="tab:red", ls="--", label="without correction")
    ax.loglog(k_with[1:], e_with[1:], color="tab:blue", label="with correction")
    ax.set_xlabel("wavenumber k")
    ax.set_ylabel("E(k)")
    ax.set_title(f"Momentum-field energy spectrum — {name}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--val_test_file", required=True,
                        help="path (relative to --data_dir) to the HDF5 file holding the evaluated trajectory")
    parser.add_argument("--traj_idx", type=int, default=5,
                        help="trajectory index inside --val_test_file (default 5: first of the test split, cf. config_command_emul_euler.yaml's test_traj_idx)")
    parser.add_argument("--emulator_run", required=True)
    parser.add_argument("--diffusion_run", required=True)
    parser.add_argument("--rollout_steps", type=int, default=99,
                        help="each GT trajectory only has 100 frames (99 steps), but this is NOT a hard "
                        "cap -- a value above 99 runs a genuine long-horizon free rollout past the "
                        "available GT (only needed for x0), truncating GT-relative diagnostics to what "
                        "GT covers while the full-length energy curve still shows the whole rollout")
    parser.add_argument("--divergence_factor", type=float, default=100.0)
    parser.add_argument("--s_init", type=int, default=7)
    parser.add_argument("--s_stop", type=int, default=3)
    parser.add_argument("--boundary_margin", type=int, default=16,
                        help="width (in pixels) of the boundary ring for the boundary-vs-interior error breakdown")
    parser.add_argument("--out_dir", default="correction_eval_euler")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                          else "cpu")

    print("Loading emulator + corrector...", flush=True)
    emulator, emul_cfg = load_emulator_fno_euler(args.emulator_run, device)
    if is_residual_corrector_run(args.diffusion_run):
        print("  corrector type: residual (single-shot, trained on real rollout drift)", flush=True)
        diffusion, diff_cfg = load_residual_corrector_euler(args.diffusion_run, device)
    else:
        print("  corrector type: diffusion (iterative DDPM denoising, synthetic noise)", flush=True)
        diffusion, diff_cfg = load_diffusion_euler(args.diffusion_run, device)
    prediction_mode = emul_cfg.get("prediction_mode", "state")
    frame_skip = emul_cfg.get("frame_skip", 1)
    mean = np.array(emul_cfg["field_mean"], dtype=np.float32)
    std = np.array(emul_cfg["field_std"], dtype=np.float32)
    print(f"  field order: {FIELD_ORDER}  frame_skip: {frame_skip}", flush=True)
    print(f"  mean={mean}  std={std}", flush=True)

    print("Loading GT trajectory...", flush=True)
    gt, gamma = load_gt_trajectory_euler(args.data_dir, args.val_test_file, args.traj_idx, mean, std, frame_skip=frame_skip)
    print(f"  gamma={gamma}  shape={tuple(gt.shape)}", flush=True)
    # Unlike an earlier version of this script, n_steps is NOT capped to the
    # GT trajectory's own length: rollout() only needs GT for x0 (cf. its
    # own docstring) and _is_diverged()'s energy-based check needs no GT at
    # all, so a genuine long-horizon divergence test (well past this
    # dataset's ~100-native-frame trajectories) is possible -- mirrors
    # correction_eval.py's own rollout, which likewise runs the full
    # requested length and only truncates the GT-relative diagnostics
    # (error curve, boundary/interior error, spectrum, snapshots) to
    # whatever GT is actually available.
    n_steps = args.rollout_steps
    if n_steps > gt.shape[0] - 1:
        print(f"  Note: rollout requested ({n_steps}) > available GT length ({gt.shape[0] - 1} steps). "
              f"The rollout will run the full {n_steps} steps; GT-relative diagnostics (error curve, "
              f"boundary/interior error, spectrum, density snapshots) are truncated to what GT covers -- "
              f"the full-length energy curve is the one diagnostic that still covers the whole rollout, "
              f"since it needs no GT beyond x0.", flush=True)
    x0 = gt[0].unsqueeze(0).to(device)

    print(f"Rolling out WITHOUT correction (up to {n_steps} steps, "
          f"divergence_factor={args.divergence_factor})...", flush=True)
    traj_no, tracker_no = rollout(emulator, prediction_mode, x0, n_steps, emulator.tau, device,
                                  diffusion=None, divergence_factor=args.divergence_factor)
    div_step = tracker_no["stopped_early_at"] or traj_no.shape[0] - 1
    print(f"  diverged at step {div_step}" if tracker_no["stopped_early_at"] else "  did not diverge", flush=True)

    print(f"Rolling out WITH correction ({div_step} steps, same horizon)...", flush=True)
    traj_with, tracker_with = rollout(emulator, prediction_mode, x0, div_step, emulator.tau, device,
                                      diffusion=diffusion, s_init=args.s_init, s_stop=args.s_stop,
                                      divergence_factor=args.divergence_factor)
    print(f"  n_corrections={tracker_with['n_corrections']}", flush=True)

    common_len = min(traj_no.shape[0], traj_with.shape[0], gt.shape[0])
    gt_c, traj_no_c, traj_with_c = gt[:common_len], traj_no[:common_len], traj_with[:common_len]

    print("Computing diagnostics + figures...", flush=True)

    # Full-length energy curve FIRST -- the one diagnostic not truncated to
    # GT length, so a divergence past the available GT trajectory is still
    # visible (cf. the n_steps note above).
    ke_no_full = kinetic_energy_curve(traj_no)
    ke_with_full = kinetic_energy_curve(traj_with)
    gt_ke_full = kinetic_energy_curve(gt)
    draw_full_energy_curve("euler_gamma%.1f" % gamma, ke_no_full, ke_with_full, gt_ke_full, gt.shape[0] - 1,
                           os.path.join(args.out_dir, "energy_curve_full.png"))

    error_no = relative_l2_curve(traj_no_c, gt_c)
    error_with = relative_l2_curve(traj_with_c, gt_c)
    draw_error_curve("euler_gamma%.1f" % gamma, error_no, error_with,
                     os.path.join(args.out_dir, "error_curve.png"))

    interior_no, boundary_no = boundary_interior_error_curves(traj_no_c, gt_c, args.boundary_margin)
    interior_with, boundary_with = boundary_interior_error_curves(traj_with_c, gt_c, args.boundary_margin)
    draw_boundary_interior_error("euler_gamma%.1f" % gamma, interior_no, boundary_no, interior_with, boundary_with,
                                 os.path.join(args.out_dir, "boundary_interior_error.png"))

    gt_sharp = shock_sharpness_series(gt_c, std)
    no_sharp = shock_sharpness_series(traj_no_c, std)
    with_sharp = shock_sharpness_series(traj_with_c, std)
    draw_shock_sharpness("euler_gamma%.1f" % gamma, gt_sharp, no_sharp, with_sharp,
                         os.path.join(args.out_dir, "shock_sharpness.png"))

    draw_density_snapshots("euler_gamma%.1f" % gamma, gt_c, traj_no_c, traj_with_c, std, mean,
                           os.path.join(args.out_dir, "density_snapshots.png"))

    draw_spectrum("euler_gamma%.1f" % gamma, gt_c, traj_no_c, traj_with_c,
                 os.path.join(args.out_dir, "spectrum.png"))

    np.savez(os.path.join(args.out_dir, "rollout_data.npz"),
            # Spatial fields kept at common_len (GT-truncated) -- saving the
            # full rollout's fields too would bloat this file badly for a
            # long-horizon divergence run (e.g. 2000 steps x 512x512x4
            # float32 ~ 8GB); the full-length signal that matters for
            # divergence (energy, not spatial detail) is saved separately
            # below, at negligible size.
            gt=gt_c.numpy(), traj_no=traj_no_c.numpy(), traj_with=traj_with_c.numpy(),
            mean=mean, std=std, div_step=div_step, n_steps_requested=n_steps,
            error_no=error_no, error_with=error_with,
            ke_no_full=ke_no_full, ke_with_full=ke_with_full, gt_ke_full=gt_ke_full,
            correction_flags=tracker_with["correction_flags"], noise_levels=tracker_with["noise_levels"])

    print(f"Done. Figures + data saved to {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
