"""
KS-equation (1D, single nu) counterpart of correction_eval.py's rollout+
correction analysis, mirroring Parameterized_Neural_Operator's
hyper_r_correction_multi_re.py in spirit (one emulator + one diffusion
corrector, roll out WITHOUT correction until divergence, then WITH
correction over the same horizon, full metric/figure suite) but for a
single nu instead of a Re sweep.

Reuses rollout()/maybe_correct()/_is_diverged()/_kinetic_energy_scalar()/
time_delay_embedding()/load_config()/load_checkpoint_dict() from
correction_eval.py AS-IS (already dimension-generic after this session's
fix to rollout()'s field-shape handling) -- only the model loaders (1D
classes) and the field-specific diagnostics (energy spectrum, "dissipation"
analogue, snapshot plots) are new here, since KS has no vorticity/velocity
concept.

Usage (on a compute node -- needs the raw .h5 data):
    python evaluation/correction_eval_ks.py \\
        --data_dir $DATA_DIR --exp_dir KS_equation/nu0p35 --sim_file sim1.h5 \\
        --emulator_run $LOG_DIR/KS_equation/nu0p35/fno_ks_nu0p35 \\
        --diffusion_run $LOG_DIR/KS_equation/nu0p35/diffusion_ks_nu0p35 \\
        --rollout_steps 5000 --nu 0.35 \\
        --out_dir correction_eval_ks_nu0p35
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
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.fno_training_1d import EmulatorFNO1D
from fno.fno_1D_classifier import FNO1D_classifier
from training.DiffusionModel import Diffusion
from evaluation.correction_eval import (
    load_config, load_checkpoint_dict, rollout, maybe_correct,
    _is_diverged, _kinetic_energy_scalar, time_delay_embedding,
)


# ── loading ──────────────────────────────────────────────────────────────────

def load_emulator_fno1d(run_dir, device):
    cfg = load_config(run_dir)
    model = EmulatorFNO1D(
        input_dim=cfg["input_dim"], output_dim=cfg["output_dim"],
        modes=cfg["k_max"], width=cfg["width"], l=cfg["l"],
        n_layer=cfg["n_fourier_layer"], hidden_proj=cfg.get("hidden_proj"), mlp=cfg.get("mlp", True),
        layers_mlp=cfg.get("layers_mlp"), tau=cfg.get("tau", 1e-5), device=device,
    )
    ckpt = load_checkpoint_dict(run_dir, ["min_test_loss.pth", "final_model.pth", "min_train_loss.pth"], device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).float().eval(), cfg


def load_diffusion_1d(run_dir, device):
    cfg = load_config(run_dir)
    base = FNO1D_classifier(
        input_dim=cfg["input_dim"], output_dim=cfg["output_dim"],
        modes=cfg["k_max"], width=cfg["width"], l=cfg["l"],
        n_layer=cfg["n_fourier_layer"], hidden_proj=cfg.get("hidden_proj"), mlp=cfg.get("mlp", True),
        layers_mlp=cfg.get("layers_mlp"), class_mlp_layers=cfg.get("class_mlp_layers"),
        n_cat=cfg["timesteps"], device=device,
    ).to(device).float()
    diffusion = Diffusion(model=base, timesteps=cfg["timesteps"], noise_sampling_coeff=cfg.get("noise_sampling_coeff"))
    ckpt = load_checkpoint_dict(run_dir, ["final_model.pth", "min_test_loss.pth", "min_train_loss.pth"], device)
    diffusion.load_state_dict(ckpt["model_state_dict"])
    return diffusion.to(device).eval(), cfg


def load_gt_trajectory_ks(data_dir, exp_dir, sim_file, ds, ratio=1):
    path = os.path.join(data_dir, exp_dir, "train_traj", sim_file)
    if not os.path.exists(path):
        path = os.path.join(data_dir, exp_dir, "test_traj", sim_file)
    with h5py.File(path, "r") as f:
        data = f["state"][()][::ratio, ::ds]
    return torch.from_numpy(data).float()


# ── 1D diagnostics ───────────────────────────────────────────────────────────

def field_energy_series(traj):
    """traj: (T, Nx, 1) -> E(t) = 0.5*<u^2>_x, the KS analogue of kinetic energy."""
    return 0.5 * (traj ** 2).mean(dim=(-1, -2)).numpy()


def gradient_energy_series(traj, L):
    """(T, Nx, 1) -> <(du/dx)^2>_x -- KS analogue of dissipation/enstrophy
    (no viscosity factor, unlike Kolmogorov flow's dissipation_series: KS's
    nu already sets its own dynamics, this is just a roughness/gradient scale)."""
    traj_np = traj.numpy() if torch.is_tensor(traj) else traj
    Nx = traj_np.shape[1]
    dx = L / Nx
    u = traj_np[..., 0]
    ux = np.gradient(u, dx, axis=1, edge_order=2)
    return (ux ** 2).mean(axis=1)


def energy_spectrum_1d(frame, L):
    """frame: (Nx, 1) or (Nx,) -- 1D counterpart of the 2D radial-binned
    spectrum: rfft along the single spatial axis, no radial binning needed
    (already 1D)."""
    u = frame[..., 0] if frame.ndim == 2 else frame
    Nx = u.shape[0]
    u_hat = np.fft.rfft(u) / Nx
    k = np.fft.rfftfreq(Nx, d=L / Nx / (2 * np.pi))  # physical wavenumbers, domain length L
    e = 0.5 * np.abs(u_hat) ** 2
    return k, e


def averaged_spectrum_1d(traj, L, n_avg=200):
    e_sum = None
    frames = traj[-n_avg:] if traj.shape[0] > n_avg else traj
    for frame in frames:
        frame_np = frame.numpy() if torch.is_tensor(frame) else frame
        k, e = energy_spectrum_1d(frame_np, L)
        e_sum = e.copy() if e_sum is None else e_sum + e
    return k, e_sum / len(frames)


# ── plots ────────────────────────────────────────────────────────────────────

def draw_space_time(gt_np, traj_no_np, traj_with_np, L, out_path):
    """Space-time heatmaps u(x,t) -- the standard way to visualize a KS
    trajectory, replaces the 2D vorticity-snapshot grid."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    vmax = np.abs(gt_np).max()
    for ax, data, title in zip(axes, (gt_np, traj_no_np, traj_with_np),
                               ("GT", "without correction", "with correction")):
        im = ax.imshow(data[..., 0].T, aspect="auto", origin="lower", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, extent=[0, data.shape[0], 0, L])
        ax.set_title(title)
        ax.set_xlabel("step")
        if ax is axes[0]:
            ax.set_ylabel("x")
    fig.colorbar(im, ax=axes, shrink=0.8, label="u")
    fig.suptitle("Kuramoto-Sivashinsky: space-time field", y=1.02)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def draw_energy_curve(gt_e, no_corr_e, with_corr_e, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gt_e, color="black", ls="--", lw=1.5, label="GT")
    ax.plot(no_corr_e, color="tab:red", lw=1, label="without correction")
    ax.plot(with_corr_e, color="tab:blue", lw=1, label="with correction")
    ax.set_xlabel("step")
    ax.set_ylabel(r"$E(t) = \frac{1}{2}\langle u^2 \rangle$")
    ax.set_title("KS field energy vs rollout step")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def draw_spectrum(gt_traj, traj_no, L, div_step, out_path, n_checkpoints=6):
    k_gt, e_gt = averaged_spectrum_1d(gt_traj, L)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(k_gt, e_gt, color="black", ls="--", lw=2, label="GT (stationary)", zorder=n_checkpoints + 1)

    checkpoints = np.linspace(0, traj_no.shape[0] - 1, n_checkpoints).astype(int)
    colors = plt.cm.viridis(np.linspace(0, 1, n_checkpoints))
    for t, c in zip(checkpoints, colors):
        k_t, e_t = energy_spectrum_1d(traj_no[t].numpy(), L)
        ax.loglog(k_t, e_t, color=c, lw=1)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, traj_no.shape[0]))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="rollout step (without correction)")

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E(k)$")
    ax.set_title("Energy spectrum: GT vs progressive divergence")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def draw_pdf_comparison(gt_np, traj_no_np, traj_with_np, out_path):
    samples_gt = gt_np[..., 0].ravel()
    samples_no = traj_no_np[..., 0].ravel()
    samples_with = traj_with_np[..., 0].ravel()

    lo = min(samples_gt.min(), samples_with.min())
    hi = max(samples_gt.max(), samples_with.max())
    margin = 0.1 * (hi - lo)
    grid = np.linspace(lo - margin, hi + margin, 300)

    fig, ax = plt.subplots(figsize=(7, 5))
    density_gt = gaussian_kde(samples_gt)(grid)
    ax.fill_between(grid, density_gt, color="black", alpha=0.2)
    ax.plot(grid, density_gt, color="black", lw=1.8, label="GT")

    finite_no = samples_no[np.isfinite(samples_no)]
    if finite_no.size >= 2 and np.abs(finite_no).max() < 1e6:
        density_no = gaussian_kde(finite_no)(grid)
        ax.fill_between(grid, density_no, color="tab:red", alpha=0.2)
        ax.plot(grid, density_no, color="tab:red", lw=1.8, label="without correction")

    density_with = gaussian_kde(samples_with)(grid)
    ax.fill_between(grid, density_with, color="tab:blue", alpha=0.3)
    ax.plot(grid, density_with, color="tab:blue", lw=1.8, label="with correction")

    ax.set_yscale("log")
    ax.set_xlabel("u")
    ax.set_ylabel("probability density")
    ax.set_title("Field-value distribution")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def draw_delay_embedding(gt_grad_e, no_corr_grad_e, with_corr_grad_e, tau, n_delays, burnin, out_path):
    embed_gt_full = time_delay_embedding(gt_grad_e, tau, n_delays)
    # Guard against a short GT window (e.g. a quick early divergence leaving
    # little history to embed): clip burnin rather than crash gaussian_kde
    # on a near-empty array.
    burnin = min(burnin, max(0, len(embed_gt_full) - 2))
    embed_gt = embed_gt_full[burnin:]
    if len(embed_gt) < 2:
        print(f"  [skip] delay embedding: not enough GT history ({len(embed_gt_full)} points) to plot", flush=True)
        return
    density_gt = gaussian_kde(embed_gt.T)(embed_gt.T)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(embed_gt[:, 0], embed_gt[:, 1], c=density_gt, s=4, cmap="plasma", linewidths=0, alpha=0.6)

    for label, series, color in [("without correction", no_corr_grad_e, "tab:red"),
                                 ("with correction", with_corr_grad_e, "tab:blue")]:
        embed = time_delay_embedding(series, tau, n_delays)
        mask = np.isfinite(embed).all(axis=1)
        embed = embed[mask]
        if embed.size:
            ax.plot(embed[:, 0], embed[:, 1], color=color, lw=0.7, alpha=0.85, label=label)

    x_range = embed_gt[:, 0].max() - embed_gt[:, 0].min()
    y_range = embed_gt[:, 1].max() - embed_gt[:, 1].min()
    margin = 2
    ax.set_xlim(embed_gt[:, 0].min() - margin * x_range, embed_gt[:, 0].max() + margin * x_range)
    ax.set_ylim(embed_gt[:, 1].min() - margin * y_range, embed_gt[:, 1].max() + margin * y_range)
    ax.set_xlabel(r"$\langle u_x^2 \rangle(t)$")
    ax.set_ylabel(r"$\langle u_x^2 \rangle(t-\tau)$")
    ax.set_title("Phase space (gradient-energy delay embedding)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--exp_dir", required=True, help="e.g. KS_equation/nu0p35")
    parser.add_argument("--sim_file", default="sim1.h5")
    parser.add_argument("--emulator_run", required=True)
    parser.add_argument("--diffusion_run", required=True)
    parser.add_argument("--nu", type=float, required=True)
    parser.add_argument("--L", type=float, default=22.0)
    parser.add_argument("--rollout_steps", type=int, default=5000)
    parser.add_argument("--divergence_factor", type=float, default=100.0)
    parser.add_argument("--s_init", type=int, default=1)
    parser.add_argument("--s_stop", type=int, default=0)
    parser.add_argument("--delay_tau", type=int, default=8)
    parser.add_argument("--delay_n_delays", type=int, default=2)
    parser.add_argument("--delay_burnin", type=int, default=40)
    parser.add_argument("--out_dir", default="correction_eval_ks")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                          else "cpu")

    print("Loading emulator + diffusion corrector...", flush=True)
    emulator, emul_cfg = load_emulator_fno1d(args.emulator_run, device)
    diffusion, diff_cfg = load_diffusion_1d(args.diffusion_run, device)
    prediction_mode = emul_cfg.get("prediction_mode", "delta")
    ds = emul_cfg.get("ds", 1)

    print("Loading GT trajectory...", flush=True)
    gt = load_gt_trajectory_ks(args.data_dir, args.exp_dir, args.sim_file, ds=ds)
    x0 = gt[0].unsqueeze(0).to(device)

    print(f"Rolling out WITHOUT correction (up to {args.rollout_steps} steps, "
          f"divergence_factor={args.divergence_factor})...", flush=True)
    traj_no, tracker_no = rollout(emulator, prediction_mode, x0, args.rollout_steps, emulator.tau, device,
                                  diffusion=None, divergence_factor=args.divergence_factor)
    div_step = tracker_no["stopped_early_at"] or traj_no.shape[0] - 1
    print(f"  diverged at step {div_step}" if tracker_no["stopped_early_at"] else "  did not diverge", flush=True)

    print(f"Rolling out WITH correction ({div_step} steps, same horizon)...", flush=True)
    traj_with, tracker_with = rollout(emulator, prediction_mode, x0, div_step, emulator.tau, device,
                                      diffusion=diffusion, s_init=args.s_init, s_stop=args.s_stop)
    print(f"  n_corrections={tracker_with['n_corrections']}", flush=True)

    gt_np = gt[:div_step + 1].numpy()
    traj_no_np = traj_no.numpy()
    traj_with_np = traj_with.numpy()

    print("Computing diagnostics + figures...", flush=True)
    gt_e = field_energy_series(gt[:div_step + 1])
    no_e = field_energy_series(traj_no)
    with_e = field_energy_series(traj_with)
    draw_energy_curve(gt_e, no_e, with_e, os.path.join(args.out_dir, "energy_curve.png"))

    draw_space_time(gt_np, traj_no_np, traj_with_np, args.L, os.path.join(args.out_dir, "space_time.png"))

    draw_spectrum(gt[:div_step + 200] if gt.shape[0] > div_step + 200 else gt, traj_no, args.L, div_step,
                 os.path.join(args.out_dir, "spectrum.png"))

    draw_pdf_comparison(gt_np, traj_no_np, traj_with_np, os.path.join(args.out_dir, "pdf_comparison.png"))

    gt_grad_e = gradient_energy_series(gt[:div_step + 1], args.L)
    no_grad_e = gradient_energy_series(traj_no, args.L)
    with_grad_e = gradient_energy_series(traj_with, args.L)
    draw_delay_embedding(gt_grad_e, no_grad_e, with_grad_e, args.delay_tau, args.delay_n_delays,
                         args.delay_burnin, os.path.join(args.out_dir, "delay_embedding.png"))

    np.savez(os.path.join(args.out_dir, "rollout_data.npz"),
            gt=gt_np, traj_no=traj_no_np, traj_with=traj_with_np,
            gt_e=gt_e, no_e=no_e, with_e=with_e, div_step=div_step,
            correction_flags=tracker_with["correction_flags"], noise_levels=tracker_with["noise_levels"])

    print(f"Done. Figures + data saved to {args.out_dir}/", flush=True)


if __name__ == "__main__":
    main()
