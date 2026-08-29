"""
Compare les 6 emulators du benchmark (FNO-delta, WNO-delta, WNO-state, FNO-state,
UNet-delta, UNet-state) avec et sans correction par le NOUVEAU modele de diffusion
U-Net (diffusion_unet_RE90), sur un rollout de Kolmogorov flow (Re90).

Script distinct de correction_eval.py (qui corrige avec le diffusion FNO) pour ne
pas perturber ce job -- reutilise ses loaders/rollout/metriques/plots par import.

Usage (depuis thermalizer/src) :
    python correction_eval_unet.py \
        --data_dir    $DATA_DIR \
        --runs_dir    $RUNS_DIR \
        --wnoarch_dir /path/to/WNO_arch \
        --rollout     2000 \
        --out_dir     correction_eval_unet_results
"""

import argparse
import os

import numpy as np
import torch

from correction_eval import (
    DEFAULT_RUNS,
    PREDICTION_MODE,
    DEFAULT_WNOARCH_FNO_STATE_RUN,
    load_config,
    load_checkpoint_dict,
    load_emulator_fno,
    load_emulator_wno,
    load_emulator_wnoarch,
    load_gt_trajectory,
    rollout,
    relative_l2_curve,
    kinetic_energy_curve,
    plot_error_curve_single,
    plot_kinetic_energy_single,
    plot_energy_spectrum_evolution,
    plot_correction_count_single,
    plot_noise_level_distribution_single,
    plot_vorticity_snapshots,
)
from training.unet_training import EmulatorUNet
from training.DiffusionModel import Diffusion
from unet.unet_2D_classifier import UNet2D_classifier


DEFAULT_UNET_RUNS = {
    "unet_delta": "unet_delta_RE90",
    "unet_state": "unet_state_RE90",
}
DEFAULT_UNET_DIFFUSION_RUN = "diffusion_unet_RE90"


def load_emulator_unet(run_dir, device):
    cfg = load_config(run_dir)
    model = EmulatorUNet(
        input_dim=cfg["input_dim"], output_dim=cfg["output_dim"],
        depth=cfg.get("depth", 3), base_width=cfg.get("base_width", 32),
        tau=cfg.get("tau", 1e-5), device=device,
    )
    ckpt = load_checkpoint_dict(run_dir, ["min_test_loss.pth", "final_model.pth", "min_train_loss.pth"], device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).float().eval(), cfg


def load_diffusion_unet(run_dir, device):
    cfg = load_config(run_dir)
    base = UNet2D_classifier(
        input_dim=cfg["input_dim"], output_dim=cfg["output_dim"],
        depth=cfg.get("depth", 3), base_width=cfg.get("base_width", 32),
        class_mlp_layers=cfg.get("class_mlp_layers"), n_cat=cfg["timesteps"], device=device,
    ).to(device).float()

    diffusion = Diffusion(model=base, timesteps=cfg["timesteps"],
                          noise_sampling_coeff=cfg.get("noise_sampling_coeff"))
    ckpt = load_checkpoint_dict(run_dir, ["final_model.pth", "min_test_loss.pth", "min_train_loss.pth"], device)
    diffusion.load_state_dict(ckpt["model_state_dict"])
    return diffusion.to(device).eval(), cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.environ.get("DATA_DIR", "../data"))
    parser.add_argument("--runs_dir", default=os.environ.get("RUNS_DIR", "../runs_mesu/Re90/Re90"))
    parser.add_argument("--exp_dir", default="kolmogorov/Re90")
    parser.add_argument("--sim_file", default="sim8.h5")
    parser.add_argument("--rollout", type=int, default=2000)
    parser.add_argument("--s_init", type=int, default=7)
    parser.add_argument("--s_stop", type=int, default=3)
    parser.add_argument("--out_dir", default="correction_eval_unet_results")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wnoarch_dir", default=os.environ.get("WNOARCH_DIR", "../../WNO_arch"))
    parser.add_argument("--wnoarch_runs_dir", default=None,
                        help="Dossier des runs WNO_arch (defaut: <wnoarch_dir>/runs/kolmogorov/Re90)")
    parser.add_argument("--fno_state_run", default=DEFAULT_WNOARCH_FNO_STATE_RUN)
    parser.add_argument("--skip_fno_state", action="store_true")
    parser.add_argument("--unet_delta_run", default=DEFAULT_UNET_RUNS["unet_delta"])
    parser.add_argument("--unet_state_run", default=DEFAULT_UNET_RUNS["unet_state"])
    parser.add_argument("--unet_diffusion_run", default=DEFAULT_UNET_DIFFUSION_RUN)
    parser.add_argument("--seed", type=int, default=101,
                        help="Seed torch pour le bruit tau*randn du rollout -- cf. correction_eval.py "
                        "pour le detail (101 choisi car il declenche de facon fiable la divergence "
                        "sans correction, plutot que de dependre d'un tirage non seede).")
    parser.add_argument("--divergence_factor", type=float, default=100.0,
                        help="Arrete un rollout des que son energie cinetique depasse ce facteur x "
                        "l'energie initiale -- cf. correction_eval.py. 0 ou negatif desactive.")
    args = parser.parse_args()
    divergence_factor = args.divergence_factor if args.divergence_factor and args.divergence_factor > 0 else None

    torch.manual_seed(args.seed)
    print(f"Seed torch : {args.seed}", flush=True)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Device utilise : {device}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)

    # ── chargement des modeles ────────────────────────────────────────────────
    print("Chargement du modele de diffusion U-Net...", flush=True)
    diffusion, diff_cfg = load_diffusion_unet(os.path.join(args.runs_dir, args.unet_diffusion_run), device)

    emulators = {}
    for name, run_name in DEFAULT_RUNS.items():
        run_dir = os.path.join(args.runs_dir, run_name)
        print(f"Chargement de l'emulator {name} ({run_name})...", flush=True)
        if name == "fno_delta":
            model, cfg = load_emulator_fno(run_dir, device)
        else:
            model, cfg = load_emulator_wno(run_dir, device)
        emulators[name] = {"model": model, "cfg": cfg, "prediction_mode": PREDICTION_MODE[name]}

    if not args.skip_fno_state:
        wnoarch_runs_dir = args.wnoarch_runs_dir or os.path.join(args.wnoarch_dir, "runs", "kolmogorov", "Re90")
        run_dir = os.path.join(wnoarch_runs_dir, args.fno_state_run)
        print(f"Chargement de l'emulator fno_state ({run_dir})...", flush=True)
        model, cfg = load_emulator_wnoarch(run_dir, args.wnoarch_dir, device)
        emulators["fno_state"] = {"model": model, "cfg": cfg, "prediction_mode": PREDICTION_MODE["fno_state"]}

    for name, run_name, prediction_mode in [
        ("unet_delta", args.unet_delta_run, "delta"),
        ("unet_state", args.unet_state_run, "state"),
    ]:
        run_dir = os.path.join(args.runs_dir, run_name)
        print(f"Chargement de l'emulator {name} ({run_name})...", flush=True)
        model, cfg = load_emulator_unet(run_dir, device)
        emulators[name] = {"model": model, "cfg": cfg, "prediction_mode": prediction_mode}

    # ── donnees GT ─────────────────────────────────────────────────────────────
    # Le rollout libre n'a besoin du GT que pour la toute premiere frame -- ne pas
    # plafonner n_steps a la longueur GT dispo (cf. correction_eval.py). gt_traj
    # garde sa longueur naturelle ; les comparaisons directes (erreur L2) sont
    # tronquees plus bas si le rollout va plus loin.
    ds = diff_cfg.get("ds", 2)
    gt_traj = load_gt_trajectory(args.data_dir, args.exp_dir, args.sim_file, ds=ds)
    n_steps = args.rollout
    n_gt_steps = gt_traj.shape[0] - 1
    if n_steps > n_gt_steps:
        print(f"Attention : rollout demande ({n_steps}) > longueur GT disponible ({n_gt_steps} pas). "
              f"Erreur L2 tronquee a {n_gt_steps} pas, les autres diagnostics couvrent le rollout complet.",
              flush=True)
    x0 = gt_traj[0:1].to(device)
    print(f"Trajectoire GT chargee : {gt_traj.shape}, rollout={n_steps} pas", flush=True)
    gt_ke = kinetic_energy_curve(gt_traj)

    # ── rollouts ───────────────────────────────────────────────────────────────
    results = {}

    for name, info in emulators.items():
        print(f"\n=== {name} ===", flush=True)
        tau = info["cfg"].get("tau", 1e-5)

        print("  Rollout sans correction...", flush=True)
        traj_no_corr, tracker_no_corr = rollout(
            info["model"], info["prediction_mode"], x0, n_steps, tau, device, diffusion=None,
            divergence_factor=divergence_factor,
        )

        print("  Rollout avec correction (diffusion U-Net)...", flush=True)
        traj_with_corr, tracker_with_corr = rollout(
            info["model"], info["prediction_mode"], x0, n_steps, tau, device,
            diffusion=diffusion, s_init=args.s_init, s_stop=args.s_stop,
            divergence_factor=divergence_factor,
        )
        print(f"  Corrections declenchees : {tracker_with_corr['n_corrections']} / "
              f"{tracker_with_corr['stopped_early_at'] or n_steps}", flush=True)
        if tracker_no_corr["stopped_early_at"]:
            print(f"  Arret anticipe (sans correction) au pas {tracker_no_corr['stopped_early_at']}/{n_steps}",
                  flush=True)
        if tracker_with_corr["stopped_early_at"]:
            print(f"  Arret anticipe (avec correction) au pas {tracker_with_corr['stopped_early_at']}/{n_steps}",
                  flush=True)

        common_len = min(traj_no_corr.shape[0], gt_traj.shape[0])
        error_no_corr = relative_l2_curve(traj_no_corr[:common_len], gt_traj[:common_len])
        error_with_corr = relative_l2_curve(traj_with_corr[:common_len], gt_traj[:common_len])
        ke_no_corr = kinetic_energy_curve(traj_no_corr)
        ke_with_corr = kinetic_energy_curve(traj_with_corr)

        results[name] = {
            "error_no_corr": error_no_corr,
            "error_with_corr": error_with_corr,
            "ke_no_corr": ke_no_corr,
            "ke_with_corr": ke_with_corr,
            "n_corrections": tracker_with_corr["n_corrections"],
            "mean_step_time_no_corr": float(tracker_no_corr["step_times"].mean()),
            "mean_step_time_with_corr": float(tracker_with_corr["step_times"].mean()),
            "stopped_early_at_no_corr": tracker_no_corr["stopped_early_at"] or "",
            "stopped_early_at_with_corr": tracker_with_corr["stopped_early_at"] or "",
        }
        plot_vorticity_snapshots(
            name, gt_traj, traj_no_corr, traj_with_corr,
            os.path.join(args.out_dir, f"vorticity_{name}.png"),
        )
        plot_energy_spectrum_evolution(
            traj_with_corr, gt_traj, f"{name} (avec correction U-Net)",
            os.path.join(args.out_dir, f"spectrum_{name}_with_corr.png"),
        )
        plot_energy_spectrum_evolution(
            traj_no_corr, gt_traj, f"{name} (sans correction)",
            os.path.join(args.out_dir, f"spectrum_{name}_no_corr.png"),
        )
        plot_error_curve_single(
            name, error_no_corr, error_with_corr,
            os.path.join(args.out_dir, f"error_curve_{name}.png"),
        )
        plot_kinetic_energy_single(
            name, ke_no_corr, ke_with_corr, gt_ke,
            os.path.join(args.out_dir, f"kinetic_energy_{name}.png"),
        )
        plot_correction_count_single(
            name, tracker_with_corr,
            os.path.join(args.out_dir, f"correction_count_{name}.png"),
        )
        plot_noise_level_distribution_single(
            name, tracker_with_corr, args.s_init,
            os.path.join(args.out_dir, f"noise_level_distribution_{name}.png"),
        )

        np.savez(
            os.path.join(args.out_dir, f"{name}_curves.npz"),
            error_no_corr=error_no_corr, error_with_corr=error_with_corr,
            ke_no_corr=ke_no_corr, ke_with_corr=ke_with_corr,
            correction_flags=tracker_with_corr["correction_flags"],
            noise_levels=tracker_with_corr["noise_levels"],
        )

    summary_path = os.path.join(args.out_dir, "summary.csv")
    with open(summary_path, "w") as f:
        f.write("emulator,n_corrections,final_error_no_corr,final_error_with_corr,"
                "mean_step_time_no_corr_s,mean_step_time_with_corr_s,"
                "stopped_early_at_no_corr,stopped_early_at_with_corr\n")
        for name, res in results.items():
            f.write(f"{name},{res['n_corrections']},"
                    f"{res['error_no_corr'][-1]:.6f},{res['error_with_corr'][-1]:.6f},"
                    f"{res['mean_step_time_no_corr']:.6f},{res['mean_step_time_with_corr']:.6f},"
                    f"{res['stopped_early_at_no_corr']},{res['stopped_early_at_with_corr']}\n")

    print(f"\nResultats sauvegardes dans : {args.out_dir}/", flush=True)
    for f in sorted(os.listdir(args.out_dir)):
        print(f"  {f}", flush=True)


if __name__ == "__main__":
    main()
