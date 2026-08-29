"""
Compare les emulators sequentiels (k=5) de WNO_arch (FNO et WNO, fenetre de 5
frames empilees) avec et sans correction par le modele de diffusion de
thermalizer, sur un rollout de Kolmogorov flow (Re90).

Script distinct de correction_eval.py (qui traite fno_delta/wno_delta/wno_state/
fno_state) pour ne pas interferer avec ce job -- meme si les deux partagent
les memes fonctions de rollout/metriques/plots.

Usage (depuis thermalizer/src) :
    python correction_eval_seq_k5.py \
        --data_dir         $DATA_DIR \
        --runs_dir         $RUNS_DIR \
        --wnoarch_dir      /path/to/WNO_arch \
        --wnoarch_runs_dir /path/to/wno_arch/runs/kolmogorov/Re90 \
        --rollout          2000 \
        --out_dir          correction_eval_seq_k5_results
"""

import argparse
import os

import numpy as np
import torch

from correction_eval import (
    DEFAULT_DIFFUSION_RUN,
    DEFAULT_WNOARCH_SEQ_K5_RUNS,
    load_diffusion,
    load_emulator_wnoarch,
    load_gt_trajectory,
    rollout_windowed,
    relative_l2_curve,
    kinetic_energy_curve,
    plot_error_curve_single,
    plot_kinetic_energy_single,
    plot_energy_spectrum_evolution,
    plot_correction_count_single,
    plot_noise_level_distribution_single,
    plot_vorticity_snapshots,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.environ.get("DATA_DIR", "../data"))
    parser.add_argument("--runs_dir", default=os.environ.get("RUNS_DIR", "../runs_mesu/Re90/Re90"),
                        help="Dossier des runs thermalizer (pour le modele de diffusion)")
    parser.add_argument("--exp_dir", default="kolmogorov/Re90")
    parser.add_argument("--sim_file", default="sim8.h5")
    parser.add_argument("--rollout", type=int, default=2000)
    parser.add_argument("--s_init", type=int, default=7)
    parser.add_argument("--s_stop", type=int, default=3)
    parser.add_argument("--out_dir", default="correction_eval_seq_k5_results")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wnoarch_dir", default=os.environ.get("WNOARCH_DIR", "../../WNO_arch"))
    parser.add_argument("--wnoarch_runs_dir", default=None,
                        help="Dossier des runs WNO_arch (defaut: <wnoarch_dir>/runs/kolmogorov/Re90)")
    parser.add_argument("--fno_seq_k5_run", default=DEFAULT_WNOARCH_SEQ_K5_RUNS["fno_seq_k5"])
    parser.add_argument("--wno_seq_k5_run", default=DEFAULT_WNOARCH_SEQ_K5_RUNS["wno_seq_k5"])
    parser.add_argument("--seed", type=int, default=101,
                        help="Seed torch pour le rollout -- cf. correction_eval.py.")
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
    print("Chargement du modele de diffusion...", flush=True)
    diffusion, diff_cfg = load_diffusion(os.path.join(args.runs_dir, DEFAULT_DIFFUSION_RUN), device)

    wnoarch_runs_dir = args.wnoarch_runs_dir or os.path.join(args.wnoarch_dir, "runs", "kolmogorov", "Re90")

    emulators = {}
    for name, run_name in [("fno_seq_k5", args.fno_seq_k5_run), ("wno_seq_k5", args.wno_seq_k5_run)]:
        run_dir = os.path.join(wnoarch_runs_dir, run_name)
        print(f"Chargement de l'emulator {name} ({run_dir})...", flush=True)
        model, cfg = load_emulator_wnoarch(run_dir, args.wnoarch_dir, device)
        emulators[name] = {"model": model, "cfg": cfg, "k": cfg["k"],
                           "n_channels": cfg.get("n_channels", 2)}

    # ── donnees GT ─────────────────────────────────────────────────────────────
    # Le rollout libre n'a besoin du GT que pour le contexte initial -- ne pas
    # plafonner n_steps a la longueur GT dispo (cf. correction_eval.py).
    ds = diff_cfg.get("ds", 2)
    max_k = max(info["k"] for info in emulators.values())
    gt_full = load_gt_trajectory(args.data_dir, args.exp_dir, args.sim_file, ds=ds)
    n_steps = args.rollout
    n_gt_steps = gt_full.shape[0] - max_k
    if n_steps > n_gt_steps:
        print(f"Attention : rollout demande ({n_steps}) > longueur GT disponible ({n_gt_steps} pas). "
              f"Erreur L2 tronquee a {n_gt_steps} pas, les autres diagnostics couvrent le rollout complet.",
              flush=True)
    gt_traj = gt_full[max_k - 1:]
    print(f"Trajectoire GT chargee : {gt_full.shape}, rollout={n_steps} pas, contexte max k={max_k}", flush=True)
    gt_ke = kinetic_energy_curve(gt_traj)

    # ── rollouts ───────────────────────────────────────────────────────────────
    results = {}

    for name, info in emulators.items():
        print(f"\n=== {name} (k={info['k']}) ===", flush=True)
        k = info["k"]
        gt_context = gt_full[max_k - k: max_k]

        print("  Rollout sans correction...", flush=True)
        traj_no_corr, tracker_no_corr = rollout_windowed(
            info["model"], k, info["n_channels"], gt_context, n_steps, device, diffusion=None,
            divergence_factor=divergence_factor,
        )

        print("  Rollout avec correction...", flush=True)
        traj_with_corr, tracker_with_corr = rollout_windowed(
            info["model"], k, info["n_channels"], gt_context, n_steps, device,
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
            traj_with_corr, gt_traj, f"{name} (avec correction)",
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
