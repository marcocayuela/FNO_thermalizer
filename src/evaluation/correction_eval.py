"""
Compare plusieurs emulators (FNO-delta, WNO-delta, WNO-state, FNO-state) avec
et sans correction par un modele de diffusion, sur un rollout de Kolmogorov
flow (Re90).

Usage (depuis thermalizer/src) :
    python correction_eval.py \
        --data_dir    $DATA_DIR \
        --runs_dir    $LOG_DIR/kolmogorov/Re90 \
        --wnoarch_dir /path/to/WNO_arch \
        --rollout     2000 \
        --out_dir     correction_eval_results

Modeles par defaut (voir DEFAULT_RUNS / DEFAULT_DIFFUSION_RUN) : selectionnes
via les metrics.csv de chaque run (meilleur relative_rmse test pour les
emulators, meilleur loss_score pour la diffusion). Voir le plan associe pour
le detail du choix.

Le 4eme combo (FNO-state) n'a pas d'equivalent entraine dans thermalizer : on
reutilise le modele "emul_seq_fno_k1" de WNO_arch, qui predit l'etat absolu a
partir d'une seule frame (k=1) -- interface identique a un FNO-state classique
(voir training/dataset_emul_seq.py : cibles absolues, pas de delta).
"""

import argparse
import math
import os
import sys
import time

import h5py
import numpy as np
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import gaussian_kde

# fno/ and training/ are siblings of evaluation/ (this file's directory) under
# src/ -- add src/ itself (not just evaluation/) so this script works without
# relying on PYTHONPATH being set externally (cf. scripts/submit_correction_eval.sh),
# whether run as `python evaluation/correction_eval.py` from src/ or from elsewhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.fno_training import EmulatorFNO
from training.wno_training import EmulatorWNO
from training.DiffusionModel import Diffusion
from fno.fno_2D_classifier import FNO2D_classifier
from fno.fno_2D_classifier_concat import FNO2D_classifier_concat


DEFAULT_RUNS = {
    "fno_delta": "16modes_emul_RE90",
    "wno_delta": "wno_emul_lvl2_lay4_db6_delta_seq1",
    "wno_state": "wno_emul_lvl3_lay4_db6_state_seq1",
}
PREDICTION_MODE = {
    "fno_delta": "delta",
    "wno_delta": "delta",
    "wno_state": "state",
    "fno_state": "state",
}
DEFAULT_DIFFUSION_RUN = "diffusion16modes_RE90"

# fno_state vient de WNO_arch (pas de thermalizer/runs_mesu) : run "emul_seq_fno_k1",
# k=1 => input_dim=output_dim=n_channels=2, cibles absolues (etat, pas delta).
DEFAULT_WNOARCH_FNO_STATE_RUN = "emul_seq_fno_k1"

# Runs "emul_seq_*_k5" de WNO_arch : fenetre de k=5 frames en entree, etat absolu en
# sortie. Rollout gere par rollout_windowed() (glissement de la fenetre + correction
# de la seule frame predite a chaque pas).
DEFAULT_WNOARCH_SEQ_K5_RUNS = {
    "fno_seq_k5": "emul_seq_fno_k5",
    "wno_seq_k5": "emul_seq_wno_k5",
}


# ── chargement config / modeles ───────────────────────────────────────────────

def load_config(run_dir):
    with open(os.path.join(run_dir, "config.yaml")) as f:
        return yaml.safe_load(f)


def load_checkpoint(run_dir, preferred_names):
    for name in preferred_names:
        path = os.path.join(run_dir, "model_weights", name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Aucun checkpoint trouve dans {run_dir}/model_weights (essaye: {preferred_names})")


def load_checkpoint_dict(run_dir, preferred_names, device):
    """
    Essaie chaque checkpoint de preferred_names dans l'ordre et retourne le premier qui se
    charge correctement. Contrairement a load_checkpoint() (qui ne verifie que l'existence du
    fichier), celui-ci passe au suivant si le fichier existe mais est corrompu/tronque (ex:
    "PytorchStreamReader failed reading zip archive" -- arrive si l'ecriture du checkpoint a
    ete interrompue sur le cluster).
    """
    last_err = None
    tried = []
    for name in preferred_names:
        path = os.path.join(run_dir, "model_weights", name)
        if not os.path.exists(path):
            continue
        tried.append(path)
        try:
            return torch.load(path, map_location=device)
        except Exception as e:
            print(f"  [AVERTISSEMENT] Checkpoint illisible, ignore : {path} ({e})", flush=True)
            last_err = e
    raise FileNotFoundError(
        f"Aucun checkpoint valide dans {run_dir}/model_weights (essayes: {tried or preferred_names}). "
        f"Derniere erreur: {last_err}"
    )


def load_emulator_fno(run_dir, device):
    cfg = load_config(run_dir)
    model = EmulatorFNO(
        input_dim=cfg["input_dim"], output_dim=cfg["output_dim"],
        modes_x=cfg["k_max_x"], modes_y=cfg["k_max_y"], width=cfg["width"], l=cfg["l"],
        n_layer=cfg["n_fourier_layer"], hidden_proj=cfg.get("hidden_proj"), mlp=cfg.get("mlp", True),
        layers_mlp=cfg.get("layers_mlp"), tau=cfg.get("tau", 1e-5), device=device,
    )
    ckpt = load_checkpoint_dict(run_dir, ["min_test_loss.pth", "final_model.pth", "min_train_loss.pth"], device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).float().eval(), cfg


def load_emulator_wno(run_dir, device):
    cfg = load_config(run_dir)
    size = cfg["n_dim"] * [cfg["res"]]
    model = EmulatorWNO(
        width=cfg["width"], level=cfg["level"], layers=cfg["layers"], size=size,
        wavelet=cfg["wavelet"], mode=cfg.get("mode", "periodization"),
        in_channel=cfg["input_dim"], out_channel=cfg["output_dim"], grid_range=cfg["domain_size"],
        tau=cfg.get("tau", 1e-5), padding=cfg.get("padding", 0), device=device,
    )
    ckpt = load_checkpoint_dict(run_dir, ["min_test_loss.pth", "final_model.pth", "min_train_loss.pth"], device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).float().eval(), cfg


def load_emulator_wnoarch(run_dir, wnoarch_dir, device):
    """
    Charge un emulator entraine avec le code de WNO_arch (via models.model_factory.build_model).
    Le config.yaml du run contient deja input_dim/output_dim calcules a l'entrainement
    (voir WNO_arch/main_emul_seq.py:63-64 : input_dim = n_channels*k, output_dim = n_channels).
    Predit toujours l'etat absolu (pas de delta). Pour k>1, le modele attend une fenetre
    de k frames empilees en canaux -- voir rollout_windowed().
    """
    if wnoarch_dir not in sys.path:
        sys.path.insert(0, wnoarch_dir)
    from models.model_factory import build_model

    cfg = load_config(run_dir)
    model = build_model(cfg, device).to(device).float()
    ckpt = load_checkpoint_dict(run_dir, ["best_model.pth", "final_model.pth"], device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).float().eval(), cfg


def load_diffusion(run_dir, device):
    """
    Charge un modele de diffusion. Si le config.yaml du run contient
    "re_values" (cf. configs/config_command_diff_concat.yaml), c'est un
    corrector Re-conditionne (FNO2D_classifier_concat, cf.
    fno/fno_2D_classifier_concat.py) -- param_mean/param_std passes ici sont
    des placeholders, les vraies valeurs (figees a l'entrainement) sont
    restaurees par load_state_dict() (ce sont des buffers du checkpoint).
    Sinon, comportement inchange : FNO2D_classifier non conditionne.
    """
    cfg = load_config(run_dir)
    is_parametric = "re_values" in cfg

    if is_parametric:
        base = FNO2D_classifier_concat(
            input_dim=cfg["input_dim"], output_dim=cfg["output_dim"],
            modes_x=cfg["k_max_x"], modes_y=cfg["k_max_y"], width=cfg["width"], l=cfg["l"],
            n_layer=cfg["n_fourier_layer"], hidden_proj=cfg.get("hidden_proj"), mlp=cfg.get("mlp", True),
            layers_mlp=cfg.get("layers_mlp"), class_mlp_layers=cfg.get("class_mlp_layers"),
            n_cat=cfg["timesteps"], param_mean=0.0, param_std=1.0, device=device,
        ).to(device).float()
    else:
        base = FNO2D_classifier(
            input_dim=cfg["input_dim"], output_dim=cfg["output_dim"],
            modes_x=cfg["k_max_x"], modes_y=cfg["k_max_y"], width=cfg["width"], l=cfg["l"],
            n_layer=cfg["n_fourier_layer"], hidden_proj=cfg.get("hidden_proj"), mlp=cfg.get("mlp", True),
            layers_mlp=cfg.get("layers_mlp"), class_mlp_layers=cfg.get("class_mlp_layers"),
            n_cat=cfg["timesteps"], device=device,
        ).to(device).float()

    diffusion = Diffusion(model=base, timesteps=cfg["timesteps"],
                          noise_sampling_coeff=cfg.get("noise_sampling_coeff"))
    ckpt = load_checkpoint_dict(run_dir, ["final_model.pth", "min_test_loss.pth", "min_train_loss.pth"], device)
    diffusion.load_state_dict(ckpt["model_state_dict"])
    return diffusion.to(device).eval(), cfg


# ── donnees ────────────────────────────────────────────────────────────────────

def load_gt_trajectory(data_dir, exp_dir, sim_file, ds, ratio=1):
    path = os.path.join(data_dir, exp_dir, "train_traj", sim_file)
    with h5py.File(path, "r") as f:
        data = f["velocity_field"][()][::ratio, ::ds, ::ds]
    return torch.from_numpy(data).float()


# ── rollout ────────────────────────────────────────────────────────────────────

def maybe_correct(x_t, diffusion, s_init, s_stop, device, re=None):
    """
    Corrige un etat (1, H, W, C) par le modele de diffusion si le niveau de bruit
    detecte par sa tete classifieur depasse s_init. Reprend compute_next_step()
    de notebooks/diffusion_results.ipynb (cellule 23).

    re : (1,) Reynolds de la trajectoire courante, requis si diffusion enveloppe
        un FNO2D_classifier_concat (corrector Re-conditionne). None (defaut) =
        comportement inchange pour un corrector mono-Re.

    diffusion peut aussi etre un training.ResidualCorrectorModel.ResidualCorrector
    (pas un vrai modele de diffusion -- correction en un seul passage,
    entrainee sur la vraie derive d'un emulator plutot que du bruit gaussien
    synthetique, cf. sa propre docstring) : branche geree ci-dessous sans
    toucher au chemin DDPM existant.

    Retourne (x_t corrige ou non, correction_declenchee: bool, niveau de bruit detecte: int).
    """
    from training.ResidualCorrectorModel import ResidualCorrector
    if isinstance(diffusion, ResidualCorrector):
        # Single forward pass: no noise schedule, no iterative reverse
        # process (cf. ResidualCorrector's own docstring for why an
        # iterative DDPM-style loop has no principled meaning for a
        # deterministic rollout-drift corruption). No explicit
        # normalize/denormalize step either -- Euler data already arrives
        # pre-normalized from the data pipeline (unlike the (u,v) Kolmogorov
        # fields Diffusion.normalize/denormalize were added for).
        residual, logits = diffusion.model(x_t, predict_class=True)
        s_hat = int(torch.softmax(logits, dim=-1).argmax(dim=-1).reshape(-1)[0].item())
        if s_hat <= s_init:
            return x_t, False, s_hat
        return x_t + residual, True, s_hat

    # diffusion.model/_forward_diffusion/_reverse_diffusion are called directly
    # here (not via Diffusion.forward()), so normalization has to be applied
    # explicitly: once in, once out, everything in between (classifier +
    # noise/denoise loop) runs in the same normalized space training used.
    x_norm = diffusion.normalize(x_t)
    if re is not None:
        _, logits = diffusion.model(x_norm, re, True)
    else:
        _, logits = diffusion.model(x_norm, True)
    t_hat = int(torch.softmax(logits, dim=-1).argmax(dim=-1).reshape(-1)[0].item())

    if t_hat <= s_init:
        return x_t, False, t_hat

    t_tensor = torch.full((x_norm.shape[0],), t_hat, dtype=torch.long, device=device)
    eps = torch.randn_like(x_norm)
    x_norm = diffusion._forward_diffusion(x_norm, t_tensor, eps)
    for s in range(t_hat, s_stop - 1, -1):
        z = torch.randn_like(x_norm)
        s_tensor = torch.full((x_norm.shape[0],), s, dtype=torch.long, device=device)
        x_norm = diffusion._reverse_diffusion(x_norm, s_tensor, z, re=re)
    return diffusion.denormalize(x_norm), True, t_hat


def _kinetic_energy_scalar(x):
    """x : (1, H, W, C) -> energie cinetique moyenne (python float)."""
    return float((0.5 * (x ** 2).sum(dim=-1)).mean().item())


def _is_diverged(x, reference_energy, divergence_factor):
    """
    True si l'etat x est non-fini, ou si son energie cinetique depasse
    divergence_factor x l'energie de reference (typiquement l'energie initiale).
    divergence_factor=None desactive la detection (comportement d'origine).
    """
    if divergence_factor is None:
        return False
    e = _kinetic_energy_scalar(x)
    return (not math.isfinite(e)) or (e > divergence_factor * reference_energy)


@torch.no_grad()
def rollout(emulator, prediction_mode, x0, n_steps, tau, device,
            diffusion=None, s_init=7, s_stop=3, divergence_factor=None, re=None):
    """
    x0 : (1, H, W, C) etat initial

    re : (1,) Reynolds de la trajectoire, transmis a maybe_correct() -- requis
        si diffusion enveloppe un corrector Re-conditionne (cf. maybe_correct).

    divergence_factor : si fourni, arrete le rollout des que l'energie cinetique
        de l'etat courant (apres correction eventuelle -- la correction a donc
        toujours sa chance d'agir avant qu'on abandonne) depasse
        divergence_factor x l'energie initiale, ou devient non-finie. Evite de
        continuer (et corriger, couteux) une trajectoire deja irrecuperable sur
        des milliers de pas restants. None (defaut) = desactive.

    Retourne la trajectoire predite (n_steps+1 pas, ou moins si arret anticipe)
    et un tracker de correction (avec 'stopped_early_at': int ou None).
    """
    # Field shape read generically from x0 (not hardcoded to 4D (1,H,W,C)) --
    # (n_steps+1, H, W, C) for a 2D Kolmogorov field, (n_steps+1, Nx, C) for
    # a 1D KS field (cf. evaluation/correction_eval_ks.py), unchanged for
    # existing 4D callers.
    traj = torch.empty((n_steps + 1, *x0.shape[1:]), device=device)
    traj[0] = x0[0]

    x_t = x0.to(device)
    reference_energy = _kinetic_energy_scalar(x_t) if divergence_factor is not None else None
    correction_flags = np.zeros(n_steps, dtype=bool)
    noise_levels = np.zeros(n_steps, dtype=np.int64)
    step_times = np.zeros(n_steps, dtype=np.float64)
    stopped_early_at = None

    for i in range(n_steps):
        t0 = time.perf_counter()

        if prediction_mode == "state":
            x_t = emulator(x_t)
        else:
            x_t = x_t + emulator(x_t) + tau * torch.randn_like(x_t)

        if diffusion is not None:
            x_t, corrected, t_hat = maybe_correct(x_t, diffusion, s_init, s_stop, device, re=re)
            correction_flags[i] = corrected
            noise_levels[i] = t_hat

        step_times[i] = time.perf_counter() - t0
        traj[i + 1] = x_t[0]

        if _is_diverged(x_t, reference_energy, divergence_factor):
            stopped_early_at = i + 1
            break

    if stopped_early_at is not None:
        traj = traj[: stopped_early_at + 1]
        correction_flags = correction_flags[:stopped_early_at]
        noise_levels = noise_levels[:stopped_early_at]
        step_times = step_times[:stopped_early_at]

    tracker = {
        "correction_flags": correction_flags,
        "noise_levels": noise_levels,
        "step_times": step_times,
        "n_corrections": int(correction_flags.sum()),
        "stopped_early_at": stopped_early_at,
    }
    return traj.cpu(), tracker


@torch.no_grad()
def rollout_windowed(emulator, k, n_channels, gt_context, n_steps, device,
                     prediction_mode="state",
                     diffusion=None, s_init=7, s_stop=3, divergence_factor=None, re=None):
    """
    Rollout pour les emulators "emul_seq_*" de WNO_arch, qui prennent une fenetre
    glissante de k frames empilees en canaux. A chaque pas : prediction, puis
    correction diffusion optionnelle sur cette seule frame (l'etat absolu
    reconstruit, pas l'increment brut si prediction_mode="delta"), puis la
    fenetre glisse (on enleve la plus ancienne frame, on ajoute la nouvelle).

    prediction_mode : "state" (sortie = etat absolu, comportement historique)
        ou "delta" (sortie = increment, ajoute au dernier snapshot du contexte
        -- meme convention que training/trainer_emul_seq.py:_rollout et
        analysis/eval_emul_seq.py:autoregressive_rollout dans WNO_arch).

    gt_context : (k, H, W, C) -- les k dernieres frames GT servant de contexte initial.
    divergence_factor : cf. rollout() -- arret anticipe si l'energie depasse ce
        facteur x l'energie du dernier frame de contexte, ou devient non-finie.
    Retourne une trajectoire (jusqu'a n_steps+1 pas, tronquee si arret anticipe)
    ou traj[0] = derniere frame du contexte (le "seed"), meme convention que rollout().
    """
    H, W, C = gt_context.shape[1], gt_context.shape[2], gt_context.shape[3]
    traj = torch.empty((n_steps + 1, H, W, C), device=device)
    traj[0] = gt_context[-1].to(device)

    # (k, H, W, C) -> (H, W, k*C), frames ordonnees par temps (comme a l'entrainement)
    ctx = gt_context.permute(1, 2, 0, 3).reshape(1, H, W, k * C).to(device)

    reference_energy = _kinetic_energy_scalar(traj[0:1]) if divergence_factor is not None else None
    correction_flags = np.zeros(n_steps, dtype=bool)
    noise_levels = np.zeros(n_steps, dtype=np.int64)
    step_times = np.zeros(n_steps, dtype=np.float64)
    stopped_early_at = None

    for i in range(n_steps):
        t0 = time.perf_counter()

        raw = emulator(ctx)  # (1, H, W, C)
        pred = ctx[..., -n_channels:] + raw if prediction_mode == "delta" else raw

        if diffusion is not None:
            pred, corrected, t_hat = maybe_correct(pred, diffusion, s_init, s_stop, device, re=re)
            correction_flags[i] = corrected
            noise_levels[i] = t_hat

        ctx = torch.cat([ctx[..., n_channels:], pred], dim=-1)
        step_times[i] = time.perf_counter() - t0
        traj[i + 1] = pred[0]

        if _is_diverged(pred, reference_energy, divergence_factor):
            stopped_early_at = i + 1
            break

    if stopped_early_at is not None:
        traj = traj[: stopped_early_at + 1]
        correction_flags = correction_flags[:stopped_early_at]
        noise_levels = noise_levels[:stopped_early_at]
        step_times = step_times[:stopped_early_at]

    tracker = {
        "correction_flags": correction_flags,
        "noise_levels": noise_levels,
        "step_times": step_times,
        "n_corrections": int(correction_flags.sum()),
        "stopped_early_at": stopped_early_at,
    }
    return traj.cpu(), tracker


# ── metriques ──────────────────────────────────────────────────────────────────

def relative_l2_curve(pred, gt):
    """pred, gt : (T, H, W, C). Retourne un vecteur (T,) d'erreur L2 relative par pas de temps."""
    diff = (pred - gt).reshape(pred.shape[0], -1)
    ref = gt.reshape(gt.shape[0], -1)
    return (diff.norm(dim=1) / (ref.norm(dim=1) + 1e-8)).numpy()


def vorticity(velocity_field):
    """velocity_field : (..., H, W, 2) -> (..., H, W)"""
    u = velocity_field[..., 0]
    v = velocity_field[..., 1]
    du_dy = np.gradient(u, axis=-1)
    dv_dx = np.gradient(v, axis=-2)
    return dv_dx - du_dy


def kinetic_energy_curve(traj):
    """traj : (T, H, W, C) -> (T,) energie cinetique moyenne spatiale."""
    return (0.5 * (traj ** 2).sum(dim=-1)).mean(dim=(1, 2)).numpy()


# ── dissipation / embedding en delai temporel ──────────────────────────────────
# Reprend exactement la methode de scale_separation.ipynb (deja utilisee pour
# fno_state plus tot dans le projet) : la comparaison ponctuelle GT/prediction
# n'a pas de sens au-dela de l'horizon de predictibilite (chaos) -- ce qui
# compte c'est de savoir si la trajectoire reste sur le bon attracteur,
# statistiquement, pas si elle suit le meme chemin precis.

def compute_dissipation_curve(traj, nu=1 / 90):
    """
    traj : (T, H, W, 2) (u, v). Retourne D(t), (T,) -- dissipation visqueuse
    moyenne spatiale, nu*mean(ux^2+uy^2+vx^2+vy^2).
    """
    traj_np = traj.numpy() if torch.is_tensor(traj) else traj
    T, H, W, _ = traj_np.shape
    dx = 2 * np.pi / H
    dy = 2 * np.pi / W
    D = np.zeros(T)
    for t in range(T):
        u, v = traj_np[t, ..., 0], traj_np[t, ..., 1]
        ux = np.gradient(u, dx, axis=0, edge_order=2)
        uy = np.gradient(u, dy, axis=1, edge_order=2)
        vx = np.gradient(v, dx, axis=0, edge_order=2)
        vy = np.gradient(v, dy, axis=1, edge_order=2)
        D[t] = nu * np.mean(ux ** 2 + uy ** 2 + vx ** 2 + vy ** 2)
    return D


def time_delay_embedding(x, tau, n_delays):
    """x : (N,). Retourne (N-(n_delays-1)*tau, n_delays)."""
    N = len(x)
    embedded = np.zeros((N - (n_delays - 1) * tau, n_delays))
    for i in range(n_delays):
        embedded[:, i] = x[(n_delays - 1 - i) * tau: N - i * tau]
    return embedded


def build_background_attractor(data_dir, exp_dir, ds, tau=8, n_delays=2):
    """
    Charge toutes les trajectoires d'entrainement disponibles (train_traj/*.h5),
    calcule leur dissipation, les embed en delai temporel -- nuage de fond
    representant l'attracteur statistique "vrai", pour comparer visuellement
    la trajectoire d'un modele (embedding en delai temporel de sa propre
    dissipation) sans dependre d'une correspondance ponctuelle avec le GT.
    """
    train_dir = os.path.join(data_dir, exp_dir, "train_traj")
    sim_files = sorted(f for f in os.listdir(train_dir) if f.endswith(".h5"))
    all_points = []
    for fname in sim_files:
        with h5py.File(os.path.join(train_dir, fname), "r") as f:
            data = f["velocity_field"][()][:, ::ds, ::ds]
        D = compute_dissipation_curve(data)
        all_points.append(time_delay_embedding(D, tau=tau, n_delays=n_delays))
    X = np.vstack(all_points)
    return X[40:, 0], X[40:, 1]  # (Xp, Yp), memes conventions que scale_separation.ipynb


def compute_classical_energy_spectrum(u):
    """
    u : ndarray (T, H, W, 2). Retourne (k_vals, E_k) moyennes sur T.
    Repris de notebooks/FNO_results.ipynb.
    """
    tsteps, nx, ny, _ = u.shape
    E_k_sum = None

    for t in range(tsteps):
        ux = u[t, :, :, 0]
        uy = u[t, :, :, 1]

        uxf = np.fft.fft2(ux)
        uyf = np.fft.fft2(uy)

        E2D = 0.5 * (np.abs(uxf) ** 2 + np.abs(uyf) ** 2) / (nx * ny)
        E2D = np.fft.fftshift(E2D)

        kx = np.fft.fftshift(np.fft.fftfreq(nx)) * nx
        ky = np.fft.fftshift(np.fft.fftfreq(ny)) * ny
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        k_mag = np.sqrt(KX ** 2 + KY ** 2)

        k_max = int(np.max(k_mag))
        E_k = np.zeros(k_max)
        for i in range(k_max):
            mask = (k_mag >= i) & (k_mag < i + 1)
            E_k[i] += np.sum(E2D[mask])

        if E_k_sum is None:
            E_k_sum = E_k
        else:
            E_k_sum += E_k

    E_k_mean = E_k_sum / tsteps
    k_vals = np.arange(len(E_k_mean))
    return k_vals, E_k_mean


# ── plots ──────────────────────────────────────────────────────────────────────
# Un plot par emulator (pas de superposition) : les echelles d'erreur/energie
# divergent trop d'un modele a l'autre (ex: wno_delta ~1e34 vs fno_delta ~1e0)
# pour etre lisibles sur un axe partage -- cf. discussion session precedente.

def plot_error_curve_single(name, error_no_corr, error_with_corr, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    t = np.arange(len(error_no_corr))
    ax.plot(t, error_no_corr, color="tab:red", ls="--", label="sans correction")
    ax.plot(t, error_with_corr, color="tab:blue", ls="-", label="avec correction")
    ax.set_xlabel("Pas de rollout")
    ax.set_ylabel("Erreur L2 relative")
    ax.set_yscale("log")
    ax.set_title(f"Erreur de trajectoire au cours du temps — {name}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_kinetic_energy_single(name, ke_no_corr, ke_with_corr, gt_ke, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(gt_ke, color="black", lw=2.5, label="GT")
    ax.plot(ke_no_corr, color="tab:red", ls="--", label="sans correction")
    ax.plot(ke_with_corr, color="tab:blue", ls="-", label="avec correction")
    ax.set_xlabel("Pas de rollout")
    ax.set_ylabel("Energie cinetique moyenne")
    ax.set_yscale("log")
    ax.set_title(f"Energie cinetique au cours du temps — {name}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_energy_spectrum_evolution(traj, gt_traj, name, out_path, block_size=100):
    nt = traj.shape[0]
    n_blocks = max(1, nt // block_size)
    cmap = cm.viridis
    colors = cmap(np.linspace(0.2, 0.95, n_blocks))

    fig, ax = plt.subplots(figsize=(8, 6))
    k_gt, E_gt = compute_classical_energy_spectrum(gt_traj.numpy())
    for i in range(n_blocks):
        start, end = i * block_size, min((i + 1) * block_size, nt)
        k, E = compute_classical_energy_spectrum(traj[start:end].numpy())
        ax.loglog(k[1:], E[1:], color=colors[i], alpha=0.6, lw=1.2,
                  label=f"t={start}-{end}" if i in (0, n_blocks - 1) else None)
    ax.loglog(k_gt[1:], E_gt[1:], color="black", lw=3, label="GT (moyenne totale)")
    ax.set_xlabel("Nombre d'onde k")
    ax.set_ylabel("E(k)")
    ax.set_title(f"Spectre d'energie au cours du temps — {name}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_correction_count_single(name, tracker, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cumulative = np.cumsum(tracker["correction_flags"])
    ax.plot(cumulative, color="tab:blue", lw=2)
    ax.set_xlabel("Pas de rollout (temps simule)")
    ax.set_ylabel("Nombre cumule de corrections")
    ax.set_title(f"Corrections declenchees au cours du temps — {name}")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_noise_level_distribution_single(name, tracker, s_init, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(tracker["noise_levels"], bins=30, color="tab:blue", alpha=0.7)
    ax.axvline(s_init, color="k", ls="--", label=f"seuil de correction (s_init={s_init})")
    ax.set_xlabel("Niveau de bruit estime par le classifieur")
    ax.set_ylabel("Nombre de pas")
    ax.set_title(f"Distribution des niveaux de bruit detectes — {name}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_time_delay_embedding_single(name, D_no_corr, D_with_corr, background_xy, out_path,
                                     tau=8, n_delays=2):
    """
    background_xy : (Xp, Yp) -- nuage de fond (attracteur "vrai", cf.
    build_background_attractor). Superpose les trajectoires du modele
    (sans/avec correction) embedees en delai temporel sur leur propre
    dissipation -- teste si le modele reste sur le bon attracteur
    statistiquement, independamment de toute correspondance ponctuelle
    avec le GT (qui n'a pas de sens au-dela de l'horizon de predictibilite).
    """
    Xp, Yp = background_xy
    phase_space = np.vstack([Xp, Yp])
    kde = gaussian_kde(phase_space)
    density = kde(phase_space)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(Xp, Yp, c=density, s=4, cmap="plasma", linewidths=0, alpha=0.6)

    for D, color, ls, label in [(D_no_corr, "tab:red", "--", "sans correction"),
                                (D_with_corr, "tab:blue", "-", "avec correction")]:
        embed = time_delay_embedding(D, tau=tau, n_delays=n_delays)
        mask = np.isfinite(embed).all(axis=1)
        embed = embed[mask]
        ax.plot(embed[:, 0], embed[:, 1], color=color, ls=ls, alpha=0.8, linewidth=0.7, label=label)

    ax.set_xlabel(r"$\epsilon_t$", fontsize=16)
    ax.set_ylabel(r"$\epsilon_{t-\Delta t}$", fontsize=16)
    ax.set_title(f"Embedding en delai temporel (dissipation) — {name}")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_vorticity_snapshots(name, gt_traj, traj_no_corr, traj_with_corr, out_path,
                             times=(0, 1, 2, 3, 4)):
    n_steps = min(traj_no_corr.shape[0], gt_traj.shape[0]) - 1
    snap_times = [int(f * n_steps) for f in np.linspace(0, 1, len(times))]

    vort_gt = vorticity(gt_traj.numpy())
    vort_no = vorticity(traj_no_corr.numpy())
    vort_with = vorticity(traj_with_corr.numpy())
    vmax = float(np.abs(vort_gt).max())

    n_rows = len(snap_times)
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["GT", "Sans correction", "Avec correction"]
    for row, t in enumerate(snap_times):
        for col, field in enumerate([vort_gt[t], vort_no[t], vort_with[t]]):
            ax = axes[row, col]
            ax.imshow(field, cmap="bwr", vmin=-vmax, vmax=vmax, origin="lower")
            ax.axis("off")
            if row == 0:
                ax.set_title(col_titles[col], fontsize=10)
            if col == 0:
                ax.text(-0.15, 0.5, f"t={t}", transform=ax.transAxes,
                       fontsize=9, va="center", ha="right")

    fig.suptitle(f"Vorticite au cours du temps — {name}", fontsize=12, y=1.0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.environ.get("DATA_DIR", "../data"))
    parser.add_argument("--runs_dir", default=os.environ.get("RUNS_DIR", "../runs_mesu/Re90/Re90"))
    parser.add_argument("--exp_dir", default="kolmogorov/Re90")
    parser.add_argument("--sim_file", default="sim8.h5")
    parser.add_argument("--rollout", type=int, default=2000)
    parser.add_argument("--s_init", type=int, default=7)
    parser.add_argument("--s_stop", type=int, default=3)
    parser.add_argument("--out_dir", default="correction_eval_results")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--wnoarch_dir", default=os.environ.get("WNOARCH_DIR", "../../WNO_arch"),
                        help="Racine du repo WNO_arch (pour le combo fno_state, run emul_seq_fno_k1)")
    parser.add_argument("--wnoarch_runs_dir", default=None,
                        help="Dossier des runs WNO_arch (defaut: <wnoarch_dir>/runs/kolmogorov/Re90)")
    parser.add_argument("--fno_state_run", default=DEFAULT_WNOARCH_FNO_STATE_RUN)
    parser.add_argument("--skip_fno_state", action="store_true",
                        help="Ne pas inclure le combo fno_state (WNO_arch/emul_seq_fno_k1)")
    parser.add_argument("--skip_k1_defaults", action="store_true",
                        help="Ne pas evaluer les combos k=1 par defaut (fno_delta, wno_delta, "
                        "wno_state, fno_state) -- utile quand ils sont deja evalues/en cache "
                        "(cf. ../results/correction_eval_results_*) et qu'on veut juste ajouter de nouveaux "
                        "runs via --wnoarch_seq_runs sans tout relancer.")
    parser.add_argument("--wnoarch_seq_runs", nargs="*", default=[],
                        help="Noms de runs WNO_arch supplementaires a evaluer (sous "
                        "--wnoarch_runs_dir), ex: emul_seq_fno_k5 emul_seq_wno_k5 "
                        "emul_seq_fno_k5_delta emul_seq_wno_k5_delta. k et prediction_mode "
                        "sont lus depuis le config.yaml de chaque run (pas besoin de les "
                        "preciser ici).")
    parser.add_argument("--seed", type=int, default=101,
                        help="Seed torch pour le bruit tau*randn du rollout (jamais fixee avant : "
                        "chaque execution tirait un bruit different, rendant les runs non reproductibles "
                        "et masquant parfois la divergence de fno_delta -- sur 5 tirages testes, 101-104 "
                        "divergent (KE -> inf), seul 100 restait borne. 101 est donc le defaut pour "
                        "montrer la divergence sans correction de facon fiable plutot que par chance.")
    parser.add_argument("--divergence_factor", type=float, default=100.0,
                        help="Arrete le rollout d'une trajectoire (avec ou sans correction) des que son "
                        "energie cinetique depasse ce facteur x l'energie initiale (apres correction "
                        "eventuelle, qui garde donc sa chance) -- evite de perdre du temps a corriger "
                        "des milliers de pas sur une trajectoire deja irrecuperable. 0 ou negatif desactive.")
    parser.add_argument("--re", type=float, default=None,
                        help="Reynolds de la trajectoire evaluee (--exp_dir/--sim_file), transmis au "
                        "corrector diffusion. Requis si le run charge par load_diffusion() est "
                        "Re-conditionne (config.yaml avec re_values, cf. FNO2D_classifier_concat) -- "
                        "sans quoi maybe_correct() plantera en essayant d'appeler le modele sans re. "
                        "None (defaut) = comportement inchange pour un corrector mono-Re.")
    parser.add_argument("--diffusion_run_dir", default=None,
                        help="Chemin complet vers un run de diffusion, en remplacement de "
                        "<runs_dir>/<DEFAULT_DIFFUSION_RUN> -- necessaire pour un run Re-conditionne "
                        "(ex: diffusion_concat_re50-90_2re), qui vit sous kolmogorov_parametric/ et "
                        "non sous runs_dir (kolmogorov/Re90/ par defaut). None (defaut) = comportement "
                        "inchange.")
    args = parser.parse_args()
    divergence_factor = args.divergence_factor if args.divergence_factor and args.divergence_factor > 0 else None
    re_tensor = torch.tensor([args.re], dtype=torch.float32) if args.re is not None else None

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
    diffusion_run_dir = args.diffusion_run_dir or os.path.join(args.runs_dir, DEFAULT_DIFFUSION_RUN)
    diffusion, diff_cfg = load_diffusion(diffusion_run_dir, device)

    emulators = {}
    wnoarch_runs_dir = args.wnoarch_runs_dir or os.path.join(args.wnoarch_dir, "runs", "kolmogorov", "Re90")

    if not args.skip_k1_defaults:
        for name, run_name in DEFAULT_RUNS.items():
            run_dir = os.path.join(args.runs_dir, run_name)
            print(f"Chargement de l'emulator {name} ({run_name})...", flush=True)
            if name == "fno_delta":
                model, cfg = load_emulator_fno(run_dir, device)
            else:
                model, cfg = load_emulator_wno(run_dir, device)
            emulators[name] = {"model": model, "cfg": cfg, "prediction_mode": PREDICTION_MODE[name], "k": 1}

        if not args.skip_fno_state:
            run_dir = os.path.join(wnoarch_runs_dir, args.fno_state_run)
            print(f"Chargement de l'emulator fno_state ({run_dir})...", flush=True)
            model, cfg = load_emulator_wnoarch(run_dir, args.wnoarch_dir, device)
            emulators["fno_state"] = {"model": model, "cfg": cfg, "prediction_mode": PREDICTION_MODE["fno_state"], "k": 1}
    else:
        print("--skip_k1_defaults : fno_delta/wno_delta/wno_state/fno_state non evalues "
              "(deja en cache).", flush=True)

    # Runs WNO_arch supplementaires (ex: emul_seq_fno_k5, emul_seq_fno_k5_delta, ...) --
    # k et prediction_mode lus depuis le config.yaml de chaque run, pas besoin de les
    # renseigner a la main comme pour DEFAULT_RUNS.
    for run_name in args.wnoarch_seq_runs:
        run_dir = os.path.join(wnoarch_runs_dir, run_name)
        print(f"Chargement de l'emulator {run_name} (WNO_arch)...", flush=True)
        model, cfg = load_emulator_wnoarch(run_dir, args.wnoarch_dir, device)
        k = cfg["k"]
        prediction_mode = cfg.get("prediction_mode", "state")
        print(f"  k={k}  prediction_mode={prediction_mode}", flush=True)
        emulators[run_name] = {"model": model, "cfg": cfg, "prediction_mode": prediction_mode, "k": k}

    if not emulators:
        print("Aucun emulator a evaluer (verifier --skip_k1_defaults / --wnoarch_seq_runs).", flush=True)
        return

    # ── donnees GT ─────────────────────────────────────────────────────────────
    # tous les combos partent du meme instant physique : on charge assez de contexte
    # pour le plus grand k (les modeles k=1 se contentent des dernieres frames de ce contexte)
    #
    # Le rollout libre n'a besoin du GT que pour la toute premiere frame -- le
    # comparer a une trajectoire GT plus courte (ex: sim8.h5 ne fait que 4000
    # frames) n'a donc pas a plafonner la longueur du rollout lui-meme. gt_full
    # garde sa longueur naturelle ; seules les comparaisons directes a GT
    # (error curve) sont tronquees a ce qui est disponible.
    ds = diff_cfg.get("ds", 2)
    max_k = max(info["k"] for info in emulators.values())
    n_steps = args.rollout
    gt_full = load_gt_trajectory(args.data_dir, args.exp_dir, args.sim_file, ds=ds)
    n_gt_steps = gt_full.shape[0] - max_k
    if n_steps > n_gt_steps:
        print(f"Attention : rollout demande ({n_steps}) > longueur GT disponible ({n_gt_steps} pas). "
              f"Le rollout ira jusqu'a {n_steps} pas ; les comparaisons directes a GT (erreur L2) "
              f"seront tronquees a {n_gt_steps} pas, les autres diagnostics (energie cinetique, "
              f"spectre, corrections) couvriront le rollout complet.", flush=True)
    # trajectoire de reference commune a tous les combos (meme instant de depart = max_k-1)
    gt_traj = gt_full[max_k - 1:]
    print(f"Trajectoire GT chargee : {gt_full.shape}, rollout={n_steps} pas, contexte max k={max_k}", flush=True)
    gt_ke = kinetic_energy_curve(gt_traj)

    print("Construction du nuage de fond (attracteur, dissipation des trajectoires d'entrainement)...", flush=True)
    background_xy = build_background_attractor(args.data_dir, args.exp_dir, ds)
    print(f"  Nuage de fond : {background_xy[0].shape[0]} points", flush=True)

    # ── rollouts ───────────────────────────────────────────────────────────────
    results = {}

    for name, info in emulators.items():
        print(f"\n=== {name} (k={info['k']}) ===", flush=True)
        tau = info["cfg"].get("tau", 1e-5)
        k = info["k"]

        if k == 1:
            x0 = gt_full[max_k - 1: max_k].to(device)
            print("  Rollout sans correction...", flush=True)
            traj_no_corr, tracker_no_corr = rollout(
                info["model"], info["prediction_mode"], x0, n_steps, tau, device, diffusion=None,
                divergence_factor=divergence_factor,
            )
            print("  Rollout avec correction...", flush=True)
            traj_with_corr, tracker_with_corr = rollout(
                info["model"], info["prediction_mode"], x0, n_steps, tau, device,
                diffusion=diffusion, s_init=args.s_init, s_stop=args.s_stop,
                divergence_factor=divergence_factor, re=re_tensor,
            )
        else:
            n_channels = info["cfg"].get("n_channels", gt_full.shape[-1])
            gt_context = gt_full[max_k - k: max_k]
            print("  Rollout sans correction...", flush=True)
            traj_no_corr, tracker_no_corr = rollout_windowed(
                info["model"], k, n_channels, gt_context, n_steps, device,
                prediction_mode=info["prediction_mode"], diffusion=None,
                divergence_factor=divergence_factor,
            )
            print("  Rollout avec correction...", flush=True)
            traj_with_corr, tracker_with_corr = rollout_windowed(
                info["model"], k, n_channels, gt_context, n_steps, device,
                prediction_mode=info["prediction_mode"],
                diffusion=diffusion, s_init=args.s_init, s_stop=args.s_stop,
                divergence_factor=divergence_factor, re=re_tensor,
            )
        print(f"  Corrections declenchees : {tracker_with_corr['n_corrections']} / {tracker_with_corr['stopped_early_at'] or n_steps}", flush=True)
        if tracker_no_corr["stopped_early_at"]:
            print(f"  Arret anticipe (sans correction) au pas {tracker_no_corr['stopped_early_at']}/{n_steps} "
                  f"(divergence x{divergence_factor:g})", flush=True)
        if tracker_with_corr["stopped_early_at"]:
            print(f"  Arret anticipe (avec correction) au pas {tracker_with_corr['stopped_early_at']}/{n_steps} "
                  f"(divergence x{divergence_factor:g})", flush=True)

        # error curve : tronquee a la longueur GT dispo si le rollout va plus loin
        common_len = min(traj_no_corr.shape[0], gt_traj.shape[0])
        error_no_corr = relative_l2_curve(traj_no_corr[:common_len], gt_traj[:common_len])
        error_with_corr = relative_l2_curve(traj_with_corr[:common_len], gt_traj[:common_len])
        ke_no_corr = kinetic_energy_curve(traj_no_corr)
        ke_with_corr = kinetic_energy_curve(traj_with_corr)
        diss_no_corr = compute_dissipation_curve(traj_no_corr)
        diss_with_corr = compute_dissipation_curve(traj_with_corr)

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
        plot_time_delay_embedding_single(
            name, diss_no_corr, diss_with_corr, background_xy,
            os.path.join(args.out_dir, f"embedding_{name}.png"),
        )

        np.savez(
            os.path.join(args.out_dir, f"{name}_curves.npz"),
            error_no_corr=error_no_corr, error_with_corr=error_with_corr,
            ke_no_corr=ke_no_corr, ke_with_corr=ke_with_corr,
            diss_no_corr=diss_no_corr, diss_with_corr=diss_with_corr,
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
