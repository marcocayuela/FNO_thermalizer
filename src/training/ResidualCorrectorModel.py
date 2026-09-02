import torch
import torch.nn as nn


class ResidualCorrector(nn.Module):
    """Thin wrapper around an FNO2D_classifier-shaped backbone (fno/fno_2D_classifier.py),
    trained to predict the RESIDUAL (true_state - corrupted_state) directly
    from a real emulator-rollout state, plus (via the backbone's existing
    classifier head) which rollout step the state looks like it's at -- NOT
    a real diffusion process (cf. training/DiffusionModel.py::Diffusion,
    kept for the synthetic-noise correctors elsewhere in this project).

    Why not reuse Diffusion: DDPM's iterative reverse-sampling loop only
    makes sense for a corruption process that is known, stochastic and
    invertible (Gaussian noise added according to a fixed schedule). A
    rollout's own drift is deterministic and has no such closed form, so
    there is nothing principled for an iterative reverse process to invert
    -- recalibrating Diffusion's "noise level" to mean "rollout step"
    without also changing its forward/reverse math would just be cosmetic.
    This class instead does the correction in a single forward pass,
    directly supervised by (corrupted, true) pairs from
    training/generate_euler_correction_data.py -- no noise schedule, no
    alpha/beta buffers, no forward/reverse diffusion methods.

    n_cat on the wrapped backbone still means "how many discrete classes
    the classifier head predicts", now interpreted as a rollout-step bucket
    (cf. training/trainer_residual_corrector.py) instead of a diffusion
    timestep -- everything else about FNO2D_classifier's forward() is
    unchanged and reused as-is.
    """

    def __init__(self, model, max_step):
        super().__init__()
        self.model = model
        self.max_step = max_step

    def forward(self, x, predict_class=True):
        # Straight passthrough to the backbone -- (residual_pred, step_logits)
        # when predict_class=True, matching FNO2D_classifier's own return
        # convention exactly (cf. fno/fno_2D_classifier.py::forward).
        return self.model(x, predict_class=predict_class)

    def correct(self, x):
        """Single-shot correction: x + predicted residual. No iterative
        loop (cf. class docstring) -- this IS the full correction."""
        residual, _ = self.model(x, predict_class=True)
        return x + residual

    def estimate_step(self, x):
        """Returns the backbone's own estimate of which rollout-step bucket
        x looks like it's at (argmax over the classifier head), used by
        evaluation/correction_eval_euler.py::maybe_correct_residual to
        decide whether a state needs correcting at all (mirrors
        maybe_correct's s_init threshold)."""
        _, step_logits = self.model(x, predict_class=True)
        return torch.softmax(step_logits, dim=-1).argmax(dim=-1)
