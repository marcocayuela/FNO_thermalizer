import torch
import torch.nn as nn


class ResidualCorrector(nn.Module):
    """Thin wrapper around an FNO2D_classifier-shaped backbone (fno/fno_2D_classifier.py),
    trained to predict the RESIDUAL (true_state - corrupted_state) directly
    from a real emulator-rollout state, plus (via the backbone's existing
    classifier head) a discretized bucket of how corrupted that state
    actually is -- NOT a real diffusion process (cf.
    training/DiffusionModel.py::Diffusion, kept for the synthetic-noise
    correctors elsewhere in this project).

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
    the classifier head predicts" -- NOT a rollout-step bucket (an earlier
    version of this design used raw step index, but that isn't a
    content-independent quantity: one trajectory can be badly drifted by
    step 50 while another is still clean at step 90, so the same step
    label would correspond to very different actual corruption levels
    across trajectories -- an ill-posed target). Instead it's a bucket of
    the ACTUAL measured relative error between corrupted and true state
    (cf. training/euler_dataset.py::LazyEulerCorrectionDataset's
    error_bin_edges), the direct analogue of a real diffusion timestep:
    a trajectory-independent measure of how corrupted a state is.
    """

    def __init__(self, model, n_bins):
        super().__init__()
        self.model = model
        self.n_bins = n_bins

    def forward(self, x, predict_class=True):
        # Straight passthrough to the backbone -- (residual_pred, bin_logits)
        # when predict_class=True, matching FNO2D_classifier's own return
        # convention exactly (cf. fno/fno_2D_classifier.py::forward).
        return self.model(x, predict_class=predict_class)

    def correct(self, x):
        """Single-shot correction: x + predicted residual. No iterative
        loop (cf. class docstring) -- this IS the full correction."""
        residual, _ = self.model(x, predict_class=True)
        return x + residual

    def estimate_error_bin(self, x):
        """Returns the backbone's own estimate of which corruption-level
        bucket x looks like it's in (argmax over the classifier head), used
        by evaluation/correction_eval.py::maybe_correct's ResidualCorrector
        branch to decide whether a state needs correcting at all (mirrors
        maybe_correct's s_init threshold, now in error-bin units)."""
        _, bin_logits = self.model(x, predict_class=True)
        return torch.softmax(bin_logits, dim=-1).argmax(dim=-1)
