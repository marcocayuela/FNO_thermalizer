"""
Builds a Re-conditioned diffusion backbone from a config dict. Mirrors
Parameterized_Neural_Operator/models/model_factory.py's philosophy: one
factory dispatching on a "model_type" string, instead of one
DiffusionTraining subclass per model (which would mostly duplicate the same
dataset/trainer/optimizer wiring already in
diffusion_training_parametric.py::DiffusionTrainingParametric).

model_type:
    "fno_concat"  : FNO2D_classifier_concat  -- Re as an extra input channel
    "fno_hyper"   : FNO2D_classifier_hyper   -- Re-conditioned spectral weight R
                    (basis mixture, cf. Parameterized_Neural_Operator's
                    pfno_hyper_2D.py -- the most long-rollout-robust
                    conditioning mechanism in that project's own ablation study)
    "unet_film"   : UNet2D_FiLM_classifier   -- FiLM at every resolution level
    "unet_concat" : UNet2D_concat_classifier -- Re as an extra input channel
"""

from fno.fno_2D_classifier_concat import FNO2D_classifier_concat
from fno.fno_2D_classifier_hyper import FNO2D_classifier_hyper
from unet.unet_2D_film_classifier import UNet2D_FiLM_classifier
from unet.unet_2D_concat_classifier import UNet2D_concat_classifier


def build_diffusion_backbone(config, param_mean, param_std, device):
    model_type = config["model_type"]

    if model_type == "fno_concat":
        return FNO2D_classifier_concat(
            input_dim=config["input_dim"], output_dim=config["output_dim"],
            modes_x=config["k_max_x"], modes_y=config["k_max_y"], width=config["width"], l=config["l"],
            n_layer=config["n_fourier_layer"], hidden_proj=config.get("hidden_proj"),
            mlp=config.get("mlp", True), layers_mlp=config.get("layers_mlp"),
            class_mlp_layers=config.get("class_mlp_layers"), n_cat=config["timesteps"],
            param_mean=param_mean, param_std=param_std, device=device,
        )

    if model_type == "fno_hyper":
        return FNO2D_classifier_hyper(
            input_dim=config["input_dim"], output_dim=config["output_dim"],
            modes_x=config["k_max_x"], modes_y=config["k_max_y"], width=config["width"], l=config["l"],
            n_layer=config["n_fourier_layer"], hidden_proj=config.get("hidden_proj"),
            mlp=config.get("mlp", True), layers_mlp=config.get("layers_mlp"),
            param_embed_dim=config.get("param_embed_dim", 32),
            param_hidden_dim=config.get("param_hidden_dim", 64),
            param_encoder_layers=config.get("param_encoder_layers", 2),
            n_basis=config.get("n_basis", 4), class_mlp_layers=config.get("class_mlp_layers"),
            n_cat=config["timesteps"], param_mean=param_mean, param_std=param_std, device=device,
        )

    if model_type == "unet_film":
        return UNet2D_FiLM_classifier(
            input_dim=config["input_dim"], output_dim=config["output_dim"],
            depth=config["depth"], base_width=config["base_width"],
            param_embed_dim=config.get("param_embed_dim", 32),
            param_hidden_dim=config.get("param_hidden_dim", 64),
            param_encoder_layers=config.get("param_encoder_layers", 2),
            class_mlp_layers=config.get("class_mlp_layers"), n_cat=config["timesteps"],
            param_mean=param_mean, param_std=param_std, device=device,
        )

    if model_type == "unet_concat":
        return UNet2D_concat_classifier(
            input_dim=config["input_dim"], output_dim=config["output_dim"],
            depth=config["depth"], base_width=config["base_width"],
            class_mlp_layers=config.get("class_mlp_layers"), n_cat=config["timesteps"],
            param_mean=param_mean, param_std=param_std, device=device,
        )

    raise ValueError(
        f"Unknown model_type: '{model_type}'. Choices: fno_concat, fno_hyper, unet_film, unet_concat"
    )
