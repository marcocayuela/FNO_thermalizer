from training.diffusion_training_parametric import DiffusionTrainingParametric
import argparse
import torch
import numpy as np
import random
import yaml
import os


def set_seed(seed: int):
    """Fix all random seeds for reproducibility, including MPS backend."""
    random.seed(seed)                     # Python random
    np.random.seed(seed)                  # NumPy
    torch.manual_seed(seed)               # PyTorch CPU
    torch.cuda.manual_seed(seed)          # PyTorch GPU
    torch.cuda.manual_seed_all(seed)      # Multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # MPS (Apple Silicon / Metal) backend
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.manual_seed(seed)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_command_diff_concat.yaml",
                        help="Path to a config_command_diff_concat*.yaml (e.g. a reduced-Re "
                        "variant like config_command_diff_concat_2re.yaml)")
    cli_args = parser.parse_args()

    if not os.path.exists("../runs"):
        os.makedirs("../runs")
    with open(cli_args.config, "r") as f:
        args = yaml.safe_load(f)

    set_seed(args["seed"])

    experiment = DiffusionTrainingParametric(args)
    experiment.execute_experience()
