from training.unet_training import UNetTraining
import torch
import numpy as np
import random
import yaml
import os
import sys


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

    if not os.path.exists("../runs"):
        os.makedirs("../runs")

    # config_command_emul_unet_delta.yaml ou config_command_emul_unet_state.yaml
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config_command_emul_unet_delta.yaml"
    with open(config_path, "r") as f:
        args = yaml.safe_load(f)

    set_seed(args["seed"])

    experiment = UNetTraining(args)
    experiment.execute_experience()
