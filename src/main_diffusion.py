from training.diffusion_training import DiffusionTraining
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
    with open("config_command_diff.yaml", "r") as f:
        args = yaml.safe_load(f)
    
    set_seed(args["seed"])

    experiment = DiffusionTraining(args)
    experiment.execute_experience()  

        
