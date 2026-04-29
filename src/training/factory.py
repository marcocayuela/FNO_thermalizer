import torch
import math
import torch.nn.functional as F
from utils import routines

def relative_rmse(y_pred, y_true, eps=1e-8):
    """
    Computes Relative Root Mean Squared Error.
    Args:
        y_pred: Tensor of predictions
        y_true: Tensor of ground truth
        eps: small value to avoid division by zero
    Returns:
        scalar tensor
    """
    rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2))
    norm = torch.sqrt(torch.mean(y_true ** 2)) + eps
    return rmse / norm

def relative_mae(y_pred, y_true, eps=1e-8):
    """
    Computes Relative Root Mean Squared Error.
    Args:
        y_pred: Tensor of predictions
        y_true: Tensor of ground truth
        eps: small value to avoid division by zero
    Returns:
        scalar tensor
    """
    mae = torch.mean(torch.abs(y_pred - y_true))
    norm = torch.mean(torch.abs(y_true)) + eps
    return mae / norm


def spectral_loss(y_pred, y_true, eps=1e-8):
    E_pred, _ = routines.energy_spectrum(y_pred)
    E_true, _ = routines.energy_spectrum(y_true)

    loss = torch.mean(((E_pred - E_true) / (E_true + eps))**2)
    return loss

def mse_phys_and_spectral(y_pred, y_true, lambda_spec=0.1):
    mse = torch.mean((y_pred - y_true)**2)
    spec = spectral_loss(y_pred, y_true)
    return mse + lambda_spec * spec

class Factory():

    OPTIMIZERS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
    SCHEDULERS = {"one_cycle_lr": torch.optim.lr_scheduler.OneCycleLR, "cosine_annealing": torch.optim.lr_scheduler.CosineAnnealingLR}

    METRICS = {"mse": torch.nn.MSELoss(),
               "rmse": lambda y_pred, y_true: torch.sqrt(torch.mean((y_pred - y_true)**2)),
               "mae": lambda y_pred, y_true: torch.mean(torch.abs(y_pred - y_true)),
               "relative_rmse": relative_rmse,
               "relative_mae": relative_mae,
               "cross_entropy": lambda x,y: F.cross_entropy(x,y),
               "mse_phys_and_fourier": lambda y_pred, y_true: mse_phys_and_spectral(y_pred, y_true, lambda_spec=0.01)
               }
    
    @staticmethod
    def get_optimizer(name, params, **kwargs):
        return Factory.OPTIMIZERS[name](params, **kwargs)
    
    @staticmethod
    def get_scheduler(scheduler_info, optimizer, num_epochs, n_batch):
        name = scheduler_info["name"]
        params = {k: v for k, v in scheduler_info.items() if k != "name"}
        if name == "one_cycle_lr":
            params["epochs"] = num_epochs
            params["steps_per_epoch"] = n_batch
        if name == "cosine_annealing":
            params["T_max"] = num_epochs

        return Factory.SCHEDULERS[name](optimizer, **params)

    @staticmethod
    def get_metric(name):
        return Factory.METRICS[name]
    