import torch
import math
import torch.nn.functional as F

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

class Factory():

    OPTIMIZERS = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
    SCHEDULERS = {"one_cycle_lr": torch.optim.lr_scheduler.OneCycleLR, "cosine_annealing": torch.optim.lr_scheduler.CosineAnnealingLR}

    METRICS = {"mse": torch.nn.MSELoss(),
               "rmse": lambda y_pred, y_true: torch.sqrt(torch.mean((y_pred - y_true)**2)),
               "mae": lambda y_pred, y_true: torch.mean(torch.abs(y_pred - y_true)),
               "relative_rmse": relative_rmse,
               "relative_mae": relative_mae,
               "cross_entropy": lambda x,y: F.cross_entropy(x,y),
               "mse_phys_and_fourier": lambda y_pred, y_true: torch.mean((y_pred - y_true)**2) + torch.mean(torch.log(1 + torch.mean((torch.fft.fft2(y_pred) - torch.fft.fft2(y_true))**2)))
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
    