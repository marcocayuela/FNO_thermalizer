"""
Trains one FNO1D per nu to identify the KS right-hand side operator
N(u) = Du/Dt (cf. training/ks_operator_dataset.py's own docstring for the
full rationale) -- reproduces Operator_Identification.ipynb's per-nu loop,
using thermalizer's own already-verified FNO1D (fno/fno_1D.py) instead of
that notebook's standalone FnoLibrary.Fourier_layer1D (same underlying
IntegralKernel1D math, just not duplicated here).

Not training.trainer.Trainer (built for an autoregressive emulator's
unrolled-sequence loss) nor training.fno_training_1d_hyper.TrainerHyperEmulator1D
(needs an extra nu argument threaded into a hyper-conditioned model): this is
the simplest case of the three, a single forward pass per batch against a
directly-supplied target, no rollout and no per-sample nu conditioning
(there IS one nu, but it's fixed for the whole model -- a separate model per
nu, not one model conditioned on it).
"""

import os
import time

import torch
import torch.nn as nn
import yaml
from tabulate import tabulate
from tqdm import tqdm

from fno.fno_1D import FNO1D
from training.ks_operator_dataset import build_operator_loaders
from training.factory import Factory, EarlyStopping
from training.metric_logger import MetricLogger

DATA_DIR = os.getenv("DATA_DIR", "../data")
LOG_DIR = os.getenv("LOG_DIR", "../runs")


class OperatorTrainerFNO1D():

    def __init__(self, model, train_loader, test_loader, loss_fn, optimizer, scheduler,
                num_epochs, device, exp_dir, exp_name, start_epoch=1, patience=None, min_delta=1e-8):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.device = device
        self.exp_dir = exp_dir
        self.exp_name = exp_name
        self.start_epoch = start_epoch
        # Same convention as training.trainer.Trainer: only armed if a
        # patience is actually given, tracked on test loss.
        self.early_stopping = EarlyStopping(patience=patience, min_delta=min_delta) if patience else None
        self.current_epoch = start_epoch

    def _relative_l2(self, pred, target):
        return (torch.mean((pred - target) ** 2) / torch.mean(target ** 2)) ** 0.5 * 100

    def _run_batches(self, loader, train):
        total_loss = 0.0
        total_rel_l2 = 0.0
        batch_iter = tqdm(loader, desc=f"Epoch {self.current_epoch}", leave=False, ncols=90) if train else loader
        for u, dudt in batch_iter:
            u = u.to(self.device).float().unsqueeze(-1)       # (B, Mx, 1)
            dudt = dudt.to(self.device).float().unsqueeze(-1)  # (B, Mx, 1)

            if train:
                self.optimizer.zero_grad()

            pred = self.model(u)
            loss = self.loss_fn(pred, dudt)

            if train:
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            total_rel_l2 += self._relative_l2(pred, dudt).item()

        n = len(loader)
        return total_loss / n, total_rel_l2 / n

    def train_epoch(self):
        self.model.train()
        train_loss, train_rel_l2 = self._run_batches(self.train_loader, train=True)
        if self.scheduler:
            self.scheduler.step()

        self.model.eval()
        with torch.no_grad():
            test_loss, test_rel_l2 = self._run_batches(self.test_loader, train=False)

        return train_loss, train_rel_l2, test_loss, test_rel_l2

    def train_loop(self):
        headers = ["Epoch", "Tr loss", "Tr rel_l2(%)", "Te loss", "Te rel_l2(%)", "LR", "Time(s)"]
        csv_path = os.path.join(self.exp_dir, self.exp_name, "logs", "metrics.csv")
        logger = MetricLogger(csv_path, headers, resume=self.start_epoch > 1)

        min_train_loss = 1e18
        min_test_loss = 1e18
        model_dir = os.path.join(LOG_DIR, self.exp_dir, self.exp_name, "model_weights")

        for epoch in range(self.start_epoch, self.num_epochs + 1):
            self.current_epoch = epoch
            t0 = time.perf_counter()
            train_loss, train_rel_l2, test_loss, test_rel_l2 = self.train_epoch()
            elapsed = time.perf_counter() - t0
            current_lr = self.optimizer.param_groups[0]["lr"]

            row = [epoch, train_loss, train_rel_l2, test_loss, test_rel_l2, current_lr, elapsed]
            print(tabulate([row], headers=headers, floatfmt=".5g"))
            logger.log(dict(zip(headers, row)))

            if test_loss < min_test_loss:
                min_test_loss = test_loss
                torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(),
                           "optimizer_state_dict": self.optimizer.state_dict()},
                          os.path.join(model_dir, "min_test_loss.pth"))
                print(f"Best model saved at epoch {epoch} with test loss: {test_loss:.6f}")

            if train_loss < min_train_loss:
                min_train_loss = train_loss
                torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(),
                           "optimizer_state_dict": self.optimizer.state_dict()},
                          os.path.join(model_dir, "min_train_loss.pth"))

            if self.early_stopping and self.early_stopping.step(test_loss):
                print(f"Early stopping at epoch {epoch} (no improvement since {self.early_stopping.patience} epochs)")
                break

        torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(),
                   "optimizer_state_dict": self.optimizer.state_dict()},
                  os.path.join(model_dir, "final_model.pth"))


class OperatorTrainingKS():
    """Loops over nu_values, training one independent FNO1D operator-
    identification model per nu (cf. module docstring) -- exp_name gets a
    "_nu<X>" suffix per model, so all 13 (or however many) runs live under
    the same exp_dir without overwriting each other."""

    def __init__(self, args):
        self.args = args

        device_asked = self.args.get("device", "auto")
        if device_asked in ["cuda", "auto"] and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif device_asked in ["mps", "auto"] and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Device used: {self.device}")

        self.exp_dir = self.args["exp_dir"]
        self.exp_name = self.args["exp_name"]
        self.nu_values = self.args["nu_values"]

        self.k_max = self.args["k_max"]
        self.width = self.args["width"]
        self.n_fourier_layer = self.args.get("n_fourier_layer", 3)
        self.l = self.args.get("l", 1)
        self.hidden_proj = self.args.get("hidden_proj", 32)

        self.batch_size = self.args.get("batch_size", 20)
        self.num_workers = self.args.get("num_workers", 2)
        self.ds = self.args.get("ds", 1)
        self.train_frac = self.args.get("train_frac", 0.7)
        self.test_frac = self.args.get("test_frac", 0.3)

        self.num_epochs = self.args.get("num_epochs", 250)
        self.optimizer_info = self.args["optimizer"]
        self.scheduler_info = self.args.get("scheduler")
        self.loss_name = self.args.get("loss_fn", "l1")
        self.patience = self.args.get("patience", None)
        self.min_delta = self.args.get("min_delta", 1e-8)

    def _loss_fn(self):
        return nn.L1Loss() if self.loss_name == "l1" else nn.MSELoss()

    def make_directories(self, exp_name_nu):
        directories = [os.path.join(LOG_DIR, self.exp_dir),
                       os.path.join(LOG_DIR, self.exp_dir, exp_name_nu),
                       os.path.join(LOG_DIR, self.exp_dir, exp_name_nu, "model_weights"),
                       os.path.join(LOG_DIR, self.exp_dir, exp_name_nu, "logs")]
        for d in directories:
            os.makedirs(d, exist_ok=True)

        save_path = os.path.join(LOG_DIR, self.exp_dir, exp_name_nu, "config.yaml")
        with open(save_path, "w") as f:
            yaml.safe_dump(self.args, f)

    def execute_experience(self):
        for nu in self.nu_values:
            exp_name_nu = f"{self.exp_name}_nu{str(nu).replace('.', 'p')}"
            print(f"\n===== Training operator-identification FNO1D for nu={nu} ({exp_name_nu}) =====\n")
            self.make_directories(exp_name_nu)

            train_loader, test_loader, n_train, n_test = build_operator_loaders(
                DATA_DIR, self.exp_dir, nu, batch_size=self.batch_size, num_workers=self.num_workers,
                ds=self.ds, train_frac=self.train_frac, test_frac=self.test_frac,
            )
            print(f"Training samples: {n_train}, Testing samples: {n_test}")

            model = FNO1D(input_dim=1, output_dim=1, modes=self.k_max, width=self.width, l=self.l,
                          n_layer=self.n_fourier_layer, hidden_proj=self.hidden_proj, device=self.device)
            model = model.to(self.device).float()

            param_dict = model.count_parameters_per_module()
            print(f"Params: {param_dict['total']:,}")

            optimizer = Factory.get_optimizer(self.optimizer_info["type"], model.parameters(),
                                              lr=self.optimizer_info["lr"])
            scheduler = None
            if self.scheduler_info:
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=self.scheduler_info["step_size"], gamma=self.scheduler_info["gamma"])

            trainer = OperatorTrainerFNO1D(
                model=model, train_loader=train_loader, test_loader=test_loader,
                loss_fn=self._loss_fn(), optimizer=optimizer, scheduler=scheduler,
                num_epochs=self.num_epochs, device=self.device,
                exp_dir=self.exp_dir, exp_name=exp_name_nu,
                patience=self.patience, min_delta=self.min_delta,
            )
            trainer.train_loop()
