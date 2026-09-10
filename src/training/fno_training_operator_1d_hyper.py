"""
Hyper-nu counterpart of fno_training_operator_1d.py -- instead of one
independent FNO1D per nu (later combined post-hoc via a linear-regression
ensemble, cf. evaluation/operator_ensemble.py), trains a SINGLE FNO1D_hyper
jointly across a given subset of nu values, sharing weights via the hyper-R
spectral mixture (fno/fno_1D_hyper.py) instead of combining separately-
trained models after the fact.

Trains one such joint model PER SUBSET listed in the config (e.g. a pair, a
trio, all 13 nu) -- the point is comparing this single-shared-network
approach against the linear-regression-of-independent-models approach on
the exact same subsets, cf. evaluation/evaluate_operator_ensemble.py's own
comparison plots.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from tabulate import tabulate
from tqdm import tqdm

from fno.fno_1D_hyper import FNO1D_hyper
from training.ks_operator_dataset import build_operator_loaders_multi_nu
from training.factory import Factory
from training.metric_logger import MetricLogger

DATA_DIR = os.getenv("DATA_DIR", "../data")
LOG_DIR = os.getenv("LOG_DIR", "../runs")


class HyperOperatorTrainerFNO1D():
    """Same single-shot-regression loop as OperatorTrainerFNO1D, with nu
    threaded through every model call (batches are (u, dudt, nu) triples,
    forward(u, nu) instead of forward(u))."""

    def __init__(self, model, train_loader, test_loader, loss_fn, optimizer, scheduler,
                num_epochs, device, exp_dir, exp_name, start_epoch=1):
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
        self.current_epoch = start_epoch

    def _relative_l2(self, pred, target):
        return (torch.mean((pred - target) ** 2) / torch.mean(target ** 2)) ** 0.5 * 100

    def _run_batches(self, loader, train):
        total_loss = 0.0
        total_rel_l2 = 0.0
        batch_iter = tqdm(loader, desc=f"Epoch {self.current_epoch}", leave=False, ncols=90) if train else loader
        for u, dudt, nu in batch_iter:
            u = u.to(self.device).float().unsqueeze(-1)       # (B, Mx, 1)
            dudt = dudt.to(self.device).float().unsqueeze(-1)  # (B, Mx, 1)
            nu = nu.to(self.device).float()

            if train:
                self.optimizer.zero_grad()

            pred = self.model(u, nu)
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

            if train_loss < min_train_loss:
                min_train_loss = train_loss
                torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(),
                           "optimizer_state_dict": self.optimizer.state_dict()},
                          os.path.join(model_dir, "min_train_loss.pth"))

        torch.save({"epoch": self.num_epochs, "model_state_dict": self.model.state_dict(),
                   "optimizer_state_dict": self.optimizer.state_dict()},
                  os.path.join(model_dir, "final_model.pth"))


class OperatorTrainingKSHyper():
    """Loops over the named nu-subsets in the config, training one
    FNO1D_hyper jointly per subset (e.g. "pair"/"trio"/"all" ->
    <exp_name>_pair, <exp_name>_trio, <exp_name>_all)."""

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
        self.subsets = self.args["subsets"]  # {tag: [nu, ...]}

        self.k_max = self.args["k_max"]
        self.width = self.args["width"]
        self.n_fourier_layer = self.args.get("n_fourier_layer", 3)
        self.l = self.args.get("l", 1)
        self.hidden_proj = self.args.get("hidden_proj", 32)

        self.n_basis = self.args.get("n_basis", 4)
        self.param_embed_dim = self.args.get("param_embed_dim", 32)
        self.param_hidden_dim = self.args.get("param_hidden_dim", 64)
        self.param_encoder_layers = self.args.get("param_encoder_layers", 2)
        self.param_log_transform = self.args.get("param_log_transform", True)

        self.batch_size = self.args.get("batch_size", 20)
        self.num_workers = self.args.get("num_workers", 2)
        self.ds = self.args.get("ds", 1)
        self.train_frac = self.args.get("train_frac", 0.7)
        self.test_frac = self.args.get("test_frac", 0.3)

        self.num_epochs = self.args.get("num_epochs", 250)
        self.optimizer_info = self.args["optimizer"]
        self.scheduler_info = self.args.get("scheduler")
        self.loss_name = self.args.get("loss_fn", "l1")

    def _loss_fn(self):
        return nn.L1Loss() if self.loss_name == "l1" else nn.MSELoss()

    def make_directories(self, exp_name_tag):
        directories = [os.path.join(LOG_DIR, self.exp_dir),
                       os.path.join(LOG_DIR, self.exp_dir, exp_name_tag),
                       os.path.join(LOG_DIR, self.exp_dir, exp_name_tag, "model_weights"),
                       os.path.join(LOG_DIR, self.exp_dir, exp_name_tag, "logs")]
        for d in directories:
            os.makedirs(d, exist_ok=True)
        with open(os.path.join(LOG_DIR, self.exp_dir, exp_name_tag, "config.yaml"), "w") as f:
            yaml.safe_dump(self.args, f)

    def execute_experience(self):
        for tag, nu_values in self.subsets.items():
            exp_name_tag = f"{self.exp_name}_{tag}"
            print(f"\n===== Training hyper-nu operator FNO1D_hyper on nu={nu_values} ({exp_name_tag}) =====\n")
            self.make_directories(exp_name_tag)

            train_loader, test_loader, n_train, n_test, x_mean, x_std, y_mean, y_std = \
                build_operator_loaders_multi_nu(
                    DATA_DIR, self.exp_dir, nu_values, batch_size=self.batch_size,
                    num_workers=self.num_workers, ds=self.ds,
                    train_frac=self.train_frac, test_frac=self.test_frac,
                )
            print(f"Training samples: {n_train}, Testing samples: {n_test}")

            log_nu = np.log(np.array(nu_values, dtype=np.float64))
            param_mean, param_std = float(log_nu.mean()), float(log_nu.std() + 1e-8)

            model = FNO1D_hyper(
                input_dim=1, output_dim=1, modes=self.k_max, width=self.width, l=self.l,
                n_layer=self.n_fourier_layer, hidden_proj=self.hidden_proj,
                param_embed_dim=self.param_embed_dim, param_hidden_dim=self.param_hidden_dim,
                param_encoder_layers=self.param_encoder_layers, param_log_transform=self.param_log_transform,
                param_mean=param_mean, param_std=param_std, n_basis=self.n_basis, device=self.device,
                x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std,
            )
            model = model.to(self.device).float()
            print(f"Params: {model.count_parameters_per_module()['total']:,}")

            optimizer = Factory.get_optimizer(self.optimizer_info["type"], model.parameters(),
                                              lr=self.optimizer_info["lr"])
            scheduler = None
            if self.scheduler_info:
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=self.scheduler_info["step_size"], gamma=self.scheduler_info["gamma"])

            trainer = HyperOperatorTrainerFNO1D(
                model=model, train_loader=train_loader, test_loader=test_loader,
                loss_fn=self._loss_fn(), optimizer=optimizer, scheduler=scheduler,
                num_epochs=self.num_epochs, device=self.device,
                exp_dir=self.exp_dir, exp_name=exp_name_tag,
            )
            trainer.train_loop()
