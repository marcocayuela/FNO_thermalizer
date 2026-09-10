"""
Hyper-nu counterpart of fno_training_1d.py -- trains a single FNO1D_hyper
emulator across several Kuramoto-Sivashinsky viscosities (nu) at once,
instead of one FNO1D model per nu. Mirrors fno_training_1d.py wherever the
mechanism carries over (EmulatorFNO1D -> EmulatorFNO1DHyper, FNOTraining1D ->
FNOTrainingKS1DHyper); the shared training.trainer.Trainer is NOT reused here
(unlike fno_training_1d.py) because its train_epoch unpacks batches as
(inputs, targets) and calls self.model(x_t) with a single argument --
DatasetManagerMultiNuKS1D's batches are (x0, y, nu) triples and
FNO1D_hyper.forward(x, nu) needs the extra argument threaded through every
call, so a small dedicated loop is clearer than bolting an optional
parameter onto the shared Trainer used by every other (non-hyper) emulator.
"""

import os
import time

import torch
import yaml
from tabulate import tabulate
from tqdm import tqdm

from fno.fno_1D_hyper import FNO1D_hyper
from training.dataset_manager import DatasetManagerMultiNuKS1D
from training.factory import Factory, EarlyStopping
from training.metric_logger import MetricLogger

DATA_DIR = os.getenv("DATA_DIR", "../data")
LOG_DIR = os.getenv("LOG_DIR", "../runs")


class EmulatorFNO1DHyper(FNO1D_hyper):

    def __init__(self, *args, tau=1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau

    def predict_sequence(self, x0, nu, pred_horizon):
        # x0: (batch_size, Nx, input_dim); nu: (batch_size,)
        *batch_shape, nx, C = x0.shape

        outputs = torch.empty(
            (*batch_shape, pred_horizon + 1, nx, C),
            device=x0.device,
            dtype=x0.dtype,
        )

        x_t = x0
        outputs[..., 0, :, :] = x_t
        for t in range(1, pred_horizon + 1):
            x_dt = self(x_t, nu)
            x_t = x_t + x_dt + self.tau * torch.randn_like(x_dt)
            outputs[..., t, :, :] = x_t

        return outputs


class TrainerHyperEmulator1D():
    """Train/test loop for EmulatorFNO1DHyper, threading nu through every
    model call. Same overall structure (per-epoch Tr/Te pass, best-checkpoint
    saving, CSV logging) as training.trainer.Trainer, just adapted for the
    extra conditioning argument."""

    def __init__(self, model, train_loader, test_loader, loss_fn, optimizer, scheduler,
                num_epochs, device, exp_dir, exp_name, metrics, start_epoch,
                prediction_mode="delta", patience=None, min_delta=1e-8):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.exp_dir = exp_dir
        self.exp_name = exp_name
        self.prediction_mode = prediction_mode
        self.metrics = metrics
        self.start_epoch = start_epoch
        self.current_epoch = start_epoch
        self.early_stopping = EarlyStopping(patience=patience, min_delta=min_delta) if patience else None

    def _run_batches(self, loader, train):
        total_loss = 0.0
        total_metrics = {k: 0.0 for k in self.metrics.keys()}

        batch_iter = tqdm(loader, desc=f"Epoch {self.current_epoch}", leave=False, ncols=90) if train else loader
        for inputs, targets, nu in batch_iter:
            x_t = inputs.to(self.device).float()
            targets = targets.to(self.device).float()
            nu = nu.to(self.device).float()

            if train:
                self.optimizer.zero_grad()

            outputs = torch.empty_like(targets)
            for t in range(targets.shape[1]):
                if self.prediction_mode == "state":
                    x_t = self.model(x_t, nu)
                    outputs[:, t, ...] = x_t
                else:
                    x_dt = self.model(x_t, nu)
                    x_t = x_t + x_dt + self.model.tau * torch.randn_like(x_dt)
                    outputs[:, t, ...] = x_dt

            loss = self.loss_fn(outputs, targets)

            if train:
                loss.backward()
                self.optimizer.step()
                if self.scheduler and self.scheduler.__class__.__name__ == "OneCycleLR":
                    self.scheduler.step()

            total_loss += loss.item()
            for name, metric_fn in self.metrics.items():
                total_metrics[name] += metric_fn(outputs, targets).item()

        n_batches = len(loader)
        avg_loss = total_loss / n_batches
        for name in total_metrics:
            total_metrics[name] /= n_batches
        total_metrics["loss"] = avg_loss
        return total_metrics

    def train_epoch(self):
        self.model.train()
        train_metrics = self._run_batches(self.train_loader, train=True)

        self.model.eval()
        with torch.no_grad():
            test_metrics = self._run_batches(self.test_loader, train=False)

        current_lr = self.optimizer.param_groups[0]["lr"]
        return train_metrics, test_metrics, current_lr

    def train_loop(self):
        headers = ["Epoch"] + [f"Tr {k}" for k in self.metrics] + [f"Te {k}" for k in self.metrics] + \
            ["Tr loss", "Te loss"] + ["LR", "Time(s)"]

        csv_path = os.path.join(self.exp_dir, self.exp_name, "logs", "metrics.csv")
        self.logger = MetricLogger(csv_path, headers, resume=self.start_epoch > 1)

        min_train_loss = 1e18
        min_test_loss = 1e18
        # MetricLogger already prepends LOG_DIR to csv_path itself (cf.
        # metric_logger.py), so self.exp_dir stays unprefixed there -- but
        # torch.save below needs LOG_DIR explicit, same convention as
        # Trainer/TrainerDiffusion's own model_weights paths.
        model_dir = os.path.join(LOG_DIR, self.exp_dir, self.exp_name, "model_weights")

        for epoch in range(self.start_epoch, self.num_epochs + 1):
            self.current_epoch = epoch
            t0 = time.perf_counter()
            train_metrics, test_metrics, current_lr = self.train_epoch()
            if self.scheduler and self.scheduler.__class__.__name__ != "OneCycleLR":
                self.scheduler.step()
            elapsed = time.perf_counter() - t0

            row = [epoch] + [train_metrics[k] for k in self.metrics] + [test_metrics[k] for k in self.metrics] + \
                [train_metrics["loss"], test_metrics["loss"]] + [current_lr, elapsed]
            print(tabulate([row], headers=headers, floatfmt=".5g"))
            self.logger.log(dict(zip(headers, row)))  # MetricLogger.log() takes a dict, not a list

            if train_metrics["loss"] < min_train_loss:
                min_train_loss = train_metrics["loss"]
                torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(),
                           "optimizer_state_dict": self.optimizer.state_dict()},
                          os.path.join(model_dir, "min_train_loss.pth"))
                print(f"Best model saved at epoch {epoch} with train loss: {train_metrics['loss']:.6f}")

            if test_metrics["loss"] < min_test_loss:
                min_test_loss = test_metrics["loss"]
                torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(),
                           "optimizer_state_dict": self.optimizer.state_dict()},
                          os.path.join(model_dir, "min_test_loss.pth"))

            if self.early_stopping and self.early_stopping.step(test_metrics["loss"]):
                print(f"Early stopping at epoch {epoch} (no improvement since {self.early_stopping.patience} epochs)")
                break

        torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(),
                   "optimizer_state_dict": self.optimizer.state_dict()},
                  os.path.join(model_dir, "final_model.pth"))


class FNOTrainingKS1DHyper():

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

        self.ratio = self.args.get("ratio", 1)
        self.seq_length = self.args["seq_length"]
        self.batch_size = self.args["batch_size"]
        self.num_workers = self.args["num_workers"]
        self.loss_fn_name = self.args["loss_fn"]
        self.optimizer_info = self.args["optimizer"]
        self.num_epochs = self.args["num_epochs"]
        self.scheduler_info = self.args["scheduler"]
        self.metrics_name = self.args["metrics"]
        self.train_frac = self.args.get("train_frac", 0.7)
        self.test_frac = self.args.get("test_frac", 0.1)
        self.stride = self.args.get("stride", 1)
        self.prediction_mode = self.args.get("prediction_mode", "delta")
        self.ds = self.args.get("ds", 1)

        self.datasets = DatasetManagerMultiNuKS1D(
            data_rep=DATA_DIR, exp_dir=self.exp_dir, nu_values=self.nu_values,
            seq_length=self.seq_length, batch_size=self.batch_size, num_workers=self.num_workers,
            ratio=self.ratio, train_frac=self.train_frac, test_frac=self.test_frac,
            stride=self.stride, ds=self.ds, prediction_mode=self.prediction_mode, normalize=True,
        )
        print("Datasets loaded.")
        print("Dataset summary:")
        print(f"nu values: {self.nu_values}")
        print(f"Training samples: {self.datasets.n_train}, Testing samples: {self.datasets.n_test}")

        self.name_weights_to_load = self.args.get("name_weights_to_load", None)
        self.last_epoch = 0

        self.input_dim = self.args["input_dim"]
        self.output_dim = self.args["output_dim"]
        self.k_max = self.args["k_max"]
        self.l = self.args["l"]
        self.n_fourier_layer = self.args["n_fourier_layer"]
        self.width = self.args["width"]
        self.hidden_proj = self.args["hidden_proj"]
        self.mlp = self.args.get("mlp", True)
        self.layers_mlp = self.args.get("layers_mlp", None)
        self.tau = self.args.get("tau", 1e-5)

        self.n_basis = self.args.get("n_basis", 4)
        self.param_embed_dim = self.args.get("param_embed_dim", 32)
        self.param_hidden_dim = self.args.get("param_hidden_dim", 64)
        self.param_encoder_layers = self.args.get("param_encoder_layers", 2)
        self.param_log_transform = self.args.get("param_log_transform", True)

        self.patience = self.args.get("patience", None)
        self.min_delta = self.args.get("min_delta", 1e-8)

    def make_directories(self):
        directories = [os.path.join(LOG_DIR, self.exp_dir),
                       os.path.join(LOG_DIR, self.exp_dir, self.exp_name),
                       os.path.join(LOG_DIR, self.exp_dir, self.exp_name, "model_weights"),
                       os.path.join(LOG_DIR, self.exp_dir, self.exp_name, "logs")]
        self.print_line()
        print("Creating directories...")
        for d in directories:
            os.makedirs(d, exist_ok=True)
            print(f"Directory created (or already existing): {d}")

        save_dir = os.path.join(LOG_DIR, self.exp_dir, self.exp_name)
        to_save = dict(self.args)
        to_save["x_mean"] = self.datasets.x_mean.tolist()
        to_save["x_std"] = self.datasets.x_std.tolist()
        to_save["y_mean"] = self.datasets.y_mean.tolist()
        to_save["y_std"] = self.datasets.y_std.tolist()
        to_save["param_mean"] = self.datasets.param_mean
        to_save["param_std"] = self.datasets.param_std
        save_path = os.path.join(save_dir, "config.yaml")
        with open(save_path, "w") as f:
            yaml.safe_dump(to_save, f)
            print(f"Configuration saved at: {save_path}")
        self.print_line()

    def execute_experience(self):
        print(f"Starting experiment: {self.exp_name}\n")
        self.make_directories()

        model = EmulatorFNO1DHyper(
            input_dim=self.input_dim, output_dim=self.output_dim,
            modes=self.k_max, width=self.width, l=self.l, n_layer=self.n_fourier_layer,
            hidden_proj=self.hidden_proj, mlp=self.mlp, layers_mlp=self.layers_mlp,
            param_embed_dim=self.param_embed_dim, param_hidden_dim=self.param_hidden_dim,
            param_encoder_layers=self.param_encoder_layers, param_log_transform=self.param_log_transform,
            param_mean=self.datasets.param_mean, param_std=self.datasets.param_std,
            n_basis=self.n_basis, tau=self.tau, device=self.device,
            x_mean=self.datasets.x_mean, x_std=self.datasets.x_std,
            y_mean=self.datasets.y_mean, y_std=self.datasets.y_std,
        )

        param_dict = model.count_parameters_per_module()
        self.print_line()
        print("Model parameters per module:")
        for name, num in sorted(param_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:20s}: {num:,} params")
        self.print_line()

        if self.name_weights_to_load is not None:
            path_model = os.path.join(LOG_DIR, self.exp_dir, self.exp_name, "model_weights")
            loaded_weights = torch.load(os.path.join(path_model, self.name_weights_to_load),
                                        map_location=self.device)
            print(f"Loading weights from {self.name_weights_to_load}, epoch {loaded_weights['epoch']}")
            self.last_epoch = loaded_weights["epoch"]
            model.load_state_dict(loaded_weights["model_state_dict"])
            print("Weights loaded successfully.\n")

        model = model.to(self.device).float()

        self.optimizer = Factory.get_optimizer(self.optimizer_info["type"], model.parameters(), lr=self.optimizer_info["lr"])
        self.scheduler = Factory.get_scheduler(self.scheduler_info, self.optimizer, self.num_epochs, self.datasets.n_batch_train)
        self.metrics = {metric: Factory.get_metric(metric) for metric in self.metrics_name}
        self.loss_fn = Factory.get_metric(self.loss_fn_name)

        trainer = TrainerHyperEmulator1D(
            model=model, train_loader=self.datasets.training_loader, test_loader=self.datasets.testing_loader,
            loss_fn=self.loss_fn, optimizer=self.optimizer, scheduler=self.scheduler,
            num_epochs=self.num_epochs, device=self.device,
            exp_dir=self.exp_dir, exp_name=self.exp_name,
            metrics=self.metrics, start_epoch=self.last_epoch + 1 if self.name_weights_to_load is not None else 1,
            prediction_mode=self.prediction_mode,
            patience=self.patience, min_delta=self.min_delta,
        )
        trainer.train_loop()

    def print_line(self):
        print("-------------------------------------------------------")
