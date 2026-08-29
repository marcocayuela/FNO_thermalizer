"""
Euler counterpart of fno_training.py/fno_training_1d.py -- mono-configuration
(gamma fixed) FNO emulator for euler_multi_quadrants_openBC (The Well),
predicting the ABSOLUTE state (not a delta -- cf. training/euler_dataset.py
and RT_FNO_vs_WNO's README: state prediction was found far more robust to
rollout divergence than delta on this dataset's shocks/discontinuities).
training.trainer.Trainer is dimension-agnostic and reused unchanged;
only the model class (padding-aware EmulatorFNO, cf. fno/fno_2D.py) and the
dataset construction (training.euler_dataset.load_euler_data, The Well's own
multi-trajectory-per-file HDF5 layout -- unlike Kolmogorov/KS's
Re<X>/train_traj/sim<N>.h5 convention) differ from fno_training.py.
"""

import os

import torch
import yaml

from training.euler_dataset import load_euler_data
from training.factory import Factory
from training.fno_training import EmulatorFNO
from training.trainer import Trainer

DATA_DIR = os.getenv("DATA_DIR", "../data")
LOG_DIR = os.getenv("LOG_DIR", "../runs")


class _EulerDatasets:
    """Thin adapter exposing load_euler_data()'s loaders/counts under the
    same attribute names DatasetManagerMulti/DatasetManagerKS1D already use
    (training_loader/testing_loader/n_train/n_test/n_batch_train) -- so
    Trainer/FNOTraining*'s execute_experience() need no special-casing for
    Euler. Trainer's periodic evaluation uses the VAL split (testing_loader
    = val_loader) -- the held-out test split is kept separately
    (self.test_loader/self.n_test_final) for a final, never-seen-during-
    training evaluation (e.g. in correction_eval_euler.py), matching
    load_euler_data's own train/val/test naming."""

    def __init__(self, config, data_root):
        (self.training_loader, self.testing_loader, self.test_loader,
         self.n_train, self.n_test, self.n_test_final, self.n_batch_train,
         self.mean, self.std, self.stats) = load_euler_data(config, data_root)


class FNOTrainingEuler():

    def __init__(self, args):

        self.args = args

        self.device_asked = self.args.get("device", "auto")
        if self.device_asked in ["cuda", "auto"] and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif self.device_asked in ["mps", "auto"] and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif self.device_asked in ["cpu", "auto"]:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        print(f"Device used: {self.device}")

        self.exp_dir = self.args["exp_dir"]
        self.exp_name = self.args["exp_name"]

        self.batch_size = self.args["batch_size"]
        self.num_workers = self.args.get("num_workers", 2)
        self.loss_fn = self.args["loss_fn"]
        self.optimizer_info = self.args["optimizer"]
        self.num_epochs = self.args["num_epochs"]
        self.scheduler_info = self.args["scheduler"]
        self.metrics_name = self.args["metrics"]
        # Always "state" for this dataset (cf. module docstring) -- kept
        # configurable rather than hardcoded so a future delta-mode
        # experiment doesn't need a second class, but the default matches
        # what actually works here.
        self.prediction_mode = self.args.get("prediction_mode", "state")

        self.datasets = _EulerDatasets(self.args, DATA_DIR)
        print("Datasets loaded.")
        print("Dataset summary:")
        print(f"Training samples: {self.datasets.n_train}, Testing samples: {self.datasets.n_test}")

        self.name_weights_to_load = self.args.get("name_weights_to_load", None)
        self.last_epoch = 0

        self.input_dim = self.args["n_channels"] * self.args["k"]
        self.output_dim = self.args["n_channels"]
        self.n_dim = 2
        self.k_max_x = self.args["modes_x"]
        self.k_max_y = self.args["modes_y"]
        self.l = self.args["kernel_size"]
        self.n_fourier_layer = self.args["n_layers"]
        self.width = self.args["width"]

        self.hidden_proj = self.args.get("hidden_proj")
        self.mlp = self.args.get("mlp", True)
        self.layers_mlp = self.args.get("layers_mlp", None)
        self.tau = self.args.get("tau", 1e-5)
        # Domain padding before the FFT (cf. fno/fno_2D.py) -- the whole
        # point of this experiment: euler_multi_quadrants_openBC has open
        # (extrapolation) BC on both axes, unlike Kolmogorov/KS's periodic
        # domains, so padding=0 (that class's own default) would leave the
        # emulator with FNO's implicit periodicity assumption baked in.
        self.padding = self.args.get("padding", 0)

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
        save_path = os.path.join(save_dir, "config.yaml")
        to_save = dict(self.args)
        to_save["field_mean"] = self.datasets.mean.tolist()
        to_save["field_std"] = self.datasets.std.tolist()
        with open(save_path, "w") as f:
            yaml.safe_dump(to_save, f)
            print(f"Configuration saved at: {save_path}")
        self.print_line()

    def execute_experience(self):
        print(f"Starting experiment: {self.exp_name}\n")

        self.make_directories()

        model = EmulatorFNO(input_dim=self.input_dim,
                            output_dim=self.output_dim,
                            modes_x=self.k_max_x,
                            modes_y=self.k_max_y,
                            width=self.width,
                            l=self.l,
                            n_layer=self.n_fourier_layer,
                            hidden_proj=self.hidden_proj,
                            mlp=self.mlp,
                            layers_mlp=self.layers_mlp,
                            tau=self.tau,
                            padding=self.padding,
                            device=self.device)

        param_dict = model.count_parameters_per_module()
        self.print_line()
        print("Model parameters per module:")
        for name, num in sorted(param_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:20s}: {num:,} params")
        self.print_line()

        if self.name_weights_to_load is not None:
            path_model = os.path.join(LOG_DIR, self.exp_dir, self.exp_name, "model_weights")
            loaded_weights = torch.load(os.path.join(path_model, self.name_weights_to_load))
            print(f"Loading weights from {self.name_weights_to_load}, epoch {loaded_weights['epoch']}")
            self.last_epoch = loaded_weights["epoch"]
            model.load_state_dict(loaded_weights["model_state_dict"])
            print("Weights loaded successfully.\n")

        model = model.to(self.device).float()

        self.optimizer = Factory.get_optimizer(self.optimizer_info["type"], model.parameters(), lr=self.optimizer_info["lr"])
        self.scheduler = Factory.get_scheduler(self.scheduler_info, self.optimizer, self.num_epochs, self.datasets.n_batch_train)
        self.metrics = {metric: Factory.get_metric(metric) for metric in self.metrics_name}
        self.loss_fn = Factory.get_metric(self.loss_fn)

        trainer = Trainer(
            model=model,
            train_loader=self.datasets.training_loader,
            test_loader=self.datasets.testing_loader,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            num_epochs=self.num_epochs,
            device=self.device,
            exp_dir=self.exp_dir,
            exp_name=self.exp_name,
            metrics=self.metrics,
            start_epoch=self.last_epoch + 1 if self.name_weights_to_load is not None else 1,
            prediction_mode=self.prediction_mode,
        )

        trainer.train_loop()

    def print_line(self):
        print("-------------------------------------------------------")
