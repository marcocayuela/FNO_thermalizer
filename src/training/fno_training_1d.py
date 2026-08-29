"""
1D counterpart of fno_training.py -- mono-parameter FNO emulator (predicts
the delta between two consecutive states) for the Kuramoto-Sivashinsky
equation, trained on a single nu instead of the 2D Kolmogorov velocity
field. training.trainer.Trainer itself is dimension-agnostic (its
train_epoch just calls self.model(x_t) and indexes outputs[:,t,...], cf.
session audit) and is reused unchanged -- only the model class and the
dataset manager differ from fno_training.py.
"""

import os

import torch
import yaml

from fno import fno_1D
from training.dataset_manager import DatasetManagerKS1D
from training.factory import Factory
from training.trainer import Trainer

DATA_DIR = os.getenv("DATA_DIR", "../data")
LOG_DIR = os.getenv("LOG_DIR", "../runs")


class EmulatorFNO1D(fno_1D.FNO1D):

    def __init__(self, input_dim, output_dim, modes, width, l,
                n_layer=4, hidden_proj=None, mlp=True, layers_mlp=None, tau=1e-5, device="cpu"):
        super().__init__(input_dim, output_dim, modes, width, l, n_layer, hidden_proj, mlp, layers_mlp, device=device)
        self.tau = tau

    def predict_sequence(self, x0, pred_horizon):
        # x0: (batch_size, Nx, input_dim)
        *batch_shape, nx, C = x0.shape

        outputs = torch.empty(
            (*batch_shape, pred_horizon + 1, nx, C),
            device=x0.device,
            dtype=x0.dtype
        )

        x_t = x0
        outputs[..., 0, :, :] = x_t
        for t in range(1, pred_horizon + 1):
            x_dt = self(x_t)
            x_t = x_t + x_dt + self.tau * torch.randn_like(x_dt)
            outputs[..., t, :, :] = x_t

        return outputs


class FNOTraining1D():

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

        self.ratio = self.args["ratio"]
        self.seq_length = self.args["seq_length"]
        self.batch_size = self.args["batch_size"]
        self.num_workers = self.args["num_workers"]
        self.loss_fn = self.args["loss_fn"]
        self.optimizer_info = self.args["optimizer"]
        self.num_epochs = self.args["num_epochs"]
        self.scheduler_info = self.args["scheduler"]
        self.metrics_name = self.args["metrics"]
        self.train_frac = self.args.get("train_frac", 0.7)
        self.test_frac = self.args.get("test_frac", 0.1)
        self.stride = self.args.get("stride", 1)
        self.prediction_mode = self.args.get("prediction_mode", "delta")
        self.ds = self.args.get("ds", 1)

        # normalize=False (unlike DatasetManagerMulti's own default of True):
        # SequenceDataset's normalization is dataset-side only -- x0/y get
        # rescaled going IN, but nothing denormalizes the model's output
        # coming back OUT at rollout time (EmulatorFNO1D/FNO1D have no
        # buffer-based denormalize, unlike PFNO2DHyper elsewhere in this
        # project). Training on a normalized delta target and then adding
        # the model's raw output straight onto an unnormalized state at
        # rollout silently mixes two different scales -- harmless for
        # Kolmogorov's velocity field (already O(1), so normalize barely
        # changes anything) but catastrophic for KS, whose delta std over
        # one dt is much smaller than the field's own std: the "corrected"
        # rollout diverges within ~15 steps regardless of the corrector,
        # because the emulator's per-step update is wrong by orders of
        # magnitude from the first step.
        self.datasets = DatasetManagerKS1D(data_rep=DATA_DIR, exp_dir=self.exp_dir,
                                           seq_length=self.seq_length, batch_size=self.batch_size,
                                           num_workers=self.num_workers, ratio=self.ratio,
                                           train_frac=self.train_frac, test_frac=self.test_frac,
                                           stride=self.stride, ds=self.ds,
                                           prediction_mode=self.prediction_mode,
                                           normalize=False)
        print("Datasets loaded.")
        print("Dataset summary:")
        print(f"Training samples: {self.datasets.n_train}, Testing samples: {self.datasets.n_test}")

        self.name_weights_to_load = self.args.get("name_weights_to_load", None)
        self.last_epoch = 0

        self.input_dim = self.args["input_dim"]
        self.output_dim = self.args["output_dim"]
        self.n_dim = self.args["n_dim"]
        self.domain_size = self.args["domain_size"]
        self.k_max = self.args["k_max"]
        self.l = self.args["l"]
        self.n_fourier_layer = self.args["n_fourier_layer"]
        self.width = self.args["width"]

        self.hidden_proj = self.args["hidden_proj"]
        self.mlp = self.args.get("mlp", True)
        self.layers_mlp = self.args.get("layers_mlp", None)
        self.tau = self.args.get("tau", 1e-5)

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
        with open(save_path, "w") as f:
            yaml.safe_dump(self.args, f)
            print(f"Configuration saved at: {save_path}")
        self.print_line()

    def execute_experience(self):
        print(f"Starting experiment: {self.exp_name}\n")

        self.make_directories()

        model = EmulatorFNO1D(input_dim=self.input_dim,
                              output_dim=self.output_dim,
                              modes=self.k_max,
                              width=self.width,
                              l=self.l,
                              n_layer=self.n_fourier_layer,
                              hidden_proj=self.hidden_proj,
                              mlp=self.mlp,
                              layers_mlp=self.layers_mlp,
                              tau=self.tau,
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
