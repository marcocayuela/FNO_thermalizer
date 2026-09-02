"""
Orchestrates training of the residual corrector (cf.
training/ResidualCorrectorModel.py, training/trainer_residual_corrector.py)
-- mirrors diffusion_training_euler.py's structure, but the "diffusion"
name doesn't apply here: this trains FNO2D_classifier (reused unchanged) to
predict a real emulator's own rollout residual in one shot, not to
denoise synthetic Gaussian noise. See training/ResidualCorrectorModel.py's
docstring for why.

Data: training/generate_euler_correction_data.py must have already been
run twice (--role train and --role val) to produce the two correction-data
HDF5 files this class's config points at (train_correction_file/
val_correction_file) -- this class does NOT run the emulator itself, only
loads its already-precomputed rollouts (cf. that script's own docstring for
why: running the emulator is the expensive part, done once, not repeated
every epoch).
"""

import os

import torch
import yaml

from fno.fno_2D_classifier import FNO2D_classifier
from training.euler_dataset import LazyEulerCorrectionDataset
from training.factory import Factory
from training.ResidualCorrectorModel import ResidualCorrector
from training.trainer_residual_corrector import TrainerResidualCorrector

DATA_DIR = os.getenv("DATA_DIR", "../data")
LOG_DIR = os.getenv("LOG_DIR", "../runs")


class ResidualCorrectorTrainingEuler():

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
        self.loss_score = self.args["loss_score"]
        self.loss_cat = self.args["loss_cat"]
        self.optimizer_info = self.args["optimizer"]
        self.num_epochs = self.args["num_epochs"]
        self.scheduler_info = self.args["scheduler"]
        self.patience = self.args.get("patience", None)
        self.min_delta = self.args.get("min_delta", 1e-8)

        train_ds = LazyEulerCorrectionDataset(
            os.path.join(DATA_DIR, self.args["train_correction_file"]), data_root=DATA_DIR,
            stride=self.args.get("stride", 1), max_step=self.args.get("max_step"))
        test_ds = LazyEulerCorrectionDataset(
            os.path.join(DATA_DIR, self.args["val_correction_file"]), data_root=DATA_DIR,
            stride=self.args.get("stride", 1), max_step=self.args.get("max_step"))
        self.training_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0)
        self.testing_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True, persistent_workers=self.num_workers > 0)
        self.n_train, self.n_test = len(train_ds), len(test_ds)
        self.n_batch_train = len(self.training_loader)
        print("Datasets loaded.")
        print("Dataset summary:")
        print(f"Training samples: {self.n_train}, Testing samples: {self.n_test}")

        self.name_weights_to_load = self.args.get("name_weights_to_load", None)
        self.last_epoch = 0

        self.input_dim = self.args["n_channels"]
        self.output_dim = self.args["n_channels"]
        self.k_max_x = self.args["modes_x"]
        self.k_max_y = self.args["modes_y"]
        self.l = self.args["kernel_size"]
        self.n_fourier_layer = self.args["n_layers"]
        self.width = self.args["width"]

        self.hidden_proj = self.args.get("hidden_proj")
        self.mlp = self.args.get("mlp", True)
        self.layers_mlp = self.args.get("layers_mlp", None)
        self.class_mlp_layers = self.args.get("class_mlp_layers", None)
        # max_step: how many rollout-step buckets the classifier head
        # predicts (cf. ResidualCorrector docstring) -- reuses the SAME
        # n_cat slot FNO2D_classifier already has for diffusion timesteps.
        self.max_step = self.args["max_step"]
        self.lambda_c = self.args.get("lambda_c", 1.)
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
        with open(save_path, "w") as f:
            yaml.safe_dump(self.args, f)
            print(f"Configuration saved at: {save_path}")
        self.print_line()

    def execute_experience(self):
        print(f"Starting experiment: {self.exp_name}\n")
        self.make_directories()

        backbone = FNO2D_classifier(input_dim=self.input_dim,
                                    output_dim=self.output_dim,
                                    modes_x=self.k_max_x,
                                    modes_y=self.k_max_y,
                                    width=self.width,
                                    l=self.l,
                                    n_layer=self.n_fourier_layer,
                                    hidden_proj=self.hidden_proj,
                                    mlp=self.mlp,
                                    layers_mlp=self.layers_mlp,
                                    class_mlp_layers=self.class_mlp_layers,
                                    n_cat=self.max_step + 1,
                                    padding=self.padding,
                                    device=self.device)

        model = ResidualCorrector(model=backbone, max_step=self.max_step)

        if self.name_weights_to_load is not None:
            path_model = os.path.join(LOG_DIR, self.exp_dir, self.exp_name, "model_weights")
            loaded_weights = torch.load(os.path.join(path_model, self.name_weights_to_load))
            print(f"Loading weights from {self.name_weights_to_load}, epoch {loaded_weights['epoch']}")
            self.last_epoch = loaded_weights["epoch"]
            model.load_state_dict(loaded_weights["model_state_dict"])
            print("Weights loaded successfully.\n")

        param_dict = backbone.count_parameters_per_module()
        self.print_line()
        print("Model parameters per module:")
        for name, num in sorted(param_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:20s}: {num:,} params")
        self.print_line()

        model = model.to(self.device).float()

        self.optimizer = Factory.get_optimizer(self.optimizer_info["type"], model.parameters(), lr=self.optimizer_info["lr"])
        self.scheduler = Factory.get_scheduler(self.scheduler_info, self.optimizer, self.num_epochs, self.n_batch_train)
        self.loss_score = Factory.get_metric(self.loss_score)
        self.loss_cat = Factory.get_metric(self.loss_cat)

        trainer = TrainerResidualCorrector(
            model=model,
            train_loader=self.training_loader,
            test_loader=self.testing_loader,
            loss_score=self.loss_score,
            loss_cat=self.loss_cat,
            lambda_c=self.lambda_c,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            num_epochs=self.num_epochs,
            device=self.device,
            exp_dir=self.exp_dir,
            exp_name=self.exp_name,
            start_epoch=self.last_epoch + 1 if self.name_weights_to_load is not None else 1,
            patience=self.patience,
            min_delta=self.min_delta,
        )

        trainer.train_loop()

    def print_line(self):
        print("-------------------------------------------------------")
