"""
1D counterpart of diffusion_training.py -- Re-blind (mono-parameter)
diffusion corrector for the Kuramoto-Sivashinsky equation, trained on a
single nu. training.trainer_diffusion.TrainerDiffusion and
training.DiffusionModel.Diffusion are both dimension-agnostic (cf. session
audit -- elementwise ops, no hardcoded field rank) and reused unchanged;
only the backbone model and dataset manager differ from diffusion_training.py.
"""

import os

import torch
import yaml

from fno.fno_1D_classifier import FNO1D_classifier
from training.dataset_manager import DatasetManagerKS1D
from training.DiffusionModel import Diffusion
from training.factory import Factory
from training.trainer_diffusion import TrainerDiffusion

DATA_DIR = os.getenv("DATA_DIR", "../data")
LOG_DIR = os.getenv("LOG_DIR", "../runs")


class DiffusionTraining1D():

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
        self.loss_score = self.args["loss_score"]
        self.loss_cat = self.args["loss_cat"]
        self.optimizer_info = self.args["optimizer"]
        self.num_epochs = self.args["num_epochs"]
        self.scheduler_info = self.args["scheduler"]
        self.stride = self.args.get("stride", 1)
        self.ds = self.args.get("ds", 1)

        self.datasets = DatasetManagerKS1D(data_rep=DATA_DIR, exp_dir=self.exp_dir,
                                           seq_length=self.seq_length, batch_size=self.batch_size,
                                           num_workers=self.num_workers, ratio=self.ratio,
                                           stride=self.stride, ds=self.ds, diffusion=True)
        print("Datasets loaded.")
        print("Dataset summary:")
        print(f"Training samples: {self.datasets.n_train}")

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

        self.class_mlp_layers = self.args.get("class_mlp_layers", None)
        self.timesteps = self.args.get("timesteps", 1000)
        self.noise_sampling_coeff = self.args.get("noise_sampling_coeff", 0.3)
        self.lambda_c = self.args.get("lambda_c", 1.)

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

        model = FNO1D_classifier(input_dim=self.input_dim,
                                 output_dim=self.output_dim,
                                 modes=self.k_max,
                                 width=self.width,
                                 l=self.l,
                                 n_layer=self.n_fourier_layer,
                                 hidden_proj=self.hidden_proj,
                                 mlp=self.mlp,
                                 layers_mlp=self.layers_mlp,
                                 class_mlp_layers=self.class_mlp_layers,
                                 n_cat=self.timesteps,
                                 device=self.device)

        if self.name_weights_to_load is not None:
            path_model = os.path.join(LOG_DIR, self.exp_dir, self.exp_name, "model_weights")
            loaded_weights = torch.load(os.path.join(path_model, self.name_weights_to_load))
            print(f"Loading weights from {self.name_weights_to_load}, epoch {loaded_weights['epoch']}")
            self.last_epoch = loaded_weights["epoch"]
            model.load_state_dict(loaded_weights["model_state_dict"])
            print("Weights loaded successfully.\n")

        param_dict = model.count_parameters_per_module()
        self.print_line()
        print("Model parameters per module:")
        for name, num in sorted(param_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:20s}: {num:,} params")
        self.print_line()

        model = model.to(self.device).float()

        diffusion_model = Diffusion(model=model,
                                    timesteps=self.timesteps,
                                    noise_sampling_coeff=self.noise_sampling_coeff)

        self.optimizer = Factory.get_optimizer(self.optimizer_info["type"], model.parameters(), lr=self.optimizer_info["lr"])
        self.scheduler = Factory.get_scheduler(self.scheduler_info, self.optimizer, self.num_epochs, self.datasets.n_batch_train)

        self.loss_score = Factory.get_metric(self.loss_score)
        self.loss_cat = Factory.get_metric(self.loss_cat)

        trainer = TrainerDiffusion(model=diffusion_model,
                                   train_loader=self.datasets.training_loader,
                                   loss_score=self.loss_score,
                                   loss_cat=self.loss_cat,
                                   optimizer=self.optimizer,
                                   scheduler=self.scheduler,
                                   num_epochs=self.num_epochs,
                                   device=self.device,
                                   exp_dir=self.exp_dir,
                                   exp_name=self.exp_name,
                                   lambda_c=self.lambda_c,
                                   start_epoch=self.last_epoch + 1 if self.name_weights_to_load is not None else 1)

        trainer.train_loop()

    def print_line(self):
        print("-------------------------------------------------------")
