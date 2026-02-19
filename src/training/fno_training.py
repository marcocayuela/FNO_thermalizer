import os
from training.dataset_manager import DatasetManagerMulti
from training.trainer import Trainer
from training.factory import Factory
from fno import fno_2D
import torch
import yaml

import time



DATA_DIR = os.getenv("DATA_DIR", "../data")
LOG_DIR = os.getenv("LOG_DIR", "../runs")


class EmulatorFNO(fno_2D.FNO2D):

    def __init__(self, input_dim, output_dim, modes_x, modes_y, width, l,
                 n_layer=4, hidden_proj=None, mlp=True, layers_mlp=None, tau=1e-5, device="cpu"):

        super().__init__(input_dim, output_dim, modes_x, modes_y, width, l, n_layer, hidden_proj, mlp, layers_mlp, device=device)

        self.tau = tau


    def predict_sequence(self, x0, pred_horizon):
        # x0: (batch_size, input_dim, nx, ny)
        *batch_shape, nx, ny, C = x0.shape

        outputs = torch.empty(
            (*batch_shape, pred_horizon + 1, nx, ny, C),
            device=x0.device,
            dtype=x0.dtype
        )

        x_t = x0
        outputs[...,0,:,:,:] = x_t
        for t in range(1, pred_horizon + 1):
            x_dt = self(x_t)
            x_t = x_t + x_dt + self.tau*torch.randn_like(x_dt)
            outputs[...,t,:,:,:] = x_t

        return outputs



class FNOTraining():

    def __init__(self, args):

        self.args = args

        self.device_asked = self.args.get("device", "auto")
        #Device parameters
        if self.device_asked in ["cuda","auto"] and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif self.device_asked in ["mps","auto"] and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif self.device_asked in ["cpu","auto"]:
            self.device = torch.device("cpu")

        print(f"Device used: {self.device}")
        
        # File and repo managning 
        self.exp_dir = self.args["exp_dir"]
        self.exp_name = self.args["exp_name"]

        # Training parameters
        self.ratio = self.args["ratio"]
        self.seq_length = self.args["seq_length"]
        self.batch_size = self.args["batch_size"]
        self.num_workers = self.args["num_workers"]
        self.loss_fn = self.args["loss_fn"]
        self.optimizer_info = self.args["optimizer"]
        self.num_epochs = self.args["num_epochs"]
        self.scheduler_info = self.args["scheduler"]
        self.metrics_name = self.args["metrics"]
        self.train_frac = self.args.get("train_frac", 0.5)
        self.test_frac = self.args.get("test_frac", 0.2)
        self.stride = self.args.get("stride", 1)
        self.ds = self.args.get("ds", 2)

        # Datasets
        self.datasets = DatasetManagerMulti(data_rep=DATA_DIR, exp_dir=self.exp_dir,
                                            seq_length=self.seq_length, batch_size=self.batch_size,
                                            num_workers=self.num_workers, ratio=self.ratio, 
                                            train_frac=self.train_frac, test_frac=self.test_frac,
                                            stride=self.stride, ds = self.ds)
        print("Datasets loaded.")
        print("Dataset summary:")
        print(f"Training samples: {self.datasets.n_train}, Testing samples: {self.datasets.n_test}")

        # Model definition
        self.name_weights_to_load = self.args.get("name_weights_to_load", None)

        self.input_dim = self.args["input_dim"]
        self.output_dim = self.args["output_dim"]
        self.n_dim = self.args["n_dim"]
        self.domain_size = self.args["domain_size"]
        self.k_max_x = self.args["k_max_x"]
        self.k_max_y = self.args["k_max_y"]
        self.l = self.args["l"]
        self.n_fourier_layer = self.args["n_fourier_layer"]
        self.width = self.args["width"]

        self.hidden_proj = self.args["hidden_proj"]
        self.mlp = self.args.get("mlp", True)
        self.layers_mlp = self.args.get("layers_mlp", None)
        self.tau = self.args.get("tau", 1e-5)


    def make_directories(self):

        directories = [os.path.join(LOG_DIR,self.exp_dir),
                       os.path.join(LOG_DIR,self.exp_dir, self.exp_name),
                       os.path.join(LOG_DIR,self.exp_dir, self.exp_name, 'model_weights'),
                       os.path.join(LOG_DIR,self.exp_dir, self.exp_name, 'logs')]
        
        self.print_line()
        print("Creating directories...")
        for d in directories:
            os.makedirs(d, exist_ok=True)
            print(f"Directory created (or already existing): {d}")

        save_dir = os.path.join(LOG_DIR, self.exp_dir, self.exp_name)
        save_path = os.path.join(save_dir, "config.yaml")
        with open(save_path, "w") as f:
            yaml.safe_dump(self.args, f)  # écrit le dictionnaire args dans le fichier
            print(f"Configuration saved at: {save_path}")
        self.print_line()
        



    def execute_experience(self):


        print(f"Starting experiment: {self.exp_name}\n")

        # Directories creation
        self.make_directories()

        # Load or create model
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
                            device=self.device
                            )
        
        param_dict = model.count_parameters_per_module()
        self.print_line()
        print("Model parameters per module:")
        for name, num in sorted(param_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"{name:20s}: {num:,} params")
        self.print_line()

        if self.name_weights_to_load is not None:
            path_model = os.path.join('runs',self.exp_dir,self.exp_name,'model_weights')
            loaded_weights = torch.load(os.path.join(path_model, self.name_weights_to_load))
            print(f"Loading weights from {self.name_weights_to_load}, epoch {loaded_weights['epoch']}")
            self.last_epoch = loaded_weights['epoch']
            state_dict = loaded_weights['model_state_dict']
            model.load_state_dict(state_dict)
            print("Weights loaded successfully.\n")

        model = model.to(self.device).float()

        self.optimizer = Factory.get_optimizer(self.optimizer_info["type"], model.parameters(), lr=self.optimizer_info["lr"])
        self.scheduler = Factory.get_scheduler(self.scheduler_info, self.optimizer, self.num_epochs, self.datasets.n_batch_train)
        self.metrics = {metric: Factory.get_metric(metric) for metric in self.metrics_name}
        self.loss_fn = Factory.get_metric(self.loss_fn)
        # Create the trainer and train

        trainer = Trainer(model=model,
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
                          start_epoch=self.last_epoch + 1 if self.name_weights_to_load is not None else 1)
        
        trainer.train_loop()

    def print_line(self):
        print("-------------------------------------------------------")


