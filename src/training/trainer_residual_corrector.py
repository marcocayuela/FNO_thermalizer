"""
Trainer for training/ResidualCorrectorModel.py::ResidualCorrector -- a
hybrid of training/trainer.py::Trainer's train/test loop (TrainerDiffusion
never had one; here it's the whole point, cf. the user's explicit ask to
verify the corrector doesn't just overfit to the emulator's behavior on the
specific trajectories it was shown) and training/trainer_diffusion.py::TrainerDiffusion's
loss composition (a regression loss + a classifier loss, weighted by
lambda_c) -- except the regression target is the real residual
(true - corrupted) from training/euler_dataset.py::LazyEulerCorrectionDataset,
not a noise-prediction loss, and there is no reverse-diffusion sampling.
"""

import os
import time

import torch
from tabulate import tabulate
from tqdm import tqdm

import training.factory as Factory
from training.metric_logger import MetricLogger

LOG_DIR = os.getenv("LOG_DIR", "../runs")


class TrainerResidualCorrector():

    def __init__(self, model, train_loader, test_loader, loss_score, loss_cat, lambda_c,
                 optimizer, scheduler, num_epochs, device, exp_dir, exp_name, start_epoch,
                 patience=None, min_delta=1e-8):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_score = loss_score
        self.loss_cat = loss_cat
        self.lambda_c = lambda_c
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.device = device
        self.exp_dir = exp_dir
        self.exp_name = exp_name
        self.start_epoch = start_epoch
        self.current_epoch = start_epoch

        self.early_stopping = Factory.EarlyStopping(patience=patience, min_delta=min_delta) if patience else None

    def _run_epoch(self, loader, train):
        self.model.train() if train else self.model.eval()
        totals = {"loss": 0., "loss_score": 0., "loss_cat": 0.}

        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            batch_bar = tqdm(enumerate(loader), total=len(loader),
                             desc=f"Epoch {self.current_epoch} ({'train' if train else 'test'})",
                             leave=False, ncols=90)
            for batch_idx, (corrupted, true, step) in batch_bar:
                corrupted = corrupted.to(self.device).float()
                true = true.to(self.device).float()
                step = step.to(self.device).long()

                if train:
                    self.optimizer.zero_grad()

                residual_pred, step_logits = self.model(corrupted, predict_class=True)
                loss_score = self.loss_score(residual_pred, true - corrupted)
                loss_cat = self.loss_cat(step_logits, step)
                loss = loss_score + self.lambda_c * loss_cat

                if train:
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler and self.scheduler.__class__.__name__ == "OneCycleLR":
                        self.scheduler.step()

                totals["loss"] += loss.item()
                totals["loss_score"] += loss_score.item()
                totals["loss_cat"] += loss_cat.item()

        for k in totals:
            totals[k] /= len(loader)
        return totals

    def train_epoch(self):
        train_metrics = self._run_epoch(self.train_loader, train=True)
        test_metrics = self._run_epoch(self.test_loader, train=False)
        current_lr = self.optimizer.param_groups[0]["lr"]
        return train_metrics, test_metrics, current_lr

    def train_loop(self):
        headers = ["Epoch", "Tr loss", "Te loss", "Tr loss_score", "Te loss_score",
                  "Tr loss_cat", "Te loss_cat", "LR", "Time(s)"]

        csv_path = os.path.join(self.exp_dir, self.exp_name, "logs", "metrics.csv")
        self.logger = MetricLogger(csv_path, headers, resume=self.start_epoch > 1)

        self.min_train_loss = 1e18
        self.min_test_loss = 1e18

        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            start_time = time.time()
            train_metrics, test_metrics, current_lr = self.train_epoch()
            epoch_duration = time.time() - start_time

            row = [epoch, train_metrics["loss"], test_metrics["loss"],
                  train_metrics["loss_score"], test_metrics["loss_score"],
                  train_metrics["loss_cat"], test_metrics["loss_cat"],
                  current_lr, epoch_duration]
            formatted = [f"{v:.5f}" if isinstance(v, float) else v for v in row]
            table_str = tabulate([formatted], headers=headers, tablefmt="simple", colalign=("right",) * len(headers))
            print("\n")
            tqdm.write(table_str)

            self.logger.log({h: v for h, v in zip(headers, row)})

            if test_metrics["loss"] < self.min_test_loss:
                self.min_test_loss = test_metrics["loss"]
                path = os.path.join(LOG_DIR, self.exp_dir, self.exp_name, "model_weights", "min_test_loss.pth")
                torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(),
                          "optimizer_state_dict": self.optimizer.state_dict()}, path)
                print(f"Best model saved at epoch {epoch} with test loss: {self.min_test_loss:.6f}")

            if train_metrics["loss"] < self.min_train_loss:
                self.min_train_loss = train_metrics["loss"]
                path = os.path.join(LOG_DIR, self.exp_dir, self.exp_name, "model_weights", "min_train_loss.pth")
                torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(),
                          "optimizer_state_dict": self.optimizer.state_dict()}, path)
                print(f"Best model saved at epoch {epoch} with train loss: {self.min_train_loss:.6f}")

            if self.scheduler and self.scheduler.__class__.__name__ == "CosineAnnealingLR":
                self.scheduler.step()

            self.current_epoch += 1

            if self.early_stopping and self.early_stopping.step(test_metrics["loss"]):
                print(f"Early stopping at epoch {epoch} (no amelioration since {self.early_stopping.patience} epochs)")
                break

        path = os.path.join(LOG_DIR, self.exp_dir, self.exp_name, "model_weights", "final_model.pth")
        torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(),
                  "optimizer_state_dict": self.optimizer.state_dict()}, path)
        print(f"Final model saved at epoch {epoch} with train loss: {train_metrics['loss']:.6f} "
              f"and test loss: {test_metrics['loss']:.6f}")

        self.logger.close()
