import torch

import os 
import time

from tabulate import tabulate
from tqdm import tqdm

from training.metric_logger import MetricLogger


LOG_DIR = os.getenv("LOG_DIR", "../runs")

class TrainerDiffusion():

    def __init__(self, model, train_loader, loss_score, loss_cat, lambda_c, optimizer, scheduler, num_epochs, device, exp_dir, exp_name, start_epoch):
        super(TrainerDiffusion, self).__init__()

        self.model = model
        self.optimizer = optimizer
        self.loss_score = loss_score
        self.loss_cat = loss_cat
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.train_loader = train_loader
        self.exp_dir = exp_dir
        self.exp_name = exp_name
        self.start_epoch = start_epoch
        self.current_epoch = start_epoch
        self.lambda_c = lambda_c

    def train_epoch(self):

        # Training loop for one epoch
        self.model.train()
        train_loss_dict = {"training_loss": 0., "loss_score": 0., "loss_cat": 0.}

        batch_bar = tqdm(enumerate(self.train_loader),
                         total=len(self.train_loader),
                         desc=f"Epoch {self.current_epoch}",
                         leave=False,
                         ncols=90
                         )
        
        for batch_idx, image in batch_bar:

            image = image.to(self.device).float()
            self.optimizer.zero_grad()

            noise = torch.randn_like(image).to(self.device).float()
            pred, _, t, pred_level = self.model(image, noise, True)
            loss_score = self.loss_score(pred, noise)
            loss_classifier = self.loss_cat(pred_level, t)
            loss = loss_score + self.lambda_c * loss_classifier

            loss.backward()
            self.optimizer.step()
            if self.scheduler and self.scheduler.__class__.__name__ == "OneCycleLR":
                self.scheduler.step()
        
            train_loss_dict["training_loss"] += loss.item()
            train_loss_dict["loss_score"] += loss_score.item()
            train_loss_dict["loss_cat"] += loss_classifier.item()


        train_loss_dict["training_loss"] /= len(self.train_loader)
        train_loss_dict["loss_score"] /= len(self.train_loader)
        train_loss_dict["loss_cat"] /= len(self.train_loader)
    
        current_lr = self.optimizer.param_groups[0]['lr']
        return train_loss_dict, current_lr
    

    def train_loop(self):

        headers = ["Epoch", "training_loss", "loss_score", "loss_cat", "LR", "Time"]
        
        csv_path = os.path.join(self.exp_dir, self.exp_name, 'logs', 'metrics.csv')
        self.logger = MetricLogger(csv_path, headers)

        self.min_train_loss = 1e18
    
        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):

            start_time = time.time()
            train_loss_dict, current_lr = self.train_epoch()
            end_time = time.time()

            epoch_duration = end_time - start_time
            row = [epoch]
            row += train_loss_dict.values()
            row += [float(current_lr), float(epoch_duration)]

            formatted = [f"{v:.5f}" if isinstance(v, float) else v for v in row]
            table_str = tabulate([formatted], headers=headers, tablefmt="simple", colalign=("right",) * len(headers))
            print("\n")
            tqdm.write(table_str)

            row_dict = {h: val for h, val in zip(headers, row)}
            self.logger.log(row_dict)

            # Save the best model based on test loss
            if train_loss_dict['training_loss'] < self.min_train_loss:
                self.min_train_loss = train_loss_dict['training_loss']
                path = os.path.join(LOG_DIR, self.exp_dir, self.exp_name, 'model_weights', 'min_train_loss.pth')
                torch.save({'epoch':epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict':self.optimizer.state_dict()},
                            path)
                
                print(f"Best model saved at epoch {epoch} with train loss: {self.min_train_loss:.6f}")

            if self.scheduler and self.scheduler.__class__.__name__ == "CosineAnnealingLR":
                self.scheduler.step()

            self.current_epoch += 1

        path = os.path.join(LOG_DIR, self.exp_dir, self.exp_name, 'model_weights', 'final_model.pth')
        torch.save({'epoch':epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict':self.optimizer.state_dict()},
                    path)
        
        print(f"Final model saved at epoch {epoch} with train loss: {train_loss_dict['training_loss']:.6f}")     

        self.logger.close()