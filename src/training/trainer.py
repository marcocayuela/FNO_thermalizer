import torch

import os 
import time

from tabulate import tabulate
from tqdm import tqdm

from training.metric_logger import MetricLogger


LOG_DIR = os.getenv("LOG_DIR", "../runs")

class Trainer():

    def __init__(self, model, train_loader, test_loader, loss_fn, optimizer, scheduler, num_epochs, device, exp_dir, exp_name, metrics, start_epoch):
        super(Trainer, self).__init__()

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
        self.metrics = metrics
        self.start_epoch = start_epoch
        self.current_epoch = start_epoch

    def train_epoch(self):


        # Training loop for one epoch
        self.model.train()
        train_loss = 0.0
        train_metrics = {k: 0.0 for k in self.metrics.keys()}

        batch_bar = tqdm(
        enumerate(self.train_loader),
        total=len(self.train_loader),
        desc=f"Epoch {self.current_epoch}",
        leave=False,
        ncols=90
    )
        for batch_idx, (inputs, targets) in batch_bar:

            x_t, targets = inputs.to(self.device).float(), targets.to(self.device).float()
            self.optimizer.zero_grad()

            outputs = torch.empty_like(targets)
            for t in range(targets.shape[1]):
                x_dt = self.model(x_t)
                x_t = x_t + x_dt + self.model.tau*torch.randn_like(x_dt)
                outputs[:,t,...] = x_dt

            #outputs = self.model.predict_sequence(inputs, pred_horizon=targets.shape[1])

            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()
            if self.scheduler and self.scheduler.__class__.__name__ == "OneCycleLR":
                self.scheduler.step()
        
            train_loss += loss.item()
            for name, metric_fn in self.metrics.items():
                train_metrics[name] += metric_fn(outputs, targets).item()

        avg_epoch_loss = train_loss / len(self.train_loader)
        for name in train_metrics.keys():
            train_metrics[name] /= len(self.train_loader)
        train_metrics["loss"] = avg_epoch_loss


        # Evaluation loop for one epoch
        self.model.eval()
        test_loss = 0.0
        test_metrics = {k: 0.0 for k in self.metrics.keys()}

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                x_t, targets = inputs.to(self.device).float(), targets.to(self.device).float()

                outputs = torch.empty_like(targets)
                for t in range(targets.shape[1]):
                    x_dt = self.model(x_t)
                    x_t = x_t + x_dt + self.model.tau*torch.randn_like(x_dt)
                    outputs[:,t,...] = x_dt
                    
                loss = self.loss_fn(outputs, targets)
                test_loss += loss.item()

                for name, metric_fn in self.metrics.items():
                    test_metrics[name] += metric_fn(outputs, targets).item()

            
            avg_test_loss = test_loss / len(self.test_loader)
            for name in test_metrics.keys():
                test_metrics[name] /= len(self.test_loader)
            test_metrics["loss"] = avg_test_loss


        current_lr = self.optimizer.param_groups[0]['lr']

        return train_metrics, test_metrics, current_lr
    

    def train_loop(self):


        headers = ["Epoch"] + \
                  [f"Tr {k}" for k in self.metrics] + \
                  [f"Te {k}" for k in self.metrics] + \
                  ["Tr loss", "Te loss"] + \
                  ["LR", "Time(s)"]
        
        csv_path = os.path.join(self.exp_dir, self.exp_name, 'logs', 'metrics.csv')
        self.logger = MetricLogger(csv_path, headers)

        self.min_train_loss = 1e18
        self.min_test_loss  = 1e18


        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):

            start_time = time.time()
            train_metrics, test_metrics, current_lr = self.train_epoch()
            end_time = time.time()

            epoch_duration = end_time - start_time
            headers = ["Epoch"]
            row = [epoch]

            for k in train_metrics.keys():
                headers += [f"Tr {k}", f"Te {k}"]
                row += [float(train_metrics[k]), float(test_metrics[k])]
                
            # Ajouter LR et temps
            headers += ["LR", "Time(s)"]
            row += [float(current_lr), float(epoch_duration)]

            formatted = [f"{v:.5f}" if isinstance(v, float) else v for v in row]
            table_str = tabulate([formatted], headers=headers, tablefmt="simple", colalign=("right",) * len(headers))
            print("\n")
            tqdm.write(table_str)

            row_dict = {h: val for h, val in zip(headers, row)}
            self.logger.log(row_dict)

            # Save the best model based on test loss
            if test_metrics['loss'] < self.min_test_loss:
                self.min_test_loss = test_metrics['loss']
                path = os.path.join(LOG_DIR, self.exp_dir, self.exp_name, 'model_weights', 'min_test_loss.pth')
                torch.save({'epoch':epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict':self.optimizer.state_dict()},
                            path)
                
                print(f"Best model saved at epoch {epoch} with test loss: {self.min_test_loss:.6f}")


            if train_metrics['loss'] < self.min_train_loss:
                self.min_train_loss = train_metrics['loss']
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
        
        print(f"Final model saved at epoch {epoch} with train loss: {train_metrics['loss']:.6f} and test loss: {test_metrics['loss']:.6f}")     

        self.logger.close()