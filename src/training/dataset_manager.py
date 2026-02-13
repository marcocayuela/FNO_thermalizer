import numpy as np
import os 
import h5py
import torch

from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import random_split
from torch.utils.data import Dataset



class SequenceDataset(Dataset):
    
    def __init__(self, data, seq_length, stride=1):
        """
        data: array ou tensor de forme (nt, ...)
        seq_length: horizon de prédiction
        stride: pas entre les séquences 
        """
        if not torch.is_tensor(data):
            data = torch.from_numpy(data)

        self.data = data.float()
        self.seq_length = seq_length
        self.stride = stride

    def __len__(self):
        # on a besoin de t + seq_length
        return (self.data.shape[0] - self.seq_length - 1) // self.stride 

    def __getitem__(self, idx):
        # snapshot initial
        idx = idx * self.stride
        x0 = self.data[idx]                      # (...)

        # séquence future
        y = self.data[idx + 1 : idx + 1 + self.seq_length] - self.data[idx : idx + self.seq_length]  # (seq_length, ...)

        return x0, y
    
class FirstSnapshot(Dataset):

    def __init__(self, data, seq_length, stride=1):
        """
        data: array ou tensor de forme (nt, ...)
        seq_length: horizon de prédiction
        stride: pas entre les séquences 
        """
        if not torch.is_tensor(data):
            data = torch.from_numpy(data)

        self.data = data.float()
        self.seq_length = seq_length
        self.stride = stride

    def __len__(self):
        # on a besoin de t + seq_length
        return (self.data.shape[0] - self.seq_length - 1) // self.stride 

    def __getitem__(self, idx):
        # snapshot initial
        idx = idx * self.stride
        x0 = self.data[idx]                      # (...)

        # séquence future
        #y = self.data[idx + 1 : idx + 1 + self.seq_length] - self.data[idx : idx + self.seq_length]  # (seq_length, ...)

        return x0#, y


class DatasetManagerMulti():

    def __init__(self, data_rep, exp_dir, seq_length, batch_size, num_workers, ratio=1, train_frac=0.3, test_frac=0.1, stride=1, ds=2, diffusion=False):

        self.exp_dir = exp_dir
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_dir = None
        self.training_loader = None
        self.testing_loader = None
        self.stride = stride
        self.ds = ds
        if type(ratio)!=int or ratio <1:
            print("Ratio must be an integer greater than 1. It has been automatically set to 1")
            self.ratio = 1
        else:
            self.ratio = ratio  

        if self.exp_dir == "kolmogorov/Re34" or self.exp_dir == "kolmogorov/Re90":

            data_dir = os.path.join(data_rep, self.exp_dir, "train_traj")

            simulation_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".h5")]
            
            datasets = []

            if not diffusion:
             
                for sim_file in simulation_files:
                    with h5py.File(sim_file, "r") as f:
                        data = f["velocity_field"][()][::self.ratio,::self.ds, ::self.ds]
                        datasets.append(SequenceDataset(data, seq_length=self.seq_length, stride=self.stride))
                    
                self.sequence_dataset = ConcatDataset(datasets)

            
                N = len(self.sequence_dataset)
                self.train_frac = train_frac
                self.test_frac = test_frac

                self.n_train = int(self.train_frac*N)
                self.n_test = int(self.test_frac*N)
                self.n_rest = N - self.n_train - self.n_test

                self.training_dataset, self.testing_dataset, _ = random_split(self.sequence_dataset, [self.n_train, self.n_test, self.n_rest])

                self.training_loader = DataLoader(self.training_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                num_workers=self.num_workers,
                                                pin_memory=False,
                                                persistent_workers=True
                                                )

                self.testing_loader = DataLoader(self.testing_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=False,
                                                num_workers=self.num_workers,
                                                pin_memory=False,
                                                persistent_workers=True
                                                )
                
                self.n_batch_train = len(self.training_loader)
                self.n_batch_test = len(self.testing_loader)


            else:
                for sim_file in simulation_files:
                        with h5py.File(sim_file, "r") as f:
                            data = f["velocity_field"][()][::self.ratio,::self.ds, ::self.ds]
                            datasets.append(FirstSnapshot(data, seq_length=self.seq_length, stride=self.stride))
                        
                self.training_dataset = ConcatDataset(datasets)
                self.n_train = len(self.training_dataset)

                self.training_loader = DataLoader(self.training_dataset,
                                                batch_size=self.batch_size,
                                                shuffle=True,
                                                num_workers=self.num_workers,
                                                pin_memory=False,
                                                persistent_workers=True
                                                )
                
                self.n_batch_train = len(self.training_loader)
 

        


