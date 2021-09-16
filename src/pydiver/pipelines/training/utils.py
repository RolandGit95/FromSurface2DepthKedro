import numpy as np
#import torch
#from torch.utils.data import DataLoader
#from pydiver.datasets.barkley_datasets import InputDataset
#from tqdm import tqdm
import yaml
from torch.utils.data.sampler import SubsetRandomSampler

#from sklearn.metrics import mean_absolute_error, mean_squared_error


def get_train_sampler(dataset_length, val_split=0.1, batch_size=64, shuffle=True, seed=42):
    indices = list(range(dataset_length))
    split = int(np.floor(val_split * dataset_length))
    
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    
    train_idx = indices[split:]
    train_sampler = SubsetRandomSampler(train_idx)
    
    return train_sampler

def get_val_sampler(dataset_length, val_split=0.1, batch_size=64, shuffle=True, seed=42):  
    indices = list(range(dataset_length))
    split = int(np.floor(val_split * dataset_length))
    
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    
    valid_idx = indices[:split]
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    return valid_sampler


def setDataloader(X, Y, cfg):
    train_dataloader = cfg['/dataloader']['train']
    test_dataloader = cfg['/dataloader']['train']
    train_dataloader.dataset.setData(X,Y)
    test_dataloader.dataset.setData(X,Y)
    return train_dataloader, test_dataloader

def train(X_train, Y_train):
    pass#import IPython ; IPython.embed() ; exit(1)