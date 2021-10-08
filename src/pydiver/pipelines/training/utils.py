import numpy as np
#import torch
#from torch.utils.data import DataLoader
#from pydiver.datasets.barkley_datasets import InputDataset
#from tqdm import tqdm
import yaml
from torch.utils.data.sampler import SubsetRandomSampler

#from sklearn.metrics import mean_absolute_error, mean_squared_error


def get_sampler(dataset_length, val_split=0.1, train=True, shuffle=True, seed=42):
    indices = list(range(dataset_length))
    split = int(np.floor(val_split * dataset_length))
    
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    
    if train:
        idx = indices[split:]
    else:
        idx = indices[:split]

    sampler = SubsetRandomSampler(idx)
    
    return sampler
