import torch
import torch.nn as nn

from pydiver.datasets import barkley_datasets as barkley_datasets

from pydiver.models import lstm as lstm
from .utils import get_sampler

#import albumentations as A
#from albumentations.pytorch import ToTensorV2
from torch.optim import lr_scheduler
#from torch.optim.lr_scheduler import ReduceLROnPlateau

CONFIG_TYPES = {
    # # utils
    "__len__": lambda obj: len(obj),
    "method_call": lambda obj, method: getattr(obj, method)(),

    # # Dataset, DataLoader
    "BarkleyDataset": barkley_datasets.BarkleyDataset,

    "Sampler": get_sampler,
    "DataLoader": torch.utils.data.DataLoader,

    #"ToTensorV2": ToTensorV2,
    
    "STLSTM": lstm.STLSTM,
    
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    
    "OneCycleLR": lr_scheduler.OneCycleLR,
    "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau,
    
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
    "NLLLoss": nn.NLLLoss,
    "CrossEntropyLoss": nn.CrossEntropyLoss,

    "MeanSquaredError": nn.MSELoss,
    #"MeanAbsoluteError": nn.MAELoss


}