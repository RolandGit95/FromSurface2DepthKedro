import numpy as np
import torch
from torch.utils.data import DataLoader
from pydiver.datasets.barkley_datasets import InputDataset
from tqdm import tqdm
    
@torch.no_grad()
def predict(model, data, depths=[0,1,2], batch_size=8, device="cuda"):
    num_layers = len(depths)  
    
    model = model.to(device)
    
    dataset = InputDataset(torch.from_numpy(data))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    y_preds = []
    for X in tqdm(dataloader, total=len(dataloader)):
        X = X.to(device)
        y_pred = model(X, max_depth=num_layers).cpu().detach().numpy()
        
        y_preds.append(y_pred)  
        #break
        
    y_preds = np.concatenate(y_preds, 0)
    
    return y_preds


def validate(y_true, y_pred, depths=[0,1,2]):
    return 2

    
    
    
    
#import IPython ; IPython.embed() ; exit(1)
