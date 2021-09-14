import numpy as np
import torch
from torch.utils.data import DataLoader
from pydiver.datasets.barkley_datasets import InputDataset
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error

loss_functions = dict(
    mae=lambda a, b: np.mean(np.abs(a-b)),
    mse=mean_squared_error,
)

    
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

def validate(y_true, y_pred, depths=[0,1,2], loss_function="mae", batch_size=128):
    assert loss_function in loss_functions, f"Loss function {loss_function} not implemented"
    assert y_pred.shape[2] == len(depths), "Number of layers to be validated don't match with the prediction-dimensions"
    assert len(y_true) == len(y_pred), "Two dataset don't have the same length"

    true_dataset = InputDataset(torch.from_numpy(y_true))
    true_dataloader = DataLoader(true_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    pred_dataset = InputDataset(torch.from_numpy(y_pred))
    pred_dataloader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    loss_fnc = loss_functions[loss_function]

    for _y_true, _y_pred in zip(true_dataloader, pred_dataloader):
        losses = []
        for i, depth in enumerate(depths):
            #import IPython ; IPython.embed() ; exit(1)
            #print(_y_pred.shape, loss_fnc)
            loss = loss_fnc(_y_true[:,:,depth].cpu().detach().numpy(), _y_pred[:,:,i].cpu().detach().numpy())
            losses.append(loss)

    return np.array(losses)
    
    
    
    
#import IPython ; IPython.embed() ; exit(1)
