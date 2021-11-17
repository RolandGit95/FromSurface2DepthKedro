import numpy as np
import re
import torch
from torch.utils.data import DataLoader
from pydiver.datasets import barkley_datasets
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error

loss_functions = dict(
    mae=lambda a, b: np.mean(np.abs(a-b)),
    mse=mean_squared_error,
)


@torch.no_grad()
def predict(model, data, depths=[0, 1, 2], time_steps=[0, 1, 2], batch_size=8, device="cuda"):
    num_layers = len(depths)

    model = model.to(device)

    if isinstance(time_steps, str):
        time_steps_int = [int(i) for i in re.findall(r'\d+', time_steps)]
    else:
        time_steps_int = time_steps

    dataset = barkley_datasets.InputDataset(torch.from_numpy(data), time_steps=time_steps_int)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def transform(data): return (data.cpu().detach().numpy()*255-128).astype(np.int8)

    y_preds = []
    for data in tqdm(dataloader, total=len(dataloader)):
        #import IPython ; IPython.embed() ; exit(1)
        X = data.to(device)
        #import IPython ; IPython.embed() ; exit(1)
        y_pred = transform(model(X, max_depth=num_layers))

        y_preds.append(y_pred)

    y_preds = np.concatenate(y_preds, 0)

    #import IPython ; IPython.embed() ; exit(1)

    return y_preds


def validate(y_true, y_pred, depths=[0, 1, 2], loss_function="mae", batch_size=8):
    assert loss_function in loss_functions, f"Loss function {loss_function} not implemented"
    assert y_pred.shape[2] == len(depths), "Number of layers to be validated don't match with the prediction-dimensions"
    assert len(y_true) == len(y_pred), "Two dataset don't have the same length"

    true_dataset = barkley_datasets.InputDataset(torch.from_numpy(y_true))
    true_dataloader = DataLoader(true_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    pred_dataset = barkley_datasets.InputDataset(torch.from_numpy(y_pred))
    pred_dataloader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    loss_fnc = loss_functions[loss_function]

    LOSSES = []
    for _y_true, _y_pred in tqdm(zip(true_dataloader, pred_dataloader), total=len(true_dataloader)):
        losses = []

        #import IPython ; IPython.embed() ; exit(1)
        for i, depth in enumerate(depths):
            loss = loss_fnc(_y_true[:, :, i].cpu().detach().numpy(), _y_pred[:, :, i].cpu().detach().numpy())
            losses.append(loss)

        LOSSES.append(np.array(losses))
    LOSSES = np.array(LOSSES)

    means = np.mean(LOSSES, axis=0)

    print(f'LOSS: {means}')
    return means


#import IPython ; IPython.embed() ; exit(1)
