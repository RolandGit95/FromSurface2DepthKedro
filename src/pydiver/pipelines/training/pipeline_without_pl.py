import os 
#import kedro
#import numpy as np
import re
#import h5py
from tqdm import tqdm
import torch
import torch.nn as nn
import yaml

from pydiver.models import lstm
from pydiver.datasets import barkley_datasets




def train_without_pl(dataset_X, dataset_Y, params):
    global cfg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files_X, files_Y = list(dataset_X.keys()), list(dataset_Y.keys())

    for name in files_X:
        m = re.search(r'train_\d+$', name)#.group()
        if isinstance(m, (type(None))):
            files_X.remove(name)

    for name in files_Y:
        m = re.search(r'train_\d+$', name)#.group()
        if isinstance(m, (type(None))):
            files_Y.remove(name)
    files_X.sort(), files_Y.sort()

    X = dataset_X['X_train_00']
    y = dataset_X['Y_train_00']

    dataset = barkley_datasets.BarkleyDataset(X, y, depths=[31], time_steps=[0,5,11,16,21,26,31])

    model = nn.DataParallel(lstm.STLSTM())
    loss_fnc = nn.MSELoss()
    val_loss_fnc = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4)

    output_length = 1
    for epoch in range(8): 
        print(f'Epoch number {epoch}')

        train_loader, val_loader = cfg['/dataloader']['train'], cfg['/dataloader']['val']
            
        val_loader_iter = iter(val_loader)
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            model.zero_grad()
            optimizer.zero_grad()

            X = data['X'].to(device)
            y = data['y'].to(device)

            outputs = model(X, max_depth=output_length)

            loss = 0.0
            loss += loss_fnc(y, outputs) # [depths,batch,features=1,:,:]

            outputs = outputs.detach()

            loss.backward()
            optimizer.step()      

            if i%10==0:
                try:
                    data = next(val_loader_iter)
                    X_val, y_val = data['X'], data['y']
                except StopIteration:
                    val_loader_iter = iter(val_loader)
                    data = next(val_loader_iter)
                    X_val, y_val = data['X'], data['y']
                X_val = X_val.to(device)
                y_val = y_val.to(device)

                with torch.no_grad():
                    val_outputs = model(X_val, max_depth=output_length)
                    val_loss = val_loss_fnc(y_val, val_outputs)

    return {cfg["/name"]:model.state_dict()}  


def create_pipeline_without_pl(**kwargs):

    model_eval_pipe = Pipeline(
        [
            node(
                func=train_without_pl,
                inputs=["X_train", "Y_train", "params:training"],
                outputs="models",
                name="training_node",
            ),  
        ]
    )

    return model_eval_pipe
