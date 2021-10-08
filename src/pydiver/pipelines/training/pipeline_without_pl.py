import os 
from kedro.pipeline import Pipeline, node
import re
from tqdm import tqdm
import torch
import torch.nn as nn

from pydiver.models import lstm
from pydiver.datasets import barkley_datasets
from .utils import get_sampler

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

    X = dataset_X['X_train_00']()
    y = dataset_Y['Y_train_00']()

    #import IPython ; IPython.embed() ; exit(1)


    dataset = barkley_datasets.BarkleyDataset(X, y, depths=params['depths'], time_steps=params['time_steps'])

    model = nn.DataParallel(lstm.STLSTM())
    loss_fnc = nn.MSELoss()
    val_loss_fnc = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'])

    output_length = 1
    for epoch in range(params['max_epochs']): 
        print(f'Epoch number {epoch}')

        train_sampler = get_sampler(len(dataset), val_split=params['val_split'], train=True, shuffle=True, seed=params['seed'])
        val_sampler = get_sampler(len(dataset), val_split=params['val_split'], train=False, shuffle=True, seed=params['seed'])

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'], num_workers=4, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'],  num_workers=4, sampler=val_sampler)
            
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

    return {params['name']:model.state_dict()}  


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
