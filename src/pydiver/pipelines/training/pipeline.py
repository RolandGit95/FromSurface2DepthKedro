import os 
from kedro.pipeline import Pipeline, node, pipeline
#import kedro
#import numpy as np
import re
#import h5py
from tqdm import tqdm
import torch
import torch.nn as nn
import yaml

from pytorch_pfn_extras.config import Config
from .CONFIG_TYPES import CONFIG_TYPES
from pydiver.models import lstm


def train(dataset_X, dataset_Y, params):
    #from .pl import 
    global cfg
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

    #import IPython ; IPython.embed() ; exit(1)
    cfg = config(dataset_X['X_train_00'], dataset_Y['Y_train_00'], params)

    model = DiverModule()
    datamodule = DataModule()
    #logger = pl_loggers.WandbLogger(name=cfg["/name"], project=cfg["/project"], save_dir="logs/")

    trainer = pl.Trainer(gpus=1,
                         max_epochs=cfg['/globals']['max_epochs'],
                         progress_bar_refresh_rate=1,
                         callbacks=[
                             #LearningRateMonitor(logging_interval='epoch'), 
                             #EarlyStopping(monitor='val_acc', patience=3, verbose=True, mode='max'),
                             ModelCheckpoint(monitor='val_loss', 
                                             dirpath="data/06_models/regimeB", 
                                             filename=cfg["/name"] + '{epoch}', 
                                             verbose=True)
                             ],
                         #logger=logger,     
                         )
    #import IPython ; IPython.embed() ; exit(1)

    trainer.fit(model, datamodule)
    #import IPython ; IPython.embed() ; exit(1)
    #return model
    return {cfg["/name"]:model.model.state_dict()}

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

    cfg = config(dataset_X['X_train_00'], dataset_Y['Y_train_00'], params)

    model = nn.DataParallel(cfg['/model'])
    loss_fnc = cfg['/loss']
    val_loss_fnc = cfg['/loss']
    optimizer = cfg['/optimizer']

    output_length = len(cfg['/dataset']['depths'])
    for epoch in range(cfg['/globals']['max_epochs']): 
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


def create_pipeline(**kwargs):

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
