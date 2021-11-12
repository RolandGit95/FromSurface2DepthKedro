import os
from kedro.pipeline import Pipeline, node
import re
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from pydiver.models import lstm
from pydiver.datasets import barkley_datasets
from .utils import get_sampler

os.environ["WANDB_MODE"] = "dryrun"

def train_without_pl(dataset_X, dataset_Y, params):

    print(params)

    wandb.init(project='FromSurface2DepthKedro',
               name=params['name'],
               config=params,
               reinit=True,
               dir="logs/")

    if isinstance(params['depths'], str):
        depths = params['depths']
        params['depths'] = [int(i) for i in re.findall(r'\d+', depths)]

    if isinstance(params['time_steps'], str):
        time_steps = params['time_steps']
        params['time_steps'] = [int(i) for i in re.findall(r'\d+', time_steps)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files_X, files_Y = list(dataset_X.keys()), list(dataset_Y.keys())

    for name in files_X:
        m = re.search(r'train_\d+$', name)  # .group()
        if isinstance(m, (type(None))):
            files_X.remove(name)

    for name in files_Y:
        m = re.search(r'train_\d+$', name)  # .group()
        if isinstance(m, (type(None))):
            files_Y.remove(name)
    files_X.sort(), files_Y.sort()

    X = dataset_X['X_train_00']()
    y = dataset_Y['Y_train_00']()

    #import IPython ; IPython.embed() ; exit(1)

    dataset = barkley_datasets.BarkleyDataset(X, y, depths=params['depths'], time_steps=params['time_steps'])

    model = nn.DataParallel(lstm.STLSTM(1, params['hidden_size'], 1)).to(device)
    loss_fnc = nn.MSELoss()
    val_loss_fnc = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])  # lr=params['training']['lr'])

    def get_lr():
        for param_group in optimizer.param_groups:
            return param_group['lr']

    callbacks = [ReduceLROnPlateau(optimizer, patience=512//5, factor=0.3, min_lr=1e-7, verbose=True)]

    wandb.watch(model, log="all", log_freq=32)

    train_sampler = get_sampler(len(dataset), val_split=params['training']['val_split'], train=True, shuffle=True, seed=params['seed'])
    val_sampler = get_sampler(len(dataset), val_split=params['training']['val_split'], train=False, shuffle=True, seed=params['seed'])

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=params['training']['batch_size'], num_workers=4, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=params['training']['batch_size'],  num_workers=4, sampler=val_sampler)

    val_loader_iter = iter(val_loader)

    output_length = len(params['depths'])
    for epoch in range(params['training']['max_epochs']):
        print(f'Epoch number {epoch}')

        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            model.zero_grad()
            optimizer.zero_grad()

            X = data['X'].to(device)
            y = data['y'].to(device)

            outputs = model(X, max_depth=output_length)

            loss = 0.0
            loss += loss_fnc(y, outputs)  # [depths,batch,features=1,:,:]

            outputs = outputs.detach()

            loss.backward()
            optimizer.step()

            wandb.log({"lr": get_lr()})

            if i % 4 == 0:
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

                for callback in callbacks:
                    callback.step(val_loss)

                wandb.log({"loss": loss, "val_loss": val_loss})

    return {params['name']: model.state_dict()}

def create_pipeline_without_pl(**kwargs):

    model_eval_pipe = Pipeline(
        [
            node(
                func=train_without_pl,
                inputs=["X_train", "Y_train", "params:data_science"],
                outputs="models",
                name="training_node",
            ),
        ]
    )
    return model_eval_pipe