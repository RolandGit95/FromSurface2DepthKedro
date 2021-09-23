#import os 
from kedro.pipeline import Pipeline, node, pipeline
#import kedro
#import numpy as np
#import h5py
import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from pytorch_pfn_extras.config import Config
from .CONFIG_TYPES import CONFIG_TYPES


class DiverModule(pl.LightningModule): 
    def __init__(self):    
        super(DiverModule, self).__init__()
        
        self.model = cfg['/model']
        self.loss = cfg['/loss']
        self.val_loss = cfg['/loss']

        self.output_length = len(cfg['/dataset']['depths'])

    def forward(self, input):
        return self.model(input, max_depth=self.output_length)
  
    def configure_optimizers(self): 
        return {"optimizer": cfg['/optimizer'], "lr_scheduler": cfg["/scheduler"], "monitor": "val_loss"}
        
    def training_step(self, train_batch, batch_idx): 
        X = train_batch['X']
        y = train_batch['y']
        
        y_pred = self.forward(X)
        loss = self.loss(y_pred, y)
        
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss 
    
    def validation_step(self, valid_batch, batch_idx): 
        X = valid_batch['X']
        y = valid_batch['y']
    
        y_pred = self.forward(X)
        loss = self.loss(y_pred, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)


class DataModule(pl.LightningDataModule):    
    def __init__(self):    
        super(DataModule, self).__init__()
        
    def train_dataloader(self):
        return cfg['/dataloader']['train']

    def val_dataloader(self):
        return cfg['/dataloader']['val']


def config(partition_fnc_X, partition_fnc_Y, params):
    CONFIG_TYPES['input_data'] = partition_fnc_X
    CONFIG_TYPES['true_output_data'] = partition_fnc_Y

    with open(params['config_file'], 'r') as file:
        pre_eval_cfg = yaml.safe_load(file)
    
    return Config(pre_eval_cfg, types=CONFIG_TYPES)


def train(dataset_X, dataset_Y, params):
    global cfg

    cfg = config(dataset_X['X_train_00'], dataset_Y['Y_train_00'], params)

    model = DiverModule()
    datamodule = DataModule()
    logger = pl_loggers.WandbLogger(name=cfg["/name"], project=cfg["/project"], save_dir="logs/")

    trainer = pl.Trainer(gpus=1,
                         max_epochs=cfg['/globals']['max_epochs'],
                         progress_bar_refresh_rate=1,
                         callbacks=[
                             #LearningRateMonitor(logging_interval='epoch'), 
                             #EarlyStopping(monitor='val_acc', patience=3, verbose=True, mode='max'),
                             ModelCheckpoint(monitor='val_loss', 
                                             dirpath="data/06_models/regimeB", 
                                             filename=cfg["/name"] + '{epoch:02d}', 
                                             auto_insert_metric_name=False)
                             ],
                         logger=logger,     
                         )

    trainer.fit(model, datamodule)

    #return {cfg["/name"]:model.model}


def create_pipeline(**kwargs):

    model_eval_pipe = Pipeline(
        [
            node(
                func=train,
                inputs=["X_train", "Y_train", "params:training"],
                outputs=None,#"models",
                name="training_node",
            ),  
        ]
    )

    return model_eval_pipe







    """
    if isinstance(multi_model_eval, dict):
        multi_model_eval_pipes = Pipeline([])
        
        for name, depths in zip(multi_model_eval['model_names'], multi_model_eval['all_depths']):
            d = {'name':name, 'depths':depths, 'device':'cuda'}#, **multi_model_eval}

            def generate_param_node(param_to_return):
                def generated_data_param():
                    return param_to_return
                return generated_data_param

            pipeline_key = f'pipeline_{name}'

            multi_model_eval_pipes += Pipeline([
                node(
                    generate_param_node(d), 
                    inputs=None,
                    outputs=pipeline_key
                )
            ])

            multi_model_eval_pipes += pipeline(
                model_eval_pipe,
                inputs={f"regime_{regime}_X_test":f"regime_{regime}_X_test", 
                        f"regime_{regime}_models":f"regime_{regime}_models"},
                outputs={f"regime_{regime}_pred"},
                parameters={"params:prediction":pipeline_key},
                namespace=pipeline_key,
            )

        return multi_model_eval_pipes

    else: return model_eval_pipe
    """