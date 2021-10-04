import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

class DiverModule(pl.LightningModule): 
    def __init__(self):    
        super(DiverModule, self).__init__()
        
        self.model = nn.DataParallel(cfg['/model'])
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