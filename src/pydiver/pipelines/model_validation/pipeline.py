from kedro.pipeline import Pipeline, node
import numpy as np
from .utils import validate, predict
import h5py
import torch
from pydiver.models.lstm import STLSTM

model_keys = dict(
    STLSTM=STLSTM
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prediction_io(X, models_dataset, kwargs):   
    for partition_id, partition_load_func in models_dataset.items(): 
        if partition_id == kwargs['name']:
            print("load model")
            model = partition_load_func()  
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not kwargs["device"]=="cpu" else "cpu"
    y_preds = predict(model, X, depths=kwargs["depths"], device=device)

    #print(y_preds)
    return {kwargs['name']: y_preds}


#def validation_io(filepath_true, filepath_pred, kwargs):
#    y_true = np.array(h5py.File(filepath_true, 'r')['Y'])
#    y_pred = np.array(h5py.File(filepath_pred, 'r')[kwargs['model']])
    
#    loss = validate(y_true, y_pred, loss=kwargs['loss'], depths=kwargs['depths'])
    
 #   return {kwargs['name']: loss}


def create_pipeline(regime="A", **kwargs):
    return Pipeline(
        [
            node(
                func=prediction_io,
                inputs=[f"regime_{regime}_X_test", f"regime_{regime}_models", "params:prediction"],
                outputs=f"regime_{regime}_pred",
                name="prediction_node",
            ),  
            
            #node(
            #    func=validation_io,
            #    inputs=[f"regime_{regime}_test", f"regime_{regime}_pred", "params:validation"],
            #    outputs=f"regime_{regime}_loss",
            #    name="validation_node",
            #),    
        ]
    )