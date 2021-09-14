from kedro.pipeline import Pipeline, node, pipeline
import kedro
import numpy as np
from .utils import validate, predict
import h5py
import torch
import yaml
from kedro.config import ConfigLoader
from pydiver.models.lstm import STLSTM

model_keys = dict(
    STLSTM=STLSTM
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prediction_io(X, models_dataset, kwargs):   
    #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",kwargs)
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


def create_pipeline(regime="A", multi_model_eval=None):

    model_eval_pipe = Pipeline(
        [
            node(
                func=prediction_io,
                inputs=[f"regime_{regime}_X_test", f"regime_{regime}_models", "params:prediction"],
                outputs=f"regime_{regime}_pred",
                name="prediction_node",
            ),  
        ]
    )

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