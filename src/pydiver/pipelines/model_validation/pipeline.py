import os 
from kedro.pipeline import Pipeline, node, pipeline
import kedro
import numpy as np
from .utils import predict, validate
import h5py
import torch
import torch.nn as nn
import yaml
from kedro.config import ConfigLoader
from pydiver.models.lstm import STLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prediction_io(X, models_dataset, kwargs):   
    for partition_id, partition_load_func in models_dataset.items(): 
        if partition_id == kwargs['name']:
            print("load model")
            model = partition_load_func()  

            break
            
    #import IPython ; IPython.embed() ; exit(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not kwargs["device"]=="cpu" else "cpu"
    y_preds = predict(model, X['X_test_00'](), depths=kwargs["depths"], device=device)

    return {kwargs['name']: y_preds}


def validation_io(y_true_dataset, y_preds_dataset, kwargs):
    for partition_id, partition_load_func in y_preds_dataset.items(): 
        print(partition_id)
        names = [partition_id, os.path.splitext(partition_id)[0]]
        if kwargs['name'] in names:
            print(f"load predictions of model {kwargs['name']}")

            y_pred = partition_load_func()  

            break

    losses = validate(y_true_dataset['Y_test_00'], y_pred, depths=kwargs['depths'], loss_function=kwargs['loss'])

    return {kwargs['name']:losses}


def create_pipeline(**kwargs):

    model_eval_pipe = Pipeline(
        [
            node(
                func=prediction_io,
                inputs=[f"X_test", f"models", "params:prediction"],
                outputs=f"pred",
                name="prediction_node",
            ),  

            node(
                func=validation_io,
                inputs=[f"Y_test", f"pred", "params:validation"],
                outputs=f"loss",
                name="validation_node",
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