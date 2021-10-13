import os 
from kedro.pipeline import Pipeline, node, pipeline
import kedro
import numpy as np
from .utils import predict, validate
import h5py
import torch
import torch.nn as nn
import re
import yaml
from kedro.config import ConfigLoader
from pydiver.models.lstm import STLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prediction_io(dataset_X_test, models_dataset, kwargs):   
    for partition_id, partition_load_func in models_dataset.items(): 
        if partition_id == kwargs['name']:
            print("load model")
            model = partition_load_func()  
            break
            
    files_X = list(dataset_X_test.keys())

    for name in files_X:
        m = re.search(r'test_\d+$', name)#.group()
        if isinstance(m, (type(None))):
            files_X.remove(name)

    files_X.sort()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not kwargs["device"]=="cpu" else "cpu"

    y_preds_dict = {}
    for test_data_file in files_X:
        y_preds = predict(model, dataset_X_test[test_data_file](), depths=kwargs["depths"], device=device, batch_size=kwargs['prediction']['batch_size'])

        y_preds_dict[kwargs['name'] + "_" + test_data_file] = y_preds
    return y_preds_dict


def validation_io(dataset_y_true, dataset_y_preds, kwargs):
    #dataset_y_true: y_test_xx
    #dataset_y_pred: [model_name]_X_test_[xx].npy

    if isinstance(kwargs['depths'], str):
        depths = kwargs['depths']
        kwargs['depths'] = [int(i) for i in re.findall(r'\d+',depths)]
        
    files_pred = list(dataset_y_preds.keys())
    files_true = list(dataset_y_true.keys())

    _files_pred = files_pred.copy()
    _files_true = files_true.copy()

    for name in _files_pred:
        m = re.search(r'test_[\d]+', name)#.group()
        if isinstance(m, type(None)):
            files_pred.remove(name)

    for name in _files_true:
        m = re.search(r'test_[\d]+', name)#.group()
        if isinstance(m, type(None)):
            files_true.remove(name)

    
    losses_dict = {}
    #import IPython ; IPython.embed() ; exit(1)
    for preds in files_pred:
        print(preds, kwargs['name'])
        
        if preds.startswith(kwargs['name'] + '_'):
        #if not isinstance(re.search(kwargs['name'], preds), type(None)):
            #import IPython ; IPython.embed() ; exit(1)
            losses = validate(dataset_y_true[files_true[0]](), 
                              dataset_y_preds[preds](), 
                              depths=kwargs['depths'], 
                              loss_function=kwargs['validation']['loss'], 
                              batch_size=kwargs['prediction']['batch_size'])

    losses_dict[kwargs['name'] + "_" + files_true[0]] = losses
    return losses_dict #{kwargs['name']: y_preds}

    #losses = validate(dataset_y_true['Y_test_00'], y_pred, depths=kwargs['depths'], loss_function=kwargs['loss'])

    #return {kwargs['name']:losses}


def create_pipeline(**kwargs):
    model_eval_pipe = Pipeline(
        [
            node(
                func=prediction_io,
                inputs=[f"X_test", f"models", "params:data_science"],
                outputs=f"pred",
                name="prediction_node",
            ),  

            node(
                func=validation_io,
                inputs=[f"Y_test", f"pred", "params:data_science"],
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