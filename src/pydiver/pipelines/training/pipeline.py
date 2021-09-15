import os 
from kedro.pipeline import Pipeline, node, pipeline
import kedro
import numpy as np
import h5py
import torch
import yaml
from kedro.config import ConfigLoader

from .utils import train


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def create_pipeline(**kwargs):

    model_eval_pipe = Pipeline(
        [
            node(
                func=train,
                inputs=["X_train", "Y_train"],
                outputs="models",
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