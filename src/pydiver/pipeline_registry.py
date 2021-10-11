
"""Project pipelines."""
from typing import Dict

import kedro
from kedro.pipeline import Pipeline
from kedro.config import ConfigLoader

#from pydiver.pipelines import data_simulation as ds
#from pydiver.pipelines import data_processing as dp
from pydiver.pipelines import model_validation as mv
from pydiver.pipelines import training as tr
        
def register_pipelines() -> Dict[str, Pipeline]:

    registered_pipelines = {}

    #data_simulation_pipeline = ds.create_pipeline()
    model_validation_pipeline = mv.create_pipeline()
    #training_pipeline = tr.create_pipeline()
    training_pipeline_without_pl = tr.create_pipeline_without_pl()

    #registered_pipelines['ds'] = data_simulation_pipeline
    registered_pipelines['mv'] = model_validation_pipeline
    #registered_pipelines['tr'] = training_pipeline
    #registered_pipelines['tr+mv'] = training_pipeline + model_validation_pipeline

    registered_pipelines['tr_without_pl'] = training_pipeline_without_pl
    registered_pipelines['tr_without_pl+mv'] = training_pipeline_without_pl + model_validation_pipeline

    registered_pipelines['__default__'] = model_validation_pipeline


    #import IPython ; IPython.embed() ; exit(1)

    return registered_pipelines






