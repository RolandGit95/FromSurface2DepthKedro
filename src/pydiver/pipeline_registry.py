
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
    training_pipeline = tr.create_pipeline()

    #registered_pipelines['ds'] = data_simulation_pipeline
    registered_pipelines['mv'] = model_validation_pipeline
    registered_pipelines['tr'] = training_pipeline
    registered_pipelines['tr+mv'] = training_pipeline + model_validation_pipeline

    registered_pipelines['__default__'] = model_validation_pipeline

        #data_processing_pipeline_A = dp.create_pipeline(regime="A")
    #model_validation_pipeline_A = mv.create_pipeline(regime="A")

    #multi_model_validation_pipeline
    
    #data_simulation_pipeline_B = ds.create_pipeline(regime="B")
    #data_processing_pipeline_B = dp.create_pipeline(regime="B")
    #model_validation_pipeline_B = mv.create_pipeline(regime="B")
    #data_scient_pipeline = ds.create_pipeline()

    #import IPython ; IPython.embed() ; exit(1)

    return registered_pipelines

    #return {
        #"__default__": data_simulation_pipeline_A + data_processing_pipeline_A + model_validation_pipeline_A,
    #    "__default__": model_validation_pipeline_A,

        #"ds_A": data_simulation_pipeline_A,
        #"dp_A": data_processing_pipeline_A,
    #    "mv_A": model_validation_pipeline_A,

        #"ds_B": data_simulation_pipeline_B,
        #"dp_B": data_processing_pipeline_B,
        #"mv_B": model_validation_pipeline_B,
     #       }











