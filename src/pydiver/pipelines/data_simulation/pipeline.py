from kedro.pipeline import Pipeline, node
import numpy as np
from .utils import simulate_barkley, main

    

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=main,
                inputs=f"params:simulation",
                outputs=f"cubes",
                name="simulation_node",
            ),     
        ]
    )