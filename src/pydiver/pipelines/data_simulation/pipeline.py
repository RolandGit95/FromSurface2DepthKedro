from kedro.pipeline import Pipeline, node
import numpy as np
from .utils import simulate_barkley, main

    

def create_pipeline(regime="A", **kwargs):
    return Pipeline(
        [
            node(
                func=main,
                inputs=f"params:regime{regime}",
                outputs=f"regime_{regime}_cubes",
                name="simulation_of_barkley",
            ),     
        ]
    )