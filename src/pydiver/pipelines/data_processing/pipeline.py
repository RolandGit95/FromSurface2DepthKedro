from kedro.pipeline import Pipeline, node
import numpy as np
from .utils import preprocess_cube, mergeCube#, split_data
import h5py


def process_io(filepath, process_function):
    dataset = h5py.File(filepath, 'r')
    modified_data = {}
    for key in dataset.keys():
        data = np.array(dataset[key])
        data = process_function(data)    
        modified_data[key] = data
    return modified_data

def process_cube_io(filepath):
    return process_io(filepath, preprocess_cube)


def process_cubes_6_sides_io(filepath):
    dataset = h5py.File(filepath, 'r')
    X, Y = [], []
    
    for key in dataset.keys():
        data = np.array(dataset[key])
        x, y = mergeCube(data)    

        X.append(x)
        Y.append(y)
        
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)

    return {'X':np.array(X), 'Y':np.array(Y)}


#def train_test_split(dataset):
#    data = {}
#    for partition_id, partition_load_func in dataset.items(): 
#        data[partition_id] = partition_load_func()
#
#    return split_data(data)
    

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=process_cube_io,
                inputs=f"cubes",
                outputs=["X", "Y"],
                name="preprocess_cubes_node",
            ),        
        ]
    )