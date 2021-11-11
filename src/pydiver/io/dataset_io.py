from typing import Any, Dict
from pathlib import Path
import numpy as np
from kedro.io import AbstractDataSet
import h5py
import torch
import torch.nn as nn

from pydiver.models.lstm import STLSTM

model_keys = dict(
    STLSTM=nn.DataParallel(STLSTM(1,8))
    )

class NumpyDataSet(AbstractDataSet):
    
    def __init__(self, filepath: str):
        self._filepath = filepath

    def _load(self) -> np.ndarray:
        return np.load(self._filepath)

    def _save(self, data: np.ndarray) -> None:
        np.save(self._filepath, data)
        return None

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath)
    

class TorchModel(AbstractDataSet):
    
    def __init__(self, filepath: str, load_args: Dict[str, Any] = dict(model="STLSTM", device="cuda")):
        self._filepath = filepath
        
        model_name = load_args['model']
        self._model = model_keys[model_name]
        
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not load_args["device"]=="cpu" else "cpu"

    def _load(self):
        self._model.load_state_dict(torch.load(self._filepath, map_location=self._device))

        return self._model

    def _save(self, model) -> None:
        #import IPython ; IPython.embed() ; exit(1)
        torch.save(model, self._filepath)
        return None

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath)

class OnnxModel(AbstractDataSet):
    
    def __init__(self, filepath: str, load_args: Dict[str, Any] = dict(model="STLSTM", device="cuda")):
        self._filepath = filepath
        
        model_name = load_args['model']
        self._model = model_keys[model_name]
        
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not load_args["device"]=="cpu" else "cpu"

    def _load(self):
        self._model.load_state_dict(torch.load(self._filepath, map_location=self._device))
        return self._model

    def _save(self, model) -> None:
        torch.save(model, self._filepath)
        return None

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath)

    
    
    
class H5PyDataSet(AbstractDataSet):
    
    def __init__(self, filepath: str):
        self._filepath = filepath
        
        if not self._exists():
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            f = h5py.File(filepath)
            f.close()
            
    def _load(self):
        return self._filepath

    def _save(self, data: Dict[str, h5py._hl.files.File]) -> None:
        f = h5py.File(self._filepath, 'r+')
        print(self._filepath)
        for key, value in data.items():
            
            try:
                del f[key]
            except KeyError:
                print(f"Overrides old dataset with key {key}")
            
            f.create_dataset(key, data=value)
        f.close()

        return None

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath)
    
    def _exists(self) -> bool:
        return Path(self._filepath).exists()
