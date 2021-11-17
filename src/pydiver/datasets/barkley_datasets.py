import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset

# %%


class BarkleyDataset(Dataset):
    # filenames of the input (X) and target data (y)

    def __init__(self, X, Y, train: bool = True, depths=[0, 1, 2], time_steps=[0, 1, 2], max_length=-1, dataset_idx=0) -> None:
        """
        Parameters
        ----------
        root : str
            folder in which the data is stored. In this file there have to be 
            the files X.npy and Y.npy for training and validation.
        train : bool, optional
            dataset for training or testing? If training,
            the input and target get rotated randomly in __getitem__
            to increase the diversity of the data for training. In the 
            validation mode this rotation is not applied.
            The default is True.
        depths : int, optional
            Layers in depth, which the network should predict. 
            The default is [0,1,2].
        time_steps : int, optional
            Number of time steps the input should have.
            The default is 32.
        max_length: int, optional
            Maximal number of example in the dataset, if -1 the original size will be used.
            The default is -1.
        num_datasets: int, optional
            Number of sub-datasets should be used for training, because of the size of the files they are splitted,
            while one contains 2048*6 examples.

        Returns
        -------
        None.
        """
        super(BarkleyDataset, self).__init__()

        self.train = train  # training set or test set
        #self.depth = depth

        self.max_length = max_length

        self.depths = np.array(depths)
        max_depth = max(depths)

        print(self.depths)

        self.time_steps = np.array(time_steps)
        #max_time_steps = max(time_steps)

        #self.time_steps = time_steps

        # This transformation function will be applied on the data if it is called in __getitem__
        self.transform = lambda data: (data.float()+127)/255.
        self.target_transform = lambda data: (data.float()+127)/255.

        # print(X)

        self.X = torch.from_numpy(X[:self.max_length])[:, self.time_steps]
        self.y = torch.from_numpy(Y[:self.max_length])[:, :, self.depths]

    # def setData(self, X, Y):
    #    self.X = torch.tensor(X[:self.max_length])[:,self.time_steps]
    #    self.y = torch.tensor(Y[:self.max_length])[:,:,self.depths]

    def __getitem__(self, idx: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (time-series at the surface, dynamic at time=0 till depth=self.depth)
                shapes: ([N,T,1,120,120], [N,1,D,120,120]), T and D are choosen in __init__, 
                The value for N depends if it is the training- or validaton-set.
        """

        # transform data of type int8 to float32 only at execution time to save memory
        X, y = self.transform(self.X[idx]), self.target_transform(self.y[idx])

        # Training data augmentation (random rotation of 0,90,180 or 270 degree)
        if self.train:
            k = np.random.randint(0, 4)
            X = torch.rot90(X, k=k, dims=[2, 3])
            y = torch.rot90(y, k=k, dims=[2, 3])

        return {'X': X, 'y': y}

    def __len__(self):
        try:
            l = len(self.X)
        except:
            l = 0
        return l

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        body.append("Max. depth: {}".format(max(self.depths)))
        body.append("Number of time-steps: {}".format(self.time_steps))
        body += self.extra_repr().splitlines()
        lines = [head]
        return '\n'.join(lines)

    def setMode(self, train=True):
        """
        Parameters
        ----------
        train : bool, optional
            dataset for training or testing? If train=True,
            the input and target get rotated randomly in __getitem__
            to increase the diversity of the data for training. In the 
            validation mode (train=False) this rotation is not applied.
            The default is True.
        """
        self.train = train

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

# %%


class InputDataset(Dataset):
    def __init__(self, data, time_steps=None, transform=lambda data: (data.float()+127.)/255.):
        self.transform = transform
        self.time_steps = np.array(time_steps)

        if isinstance(time_steps, type(None)):
            self.data = data[:,:]
        else:
            self.data = data[:,time_steps]
    def __getitem__(self, index):
        if self.transform:
            data = self.transform(self.data[index])
            return data
        else:
            data = self.data[index]
            return data

    def __len__(self):
        return len(self.data)


# %%

class TestDataset(Dataset):
    def __init__(self, X, Y):
        #super(TestDataset, self).__init__(root='')
        self.X = X
        self.Y = Y

        self.transform = lambda data: (torch.from_numpy(data).float()+127.)/255.

    def __getitem__(self, idx):
        X, y = self.transform(self.X[idx]), self.transform(self.Y[idx])
        return {'X': X, 'y': y}

    def __len__(self):
        return len(self.X)
