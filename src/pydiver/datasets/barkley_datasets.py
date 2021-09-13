from torch.utils.data import Dataset


# %%

class InputDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = lambda data:(data.float()+127)/255.

    def __getitem__(self , index):
         return self.transform(self.data[index])

    def __len__(self):
        return len(self.data)


# %%

class TestDataset(Dataset):
    def __init__(self, ):
        super(TestDataset, self).__init__(root='') 
        
        self.transform = lambda data:(data.float()+127)/255.

    def __getitem__(self, idx):
        return self.transform(self.X[idx]), self.transform(self.Y[idx])

    def __len__(self):
        return len(self.X)