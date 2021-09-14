from torch.utils.data import Dataset


# %%

class InputDataset(Dataset):
    def __init__(self, data, transform=lambda data:(data.float()+127.)/255.):
        self.data = data
        self.transform = transform

    def __getitem__(self , index):
        if self.transform:
            return self.transform(self.data[index])
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)


# %%

class TestDataset(Dataset):
    def __init__(self, ):
        super(TestDataset, self).__init__(root='') 
        
        self.transform = lambda data:(data.float()+127.)/255.

    def __getitem__(self, idx):
        return self.transform(self.X[idx]), self.transform(self.Y[idx])

    def __len__(self):
        return len(self.X)