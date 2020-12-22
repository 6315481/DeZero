import numpy as np

class Dataset:

    def __init__(self, train=True):
        self.train = train
        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, index):
        assert np.isscalar(index)

        if self.label is None:
            return self.data[index], None
        else:
            return self.data[index], self.label[index]

    
    def __len__(self):
        return len(self.data)
    
    def prepare(self):
        pass


class MNIST(Dataset):

    def prepare(self):
