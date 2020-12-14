import numpy as np
import unittest
from dezero.datasets import Dataset
from dezero.dataloaders import DataLoader

class TestDatasets(Dataset):

    def prepare(self):
        self.data = np.arange(100)
        self.label = np.arange(100)

class TestDataLoaderModuele(unittest.TestCase):

    def test_iterator(self):

        test_dataset = TestDatasets()
        test_dataloader = DataLoader(test_dataset, 10, True)

        for i, t in test_dataloader:
            self.assertEqual(i.shape, (10,))
            self.assertEqual(t.shape, (10,))

    
        

