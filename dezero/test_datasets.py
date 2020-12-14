import numpy as np
import unittest
from dezero.datasets import Dataset


class TestDatasets(Dataset):

    def prepare(self):
        self.data = np.arange(100)
        self.label = np.arange(100)

class TestDatasetsModule(unittest.TestCase):

    def test_getitem(self):
        
        test_dataset = TestDatasets()
        for i in range(100):
            self.assertEqual((i, i), test_dataset[i])

    def test_len(self):
        test_dataset = TestDatasets()
        self.assertEqual(len(test_dataset), 100)
