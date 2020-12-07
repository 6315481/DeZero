import unittest
import numpy as np
from numpy.testing import assert_array_equal
from dezero.utils import sum_to, reshape_sum_backward

class tset_utils(unittest.TestCase):

    def test_sum_to(self):
        y = np.ones((3, 4, 5, 6))

        self.assertEqual(sum_to(y, (5, 6)).shape, (5, 6))
        self.assertEqual(sum_to(y, (5, 1)).shape, (5, 1))

    def test_reshape_sum_backward(self):

        gy = np.ones((3, 4))
        self.assertEqual(reshape_sum_backward(gy, (2, 3, 4, 2), axis = (0, 3), keepdims=False).shape, (1, 3, 4, 1))
        
        gy = np.array(1.0)
        self.assertEqual(reshape_sum_backward(gy, (3, 4), axis=(0, 1), keepdims=False).shape, (1, 1))

        gy = np.array(1.0)
        self.assertEqual(reshape_sum_backward(gy, (3, 4), axis=None, keepdims=False).shape, ())