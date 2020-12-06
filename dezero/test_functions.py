import unittest
import numpy as np
from numpy.testing import assert_array_equal
from dezero.functions import sin, cos, reshape, transpose
from dezero.core import Variable, Function


class TestFunctions(unittest.TestCase):

    def test_sin(self):
        x = Variable(np.array(2.0))
        y = sin(x)
        y.backward()
        self.assertEqual(y.data, np.sin(x.data))
        self.assertEqual(x.grad.data, np.cos(x.data))

    def test_cos(self):
        x = Variable(np.array(2.0))
        y = cos(x)
        y.backward()
        self.assertEqual(y.data, np.cos(x.data))
        self.assertEqual(x.grad.data, -np.sin(x.data))

    def test_reshape(self):
        x = Variable(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        y = reshape(x, (6,))
        y.backward()
        assert_array_equal(y.data, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
        assert_array_equal(x.grad.data, np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]))

    def test_reshape_method_of_variable(self):
        x = Variable(np.array([1, 2, 3, 4, 5, 6]))
        y = x.reshape(2, 3)
        y.backward()
        assert_array_equal(y.data, np.array([[1, 2, 3], [4, 5, 6]]))
        assert_array_equal(x.grad.data, np.array([1, 1, 1, 1, 1, 1]))

    def test_transpose(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = transpose(x)
        y.backward()
        assert_array_equal(y.data, np.transpose(x.data))
        assert_array_equal(x.grad.data, np.array([[1, 1, 1], [1, 1, 1]]))

    def test_transpose_method_of_variable(self):
        x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
        y = x.transpose()
        y.backward()
        assert_array_equal(y.data, np.transpose(x.data))
        assert_array_equal(x.grad.data, np.array([[1, 1, 1], [1, 1, 1]]))