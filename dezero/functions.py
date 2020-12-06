import numpy as np
from dezero.core import Variable, Function


class Sin(Function):

    def forward(self, x):
        return np.sin(x)

    def backward(self, gy):
        x = self.inputs[0].data
        return np.cos(x) * gy

class Cos(Function):

    def forward(self, x):
        return np.cos(x)

    def backward(self, gy):
        x = self.inputs[0].data
        return -np.sin(x) * gy

class Reshape(Function):

    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        return np.reshape(x, self.shape)

    def backward(self, gy):
        return reshape(gy, self.x_shape)

class Transpose(Function):

    def forward(self, x):
        return np.transpose(x)
    
    def backward(self, gy):
        return Transpose()(gy)


def sin(x):
    return Sin()(x)

def cos(x):
    return Cos()(x)

def reshape(x, shape):
    return Reshape(shape)(x)

def transpose(x):
    return Transpose()(x)