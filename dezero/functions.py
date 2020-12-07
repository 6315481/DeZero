import numpy as np
import dezero.utils as utils
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

class Sum(Function):

    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        return x.sum(axis=self.axis, keepdims=self.keepdims)
    
    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy, self.x_shape, self.axis, self.keepdims)
        return broadcast_to(gy, self.x_shape)

class BroadCast(Function):

    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        gx = sum_to(gy, self.x_shape)
        return gx

class SumTo(Function):

    def __init__(self, shape):
        self.shape = shape
    
    def forward(self, x):
        self.x_shape = x.shape
        y = utils.sum_to(x, self.shape)
        return y
    
    def backward(self, gy):
        gx = broadcast_to(gy, self.x_shape)
        return gx
    
class MatMul(Function):

    def forward(self, x, W):
        y = x.dot(W)
        return y

    def backward(self, gy):
        x, W = self.inputs[0], self.inputs[1]
        gx = matmul(gy, W.transpose())
        gW = matmul(x.transpose(), gy)

        return gx, gW
        

def sin(x):
    return Sin()(x)

def cos(x):
    return Cos()(x)

def reshape(x, shape):
    return Reshape(shape)(x)

def transpose(x):
    return Transpose()(x)

def sum(x ,axis=None, keepdims=False):
    return Sum(axis, keepdims)(x)

def broadcast_to(x, shape):
    return BroadCast(shape)(x)

def sum_to(x, shape):
    return SumTo(shape)(x)

def matmul(x, W):
    return MatMul()(x, W)