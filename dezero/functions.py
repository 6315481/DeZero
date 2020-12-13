import numpy as np
import dezero.utils as utils
from dezero.core import Variable, Function, as_variable

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

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx

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


class Sigmoid(Function):

    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        return y
    
    def backward(self, gy):
        x = self.inputs[0]
        gx = gy * exp(-x) / (1 + exp(-x)) ** 2
        return gx

class Log(Function):

    def forward(self, x):
        y = np.log(x)
        return y
    
    def backward(self, gy):
        x = self.inputs[0]
        gx = gy * 1 / x
        return gx

class GetItem(Function):

    def __init__(self, slices):
        self.slices = slices
    
    def forward(self, x):
        y = x[self.slices]
        return y
    
    def backward(self, gy):
        x = self.inputs[0]
        f =  GetItemGrad(self.slices, x.shape)
        return f(gy)

class GetItemGrad(Function):

    def __init__(self, slices, in_shape):
        self.slices = slices
        self.in_shape = in_shape
    
    def forward(self, gy):
        gx = np.zeros(self.in_shape)
        np.add.at(gx, self.slices, gy)
        return gx
    
    def backward(self, ggx):
        return get_item(ggx, self.slices)


def get_item(x, slices):
    return GetItem(slices)(x)

def sin(x):
    return Sin()(x)

def cos(x):
    return Cos()(x)

def exp(x):
    return Exp()(x)

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

def linear(x, W, b):
    return matmul(x, W) + b

def sigmoid(x):
    return Sigmoid()(x)

def log(x):
    return Log()(x)

def softmax(x, axis=1):
    x = as_variable(x)
    y = exp(x)
    sum_y = sum(y, axis=1, keepdims=True)
    return y / sum_y


def softmax_cross_entropy(x, t):
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]
    p = softmax(x)
    log_p = log(p)
    tlog_p = log_p[np.arange(N), t.data]
    y = -1 * sum(tlog_p) / N
    return y
