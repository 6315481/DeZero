import numpy as np
from abc import ABCMeta, abstractmethod
import dezero

class Variable:
    __array_priority__ = 200

    def __init__(self, x, name=None):
        if x is not None and not isinstance(x, np.ndarray):
            raise TypeError('{} is not supported'.format(type(x)))

        self.data = x
        self.grad = None
        self.name = name
        self.creator = None
        self.generation = 0
    def __repr__(self):
        return f'Variable({self.data})'

    def __len__(self):
        return len(self.data)
     
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def clear_grad(self):
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx
            
                if x.creator is not None:
                    add_func(x.creator)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)

    def transpose(self):
        return dezero.functions.transpose(self)
    
    def T(self):
        return dezero.functions.transpose(self)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype



class Function:

    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)

        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()



class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0]
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx

class Add(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        return x0 + x1
    
    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape:
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

class Sub(Function):
    def forward(self, x0, x1):
        return x0 - x1
    
    def backward(self, gy):
        return gy, -gy

class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1
    
    def backward(self, gy):
        x0, x1 = self.inputs
        return gy * x1, gy * x0

class Div(Function):
    def forward(self, x0, x1):
        return x0 / x1

    def backward(self, gy):
        x0, x1 = self.inputs
        return gy * (1/x1), gy * (-x0 / x1 ** 2)

class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self, x):
        y = x ** self.c
        return y
    
    def backward(self, gy):
        x = self.inputs[0]
        c = self.c
        return c * gy * (x ** (c-1))


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(x):
    if isinstance(x, Variable):
        return x
    return Variable(x)

def add(x0, x1):
    return Add()(x0, as_variable(as_array(x1)))

def sub(x0, x1):
    return Sub()(x0, as_variable(as_array(x1)))

def rsub(x0, x1):
    return Sub()(as_variable(as_array(x1)), x0)

def mul(x0, x1):
    return Mul()(x0, as_variable(as_array(x1)))

def div(x0, x1):
    return Div()(x0, as_variable(as_array(x1)))

def rdiv(x0, x1):
    return Div()(as_variable(as_array(x1)), x0)

def neg(x1):
    return Variable(as_array(-x1.data))

def pow(x, c):
    return Pow(c)(x)


def setup_variables():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow
    Variable.__neg__ = neg

