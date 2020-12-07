import numpy as np
from dezero import Variable
import dezero.functions as F


np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

I, H, O = 1, 10, 1
W1 = Variable(0.01 * np.random.randn(I, H))
b1 = Variable(np.zeros(H))
W2 = Variable(0.01 * np.random.randn(H, O))
b2 = Variable(np.zeros(O))

def predict(x):
    x2 = F.sigmoid(F.linear(x, W1, b1))
    x2 = F.linear(x2, W2, b2)
    return x2


def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)

lr = 0.5
iters = 100000


for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y_pred, y)
    
    W1.clear_grad()
    b1.clear_grad()
    W2.clear_grad()
    b2.clear_grad()
    loss.backward()

    W1.data -= lr * W1.grad.data
    b1.data -= lr * b1.grad.data
    W2.data -= lr * W2.grad.data
    b2.data -= lr * b2.grad.data
    if i % 1000 == 0:
        print(loss)

