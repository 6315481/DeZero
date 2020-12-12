import numpy as np
from dezero import Variable
from dezero import optimizers
import dezero.functions as F
import dezero.layers as L


def mean_squared_error(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)



np.random.seed(10)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
iters = 10000

model = L.MLP((10, 10, 10, 10, 1))
optimizer = optimizers.SGD(lr)
optimizer.setup(model)
for i in range(iters):
    y_pred = model(x)
    loss = mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()
    if i % 1000 == 0:
        print(loss)

