
import numpy as np
from dezero.core import Parameter
from dezero.layers import TestClass


test = TestClass()
test.p1 = Parameter(np.array(1.0))
test.p2 = Parameter(np.array(2.0))