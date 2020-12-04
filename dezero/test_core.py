import unittest
import numpy as np
from dezero.core import Function, Variable, Square, Exp, Add


class TestCalcGraph(unittest.TestCase):

    def test_connection(self):
        x = Variable(np.array(1.0))
        A = Square()
        B = Exp()
        C = Square()

        a = A(x)
        b = B(a)
        y = C(b)

        self.assertEqual(y.creator, C)
        self.assertEqual(y.creator.inputs[0], b)
        self.assertEqual(y.creator.inputs[0].creator, B)
        self.assertEqual(y.creator.inputs[0].creator.inputs[0], a)
    


class TestGradient(unittest.TestCase):

    def test_gradient_square(self):
        x = Variable(np.array(1.0))
        A = Square()
        y = A(x)

        y.backward()
        self.assertEqual(x.grad.data, 2.0)

    def test_gradient_composition(self):
        x = Variable(np.array(0.5))
        A = Square()
        B = Exp()
        C = Square()

        a = A(x)
        b = B(a)
        y = C(b)

       
        y.backward()

        self.assertAlmostEqual(x.grad.data, 3.2974425)
    
    def test_gradient_add(self):
        x = Variable(np.array(2.0))
        f = Add()
        y = f(x, x)

        y.backward()
        self.assertEqual(x.grad.data, 2.0)

        x0 = Variable(np.array(2.0))
        x1 = Variable(np.array(1.0))
        f = Add()

        y = f(x0, x1)
        y.backward()
        self.assertEqual(x0.grad.data, 1.0)
        self.assertEqual(x1.grad.data, 1.0)

    def test_gradient_quater_function(self):
        x = Variable(np.array(2.0))
        a = Square()(x)
        y = Add()(Square()(a), Square()(a))
        y.backward()

        self.assertEqual(x.grad.data, 64.0)

    def test_gradient_with_constant(self):
        x = Variable(np.array(4.0))
        y = 2.0 * x
        y.backward()
        self.assertEqual(x.grad.data, 2.0)

        x = Variable(np.array(4.0))
        y = x / 2.0
        y.backward()
        self.assertEqual(x.grad.data, 0.5)

        x = Variable(np.array(2.0))
        y = 2.0 / x
        y.backward()
        self.assertEqual(x.grad.data, -0.5)

    def test_gradient_matyas(self):
   
        def matyas(x, y):
            z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
            return z
        x = Variable(np.array(1.0))
        y = Variable(np.array(1.0))
        z = matyas(x, y)
        z.backward()

        self.assertAlmostEqual(x.grad.data, 0.040)
        self.assertAlmostEqual(y.grad.data, 0.040)


class TestOperation(unittest.TestCase):

    def test_basic_operation(self):

        #Variable and Variable 
        a = Variable(np.array(3.0))
        b = Variable(np.array(2.0))
        self.assertEqual((a * b).data, np.array(6.0))
        self.assertEqual((a + b).data, np.array(5.0))
        self.assertEqual((a - b).data, np.array(1.0))
        self.assertEqual((a / b).data, np.array(3.0/2.0))
        
        #Variable and Scalar
        b = 2.0
        self.assertEqual((a * b).data, np.array(6.0))
        self.assertEqual((a + b).data, np.array(5.0))
        self.assertEqual((a - b).data, np.array(1.0))
        self.assertEqual((a / b).data, np.array(3.0/2.0))
    
        #Scalar and Variable
        self.assertEqual((b * a).data, np.array(6.0))
        self.assertEqual((b + a).data, np.array(5.0))
        self.assertEqual((b - a).data, np.array(-1.0))
        self.assertEqual((b / a).data, np.array(2.0/3.0))
        
        #negative
        self.assertEqual((-a).data, np.array(-3.0))

        #pow
        self.assertEqual((a ** 2).data, np.array(9.0))
   
class TestException(unittest.TestCase):
    
    def test_type_error(self):
        with self.assertRaises(TypeError):
            Variable(1.0)


class TestProperty(unittest.TestCase):

    def test_property(self):
        x = Variable(np.array([[1, 2, 3], [1, 2, 3]]))
        self.assertEqual(x.shape, (2, 3))
        self.assertEqual(len(x), 2)


