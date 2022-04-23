import math
import numpy as np


class QuadraticFunction:
    def __init__(self, q2, q1=None, q0=0):
        self.q2 = q2
        self.q1 = np.zeros(shape=(q2.shape[0], 1)) if q1 is None else q1
        self.q0 = q0
        pass

    def __call__(self, x, calc_hessian=False, calc_gradient=True):
        # x might be a matrix of vertical vectors of (x1,x2)
        # f = (x.T.dot(self.q2).dot(x) + self.q1.T.dot(x) + self.q0).item()
        f = np.einsum("ij,ji->i", x.T.dot(self.q2), x) + self.q1.T.dot(x) + self.q0
        if f.size == 1:
            f = f.item()
        if not calc_gradient:
            return f
        qqt = self.q2 + self.q2.T
        g = qqt.dot(x) + self.q1
        h = qqt if calc_hessian else None
        return f, g, h

    pass


quad_func1 = QuadraticFunction(np.array([[1, 0], [0, 1]], dtype=float))
q1_1000 = np.array([[1, 0], [0, 1000]], dtype=float)
quad_func2 = QuadraticFunction(q1_1000)
q_rotate = np.array([[math.sqrt(3) / 2, -0.5], [0.5, math.sqrt(3) / 2]], dtype=float)
quad_func3 = QuadraticFunction(q_rotate.T.dot(q1_1000).dot(q_rotate))


def rosenbrock(x, calc_hessian=False, calc_gradient=True):
    # x might be a matrix of vertical vectors of (x1,x2)
    x1, x2 = x[0], x[1]
    f = 100 * (x2 - x1 * x1) ** 2 + (1 - x1) ** 2
    if f.size == 1:
        f = f.item()
    if not calc_gradient:
        return f
    g = np.vstack([400 * x1 ** 3 - 400 * x1 * x2 + 2 * x1 - 2, 200 * (x2 - x1 * x1)])
    assert g.shape == x.shape
    if calc_hessian and x.shape[1] == 1:
        x1, x2 = x1.item(), x2.item()
        h = np.array([[1200 * x1 * x1 - 400 * x2 + 2, -400 * x1], [-400 * x1, 200]], dtype=float)
    else:
        h = None
    return f, g, h


def boyd(x, calc_hessian=False, calc_gradient=True):
    # x might be a matrix of vertical vectors of (x1,x2)
    x1, x2 = x[0], x[1]
    f1 = np.exp(x1 + 3 * x2 - 0.1)
    f2 = np.exp(x1 - 3 * x2 - 0.1)
    f3 = np.exp(-x1 - 0.1)
    f = f1 + f2 + f3
    if f.size == 1:
        f = f.item()
    if not calc_gradient:
        return f
    g2 = 3 * f1 - 3 * f2
    g = np.vstack([f1 + f2 - f3, g2])
    assert g.shape == x.shape
    if calc_hessian and x.shape[1] == 1:
        f1 = f1.item()
        f2 = f2.item()
        g2 = g2.item()
        h = np.array([[f, g2], [g2, 9 * f1 + 9 * f2]])
    else:
        h = None
    return f, g, h


class LinearFunction:
    def __init__(self, a):
        self.a = np.array(a, dtype=float).reshape(-1, 1)
        pass

    def __call__(self, x, calc_hessian=False, calc_gradient=True):
        f = self.a.T.dot(x)
        if f.size == 1:
            f = f.item()
        if not calc_gradient:
            return f
        g = self.a
        h = np.zeros(shape=(x.shape[0], x.shape[0])) if calc_hessian else None
        return f, g, h

    pass


linear_function = LinearFunction([1, 1])

