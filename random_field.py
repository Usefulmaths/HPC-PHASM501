import numpy as np
from dolfin import *

class RandomDiffusionField(Expression):
    def __init__(self, m, n, element):

        self._rand_field = np.exp(-np.random.randn(m, n))
        self._m = m
        self._n = n
        self._ufl_element = element
        
    def eval(self, value, x):

        x_index = np.int(np.floor(self._m * x[0]))
        y_index = np.int(np.floor(self._n * x[1]))
        
        i = min(x_index, self._m - 1)
        j = min(y_index, self._n - 1)
        
        value[0] = self._rand_field[i, j]
        
    def value_shape(self):

        return (1, )

class RandomRhs(Expression):
    def __init__(self, m, n, element):

        self._rand_field = np.random.randn(m, n)
        self._m = m
        self._n = n
        self._ufl_element = element
        
    def eval(self, value, x):
        x_index = np.int(np.floor(self._m * x[0]))
        y_index = np.int(np.floor(self._n * x[1]))
        
        i = min(x_index, self._m - 1)
        j = min(y_index, self._n - 1)
        
        value[0] = self._rand_field[i, j]
        
    def value_shape(self):
        return (1, )


