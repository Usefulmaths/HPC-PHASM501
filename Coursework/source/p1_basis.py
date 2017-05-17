'''
This module contains the p1 basis functions
and their corresponding gradient.
'''
import numpy as np

def p1_basis(index, x_cord, y_cord):
    '''Returns the p1 basis functions given an x and y.'''
    if index == 0:
        return 1 - x_cord - y_cord
    elif index == 1:
        return x_cord
    elif index == 2:
        return y_cord

def grad_p1(index):
    '''Returns the gradients of the p1 basis functions
        (Independent of coordinates)'''
    if index == 0:
        return np.matrix([-1, -1]).T
    elif index == 1:
        return np.matrix([1, 0]).T
    elif index == 2:
        return np.matrix([0, 1]).T
        