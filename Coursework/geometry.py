'''
This module contains the Geometry class.
'''
from p1_basis import p1, grad_p1
from numpy.linalg import det, inv
import numpy as np
from quadrature_triangle import tri_gauss_points, basis_transformation
from time import time

class Geometry(object):
    '''
    Contains methods concerning an individual element of
    a grid. Allows for transformation calculations.
    '''
    def __init__(self, points, element_point_ids):
        self.points = points
        self.element_point_ids = element_point_ids
        self.element_x, self.element_y = self.global_points()
        self.bias = self.get_bias()


    def local2global(self, local_index):
        '''Returns corresponding global index from local index'''
        if local_index < 3:
            return self.element_point_ids[0, local_index]

    def global_points(self):
        '''Returns corresponding point coordinates in the global
           basis given a local index'''
        global_x = []
        global_y = []
        
        for i in range(3):
            points = self.points[self.local2global(i)]
            global_x.append(points.item(0))
            global_y.append(points.item(1))
        
        return global_x, global_y

    def get_bias(self):
        '''Returns the bias from the transformation of coordinates'''
        return np.matrix([self.element_x[0], self.element_y[0]]).T

    @property
    def jacobian(self):
        '''Returns the Jacobian matrix of local to global.'''
        return np.matrix([[self.element_x[1] - self.element_x[0], self.element_x[2] - self.element_x[0]],
                        [self.element_y[1] - self.element_y[0], self.element_y[2] - self.element_y[0]]])

    @property
    def integration_element(self):
        '''Returns the absolute determinant of Jacobian.'''
        return abs(det(self.jacobian))

    @property
    def inverse_transpose_jacobian(self):
        '''Returns the inverse Jacobian transposed.'''
        return inv(self.jacobian).T


    def a_element(self, sigma, integration_order):
        stiffness_local = np.zeros((3, 3))
        area = 1./2 * self.integration_element
        quad_points = tri_gauss_points(integration_order)


        for i in range(3):
            global_grad_p1_i = self.inverse_transpose_jacobian * grad_p1(i)
            
            for j in range(3):
                global_grad_p1_j = self.inverse_transpose_jacobian * grad_p1(j)

                a = 0
                for k in range(len(quad_points)):
                    global_quad_points = self.jacobian * np.matrix([[quad_points[k][0]], [quad_points[k][1]]]) + self.get_bias()
                    x = global_quad_points.item(0)
                    y = global_quad_points.item(1)
                    omega = quad_points[k][2]

                    a += omega * area * sigma(x, y) * np.dot(np.array(global_grad_p1_i)[:, 0], global_grad_p1_j)

                stiffness_local[i, j] = a

        return stiffness_local

    def f_element(self, f, integration_order):
        f_vec = np.zeros((3, 1))
        area = 1./2 * self.integration_element
        quad_points = tri_gauss_points(integration_order)

        for i in range(3):
            
            f_value = 0
            for k in range(len(quad_points)):
                global_points = self.jacobian * np.matrix([[quad_points[k][0]], [quad_points[k][1]]]) + self.get_bias()
                x = global_points.item(0)
                y = global_points.item(1)

                f_value += quad_points[k][2] * area * f(x, y) * p1(i, x, y)

            f_vec[i] = f_value
        return f_vec
