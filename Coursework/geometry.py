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


    @property
    def jacobian_transpose(self):
        '''Returns the Jacobian matrix of local to global.'''
        return np.matrix([[self.element_x[1] - self.element_x[0], self.element_y[1] - self.element_y[0]],
                        [self.element_x[2] - self.element_x[0], self.element_y[2] - self.element_y[0]]])

    @property
    def integration_element(self):
        '''Returns the absolute determinant of Jacobian.'''
        det = (self.element_x[1] - self.element_x[0]) * (self.element_y[2] - self.element_y[0]) - (self.element_x[2] - self.element_x[0]) * (self.element_y[1] - self.element_y[0])
        return abs(det)

    @property
    def inverse_transpose_jacobian(self):
        '''Returns the inverse Jacobian transposed.'''
        return inv(self.jacobian_transpose)


    def system_elements(self, f, sigma, integration_order):
        stiffness_local = np.zeros((3, 3))
        f_vec = np.zeros((3, 1))
        area = 1./2 * self.integration_element
        quad_points = tri_gauss_points(integration_order)


        for i in range(3):
            global_grad_p1_i = self.inverse_transpose_jacobian * grad_p1(i)
            
            for j in range(3):
                global_grad_p1_j = self.inverse_transpose_jacobian * grad_p1(j)

                a = 0
                f_value = 0

                for k in range(len(quad_points)):
                    x = self.element_x[0] * p1(0, 0, 0) + self.element_x[1] * p1(1, 0, 1) + self.element_x[2] * p1(2, 1, 0)
                    y = self.element_y[0] * p1(0, 0, 0) + self.element_y[1] * p1(1, 0, 1) + self.element_y[2] * p1(2, 1, 0)
            
                    omega = quad_points[k][2]

                    a += omega * area * sigma(x, y) * np.dot(np.array(global_grad_p1_i)[:, 0], global_grad_p1_j)
                    f_value += quad_points[k][2] * area * f(x, y) * p1(i, x, y)

                stiffness_local[i, j] = a

            f_vec[i] = f_value

        return stiffness_local, f_vec
