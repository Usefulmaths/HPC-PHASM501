'''
This module contains the Geometry class.
'''

from field_functions import p1, grad_p1, sigma, f
from numpy.linalg import det, inv
import numpy as np
from quadrature_triangle import tri_gauss_points, basis_transformation

class Geometry(object):
    '''
    Contains methods concerning an individual element of
    a grid. Allows for transformation calculations.
    '''
    def __init__(self, points, element_points):
        self.points = points
        self.element_points = element_points

    def local2global(self, local_index):
        '''Returns corresponding global index from local index'''
        if local_index < 3:
            return self.element_points[0, local_index]

    def global_point(self, index):
        return self.points[self.local2global(index)]

    @property
    def jacobian(self):
        '''Returns the Jacobian matrix of local to global.'''
        x_0 = self.global_point(0).item(0)
        x_1 = self.global_point(1).item(0)
        x_2 = self.global_point(2).item(0)

        y_0 = self.global_point(0).item(1)
        y_1 = self.global_point(1).item(1)
        y_2 = self.global_point(2).item(1)

        return np.matrix([[x_1 - x_0, x_2 - x_0], [y_1 - y_0, y_2 - y_0]])

    @property
    def integration_element(self):
        '''Returns the absolute determinant of Jacobian.'''
        return abs(det(self.jacobian))

    @property
    def inverse_transpose_jacobian(self):
        '''Returns the inverse Jacobian transposed.'''
        return inv(self.jacobian).T


    def a_ij(self, i, j, sigma, integration_order):
        area = 1./2 * self.integration_element
        global_grad_p1_i = self.inverse_transpose_jacobian * grad_p1(i)
        global_grad_p1_j = self.inverse_transpose_jacobian * grad_p1(j)

        x_0 = self.global_point(0).item(0)
        x_1 = self.global_point(1).item(0)
        x_2 = self.global_point(2).item(0)

        y_0 = self.global_point(0).item(1)
        y_1 = self.global_point(1).item(1)
        y_2 = self.global_point(2).item(1)

        quad_points = tri_gauss_points(integration_order)
        b = np.matrix([x_0, y_0]).T

        a = 0
        for k in range(len(quad_points)):
            global_points = self.jacobian * np.matrix([[quad_points[k][0]], [quad_points[k][1]]]) + b
            x = global_points.item(0)
            y = global_points.item(1)

            a += quad_points[k][2] * area * sigma(x, y) * np.dot(np.array(global_grad_p1_i)[:, 0], global_grad_p1_j)

        return a

    def a_ij(self, sigma, integration_order):
        stiffness_local = np.zeros((3, 3))

        for i in range(3):
            for j in range(3):
                area = 1./2 * self.integration_element
                global_grad_p1_i = self.inverse_transpose_jacobian * grad_p1(i)
                global_grad_p1_j = self.inverse_transpose_jacobian * grad_p1(j)

                x_0 = self.global_point(0).item(0)
                x_1 = self.global_point(1).item(0)
                x_2 = self.global_point(2).item(0)

                y_0 = self.global_point(0).item(1)
                y_1 = self.global_point(1).item(1)
                y_2 = self.global_point(2).item(1)

                quad_points = tri_gauss_points(integration_order)
                b = np.matrix([x_0, y_0]).T

                a = 0
                for k in range(len(quad_points)):
                    global_points = self.jacobian * np.matrix([[quad_points[k][0]], [quad_points[k][1]]]) + b
                    x = global_points.item(0)
                    y = global_points.item(1)

                    a += quad_points[k][2] * area * sigma(x, y) * np.dot(np.array(global_grad_p1_i)[:, 0], global_grad_p1_j)

                stiffness_local[i, j] = a

        return stiffness_local

    def f_element(self, f, integration_order):
        f_vec = np.zeros((3, 1))

        for i in range(3):
            area = 1./2 * abs(self.integration_element)

            x_0 = self.global_point(0).item(0)
            x_1 = self.global_point(1).item(0)
            x_2 = self.global_point(2).item(0)

            y_0 = self.global_point(0).item(1)
            y_1 = self.global_point(1).item(1)
            y_2 = self.global_point(2).item(1)

            quad_points = tri_gauss_points(integration_order)
            b = np.matrix([x_0, y_0]).T
            
            f_value = 0
            for k in range(len(quad_points)):
                global_points = self.jacobian * np.matrix([[quad_points[k][0]], [quad_points[k][1]]]) + b
                x = basis_transformation(quad_points[k][0], quad_points[k][1], x_0, x_1, x_2)
                y = basis_transformation(quad_points[k][0], quad_points[k][1], y_0, y_1, y_2)

                f_value += quad_points[k][2] * area * f(x, y) * p1(i, [x, y])

            f_vec[i] = f_value
        return f_vec
