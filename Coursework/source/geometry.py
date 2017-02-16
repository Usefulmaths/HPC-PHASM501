'''
This module contains the Geometry class.
'''
import numpy as np
from numpy.linalg import inv
from .quadrature_triangle import tri_gauss_points
from .p1_basis import p1_basis, grad_p1

class Geometry(object):
    '''
    Contains methods concerning an individual element of
    a grid. Allows for transformation from local to global
    calculates the local assembly matrix and force vector.
    '''
    def __init__(self, points, element_point_ids):
        self.points = points
        self.element_point_ids = element_point_ids
        self.element_x, self.element_y = self.global_points()


    def local2global(self, local_index):
        '''Returns corresponding global index from local index'''
        if local_index < 3:
            return self.element_point_ids[local_index]

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
        element1 = self.element_x[1] - self.element_x[0]
        element2 = self.element_y[1] - self.element_y[0]
        element3 = self.element_x[2] - self.element_x[0]
        element4 = self.element_y[2] - self.element_y[0]

        return np.matrix([[element1, element2],
                          [element3, element4]])

    @property
    def integration_element(self):
        '''Returns the absolute determinant of Jacobian.'''
        det_j = (self.element_x[1] - self.element_x[0]) * \
                (self.element_y[2] - self.element_y[0]) - \
                (self.element_x[2] - self.element_x[0]) * \
                (self.element_y[1] - self.element_y[0])
        return abs(det_j)

    @property
    def inverse_transpose_jacobian(self):
        '''Returns the inverse Jacobian transposed.'''
        return inv(self.jacobian_transpose)


    def system_elements(self, func, sigma, integration_order):
        '''Computes the local assembly matrix and force vector
        by using the quadrature rule integration over the element'''

        assembly_local = np.zeros((3, 3))
        f_vec = np.zeros((3, 1))
        quad_points = tri_gauss_points(integration_order)

        for i in range(3):
            global_grad_p1_i = self.inverse_transpose_jacobian * grad_p1(i)

            for j in range(3):
                global_grad_p1_j = self.inverse_transpose_jacobian * grad_p1(j)

                a_value = 0
                f_value = 0

                for _, qdp in enumerate(quad_points):
                    global_quad_x = self.element_x[0] * p1_basis(0, qdp[0], qdp[1]) + \
                                    self.element_x[1] * p1_basis(1, qdp[0], qdp[1]) + \
                                    self.element_x[2] * p1_basis(2, qdp[0], qdp[1])

                    global_quad_y = self.element_y[0] * p1_basis(0, qdp[0], qdp[1]) + \
                                    self.element_y[1] * p1_basis(1, qdp[0], qdp[1]) + \
                                    self.element_y[2] * p1_basis(2, qdp[0], qdp[1])

                    a_value += qdp[2] * sigma(global_quad_x, global_quad_y) * \
                        np.dot(np.array(global_grad_p1_i)[:, 0], global_grad_p1_j)

                    f_value += qdp[2] * func(global_quad_x, global_quad_y) * \
                        p1_basis(i, global_quad_x, global_quad_y)

                assembly_local[i, j] = 1./2 * self.integration_element * a_value

            f_vec[i] = 1./2 * self.integration_element * f_value

        return assembly_local, f_vec
