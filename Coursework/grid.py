'''
Contains the Grid class.
'''

import numpy as np
from geometry import Geometry

class Grid(object):
    '''
    Grid class represents a grid populated with cells and points.
    '''
    def __init__(self, vtk_data):
        self.vtk_data = vtk_data

    @property
    def number_of_points(self):
        '''Returns number of points in grid.'''
        return self.vtk_data.GetNumberOfPoints()

    @property
    def number_of_cells(self):
        '''Return number of cells in grid.'''
        return self.vtk_data.GetNumberOfCells()

    def set_grid(self):
        '''Reads through vtkDataObject and sets points and elements in the grid.'''
        self.points = np.zeros((self.number_of_points, 2))
        self.cells = []
        self.boundary_cells = []

        for i in range(self.number_of_points):
            point_x, point_y, point_z = self.vtk_data.GetPoint(i)
            self.points[i, 0] = point_x
            self.points[i, 1] = point_y

        for j in range(self.number_of_cells):
            cell = self.vtk_data.GetCell(j)
            point_ids = cell.GetPointIds()
            cell_type = cell.GetCellType()

            number_of_points_in_cell = point_ids.GetNumberOfIds()
            points_in_cell = [point_ids.GetId(k) for k in range(number_of_points_in_cell)]

            if cell_type > 3:
                self.cells.append(points_in_cell)
            else:
                self.boundary_cells.append(points_in_cell)

        self.points = np.matrix(self.points)
        self.cells = np.matrix(self.cells)
        

    def geometry(self, index):
        '''Returns a geometry object for a given cell in the grid.'''
        return Geometry(self.points, self.cells[index])

    def construct_system(self, f, sigma, integration_order):
        '''Returns the global matrix of the grid.'''
        matrix_a = np.zeros((self.number_of_points, self.number_of_points))
        vector_f = np.zeros((self.number_of_points, 1))

        for index, cell in enumerate(self.cells):
            stiffness_local, f_vec = self.geometry(index).system_elements(f, sigma, integration_order)

            for i in range(3):
                global_i = self.geometry(index).local2global(i)
                vector_f[global_i] += f_vec[i]
                for j in range(3):
                    global_j = self.geometry(index).local2global(j)
                    matrix_a[global_i, global_j] += stiffness_local[i, j]
                
        for index, cell in enumerate(self.boundary_cells):
            for point in range(len(cell)):
                matrix_a[cell[point], :] = 0
                matrix_a[cell[point], cell[point]] = 1.0
                vector_f[cell[point]] = 0
       
        return matrix_a, vector_f
