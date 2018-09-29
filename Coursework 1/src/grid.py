'''
Contains the Grid class.
'''
from multiprocessing import Process, Manager
import numpy as np
from scipy.sparse import lil_matrix
from .cell import Cell
from .geometry import Geometry

class Grid(object):
    '''
    Grid class represents a grid populated with cells and points.
    '''
    def __init__(self, vtk_data):
        self.vtk_data = vtk_data
        self.cells = []
        self.points = np.zeros((self.number_of_points, 2))

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
        for i in range(self.number_of_points):
            point_x, point_y, _ = self.vtk_data.GetPoint(i)
            self.points[i, 0] = point_x
            self.points[i, 1] = point_y

        for j in range(self.number_of_cells):
            cell = self.vtk_data.GetCell(j)

            cell_id = j
            point_ids = cell.GetPointIds()
            cell_type = cell.GetCellType()

            number_of_points_in_cell = point_ids.GetNumberOfIds()
            points_in_cell = [point_ids.GetId(k) for k in range(number_of_points_in_cell)]

            cell_obj = Cell(points_in_cell, cell_type, cell_id)

            self.cells.append(cell_obj)

        self.points = np.matrix(self.points)

    def geometry(self, index):
        '''Returns a geometry object for a given cell in the grid.'''
        return Geometry(self.points, self.cells[index].points)

    def worker(self, chunk, field_tuple, worker_number, return_dict):
        '''
        Computes the global assembly matrix and force vector for a given
        array of cell indices and adds them to a global dictionary.
        '''
        lil_matrix_a = lil_matrix((self.number_of_points, self.number_of_points), dtype=float)
        vector_f = np.zeros((self.number_of_points, 1))

        func, sigma, integration_order = field_tuple
        for cell_index in chunk:
            assembly_local, f_vec = \
                    self.geometry(cell_index).system_elements(func, sigma, integration_order)

            for i in range(3):
                global_i = self.geometry(cell_index).local2global(i)
                vector_f[global_i] += f_vec[i]
                for j in range(3):
                    global_j = self.geometry(cell_index).local2global(j)
                    if abs(assembly_local[i, j]) > 1e-10:
                        lil_matrix_a[global_i, global_j] += assembly_local[i, j]

        return_dict['A' + str(worker_number)] = lil_matrix_a
        return_dict['F' + str(worker_number)] = vector_f


    def construct_system(self, func, sigma, integration_order):
        '''Returns the global matrix and force vector of the grid.'''
        matrix_a = lil_matrix((self.number_of_points, self.number_of_points), dtype=float)
        vector_f = np.zeros((self.number_of_points, 1))

        cell_indices = [cell.cell_id for cell in self.cells if cell.cell_type > 3]

        # Keep this on a sensible number, ideally 4.
        number_of_workers = 4
        cell_index_chunks = chunks(cell_indices, len(self.cells)/number_of_workers)

        return_dict = Manager().dict()
        jobs = []

        field_tuple = (func, sigma, integration_order)
        for worker_number, chunk in enumerate(cell_index_chunks):
            process = Process(target=self.worker,
                              args=(chunk, field_tuple, worker_number, return_dict,))
            jobs.append(process)
            process.start()

        for job in jobs:
            job.join()

        for key, value in return_dict.items():
            if 'A' in key:
                matrix_a += value
            elif 'F' in key:
                vector_f += value

        boundary_cells = [cell for cell in self.cells if cell.cell_type < 5]

        for cell in boundary_cells:
            for point in cell.points:
                matrix_a[point, :] = 0
                matrix_a[point, point] = 1.0
                vector_f[point] = 0

        return matrix_a, vector_f

def chunks(array, number):
    '''Splits an array into chunks of size number'''
    return [array[i:i + number] for i in range(0, len(array), number)]
