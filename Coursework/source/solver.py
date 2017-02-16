'''
This module contains the global solver routine.
'''
from scipy.sparse.linalg import spsolve
from grid import Grid
from IO import import_vtk_file, export_vtk_file

def solver(vtk_file_name, func, sigma, integration_order=4):
    '''
    Imports grid data from a VTK file, sets up a Grid object,
    calculates the assembly matrix and function vector,
    solves for the solution over the grid, and exports
    the solution to a VTK file.
    '''

    print "Importing VTK file grid data."
    data = import_vtk_file(vtk_file_name)

    print "Setting up grid."
    grid = Grid(data)
    grid.set_grid()

    print "Constructing system of assembly matrix and force vector."
    assembly_matrix, force_vector = grid.construct_system(func, sigma, integration_order)

    print "Solving for the solution."
    solution = spsolve(assembly_matrix.tocsr(), force_vector)

    print "Exporting solution to a VTK file located at: " + str(vtk_file_name) + "_solution.vtk"
    export_vtk_file(vtk_file_name, grid.number_of_points, solution)
