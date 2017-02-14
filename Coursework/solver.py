from grid import * 
from IO import *
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix
from time import time
from numpy.linalg import inv
def solver(vtk_file_name, f, sigma, integration_order=4):
	data = import_vtk_file(vtk_file_name)

	grid = Grid(data)
	grid.set_grid()

	A, f = grid.construct_system(f, sigma, integration_order)
	u = np.dot(inv(A), f)
	print(u)

	return A, f, u