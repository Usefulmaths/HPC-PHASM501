from grid import * 
from IO import *
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix

def solver(vtk_file_name, f, sigma, integration_order=4):
	data = import_vtk_file(vtk_file_name)

	grid = Grid(data)
	grid.set_grid()

	A, f = grid.construct_system(f, sigma, integration_order)

	row = []
	col = []
	data = []

	for i in range(len(A)):
	    for j in range(len(A)):
	        if(A[i, j] != 0):
	            row.append(i)
	            col.append(j)
	            data.append(A[i, j])
	
	A = coo_matrix((data, (row, col)), shape=(A.shape[0], A.shape[0]))
	u = np.matrix(spsolve(A.tocsr(), f)).T

	return u