from grid import * 
from IO import *
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix
from time import time
def solver(vtk_file_name, f, sigma, integration_order=4):
	data = import_vtk_file(vtk_file_name)

	grid = Grid(data)
	grid.set_grid()

	start = time()
	A, f = grid.construct_system(f, sigma, integration_order)
	end = time()
	print("Matrix Assembly", end - start)
	row = []
	col = []
	data = []

	for i in range(len(A)):
	    for j in range(len(A)):
	        if(A[i, j] != 0):
	            row.append(i)
	            col.append(j)
	            data.append(A[i, j])
	
	B = coo_matrix((data, (row, col)), shape=(A.shape[0], A.shape[0]))
	u = np.matrix(spsolve(B.tocsr(), f)).T

	xlist = []
	ylist = []

	for point in grid.points:
	    x = point[0, 0]
	    y = point[0, 1]
	    xlist.append(x)
	    ylist.append(y)

	for i in range(len(xlist)):
		print(xlist[i], ylist[i], u[i])
    

	return A, f, u