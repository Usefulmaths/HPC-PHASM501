from schwarz_solver import SchwarzSolver
from graph_mesh import *

import numpy as np
from scipy.sparse.linalg import gmres
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

########################
# Pure Schwarz Example #
########################

'''Initialise system variables'''
number_of_partitions = 4

diffusion_type = 'constant'
diffusion_expression = Constant(1.0)

mesh_file = "meshes/holey_square.xml"

'''Instantiate SchwarzSolver class'''
solver = SchwarzSolver(diffusion_expression, diffusion_type, mesh_file=mesh_file, method='multiplicative', solver_type=None)
#solver = SchwarzSolver(diffusion_expression, diffusion_type, mesh_file=mesh_file, method='multiplicative', solver_type=None)

'''Set up the solver system'''
solver.set_up_system(number_of_partitions)

'''Solve the system for the solution and residuals'''
solution, residuals = solver.solve()

if(rank == 0):	
	'''Plot the solution'''
	plot_mesh(solution, solver.mesh, solver.V)


#####################################
# Precondition with Schwarz Example #
#####################################

'''Initialise system variables'''
number_of_partitions = 4

diffusion_type = 'constant'
diffusion_expression = Constant(1.0)
function_rhs = Constant(1.0)

mesh_file = "meshes/holey_square_IV.xml"

'''Instantiate SchwarzSolver class'''
solver = SchwarzSolver(diffusion_expression, diffusion_type, function=function_rhs, mesh_file=mesh_file, method='multiplicative')

'''Set up the solver system'''
solver.set_up_system(number_of_partitions)

'''Calculate the preconditioner'''
preconditioner = solver.preconditioner()

if(rank == 0):
	'''Retrieve global matrix and function from system'''
	matrix, function = solver.get_matrix_and_function()

	'''Use a Krylov subspace solver'''
	solution, _ = gmres(matrix, function, M=preconditioner)

	'''Plot the solution'''
	plot_mesh(solution, solver.mesh, solver.V)

#######################################
# Example of unit square with overlap #
#######################################

'''Initialise system variables'''
dimension = 20
overlap = 3 # Defines the 'level' of overlap (how many nodes across from the central vertical)

diffusion_type = 'constant'
diffusion_expression = Constant(1.0)

'''Instantiate SchwarzSolver class'''
solver = SchwarzSolver(diffusion_expression, diffusion_type, dimension=dimension, overlap=overlap, method='additive')

'''Set up the solver system'''
solver.set_up_system()

'''Solve the system for the solution and residuals'''
solution, residuals = solver.solve()

if(rank == 0):	
	'''Plot the solution'''
	plot_mesh(solution, solver.mesh, solver.V)