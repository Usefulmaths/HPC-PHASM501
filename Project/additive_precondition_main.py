from system import System
from decomposition_methods import *
from graph_mesh import plot_mesh

import numpy as np
from mpi4py import MPI
from scipy.sparse.linalg import gmres

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

dimension = 31
sys = System(dimension)

global_A, global_f, mesh, V = sys.create_system(expression=Constant(1.0), diffusion_type='constant', mesh_file="lom6.xml")

initial_solution = np.zeros(global_f.shape)
dofs, positions = generate_dof_and_positions(V, mesh)
regions = split_regions(dofs, positions)

restriction_operators = construct_restrictions(dofs, regions)
subdomain_matrices = generate_subdomain_matrices(mesh, V, dofs, regions, global_A)

preconditioner = additive_schwarz_preconditioner(global_A, global_f, subdomain_matrices, restriction_operators, initial_solution)

def callback(rk):
	rkvalues.append(rk)
	return rk

if rank == 0:
	rkvalues = []
	solution, flag = gmres(global_A, global_f, callback=callback, tol=10E-15)
	print("Number of iterations: " + str(len(rkvalues)))
	plot_mesh(solution, mesh, V)

	rkvalues = []
	solution_pre, flag = gmres(global_A, global_f, M=preconditioner, callback=callback, tol=10E-15)
	print("Number of iterations: " + str(len(rkvalues)))
	plot_mesh(solution_pre, mesh, V)

	plot_mesh(solution - solution_pre, mesh, V)

