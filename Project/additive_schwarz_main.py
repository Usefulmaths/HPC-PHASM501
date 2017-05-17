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

solution = additive_schwarz_decomposition(global_A, global_f, subdomain_matrices, restriction_operators, initial_solution)

plot_mesh(solution, mesh, V)
