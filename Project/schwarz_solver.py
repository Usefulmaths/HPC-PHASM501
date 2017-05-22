'''
SchwarzSolver class - solves a linear system using
Schwarz decomposition methods.
'''
from dolfin import *
from system import System
from decomposition_methods import *
from partition import *
import numpy as np
from mpi4py import MPI

class SchwarzSolver:
	def __init__(self, diffusion_field, diffusion_type, dimension=None, mesh_file=None, overlap=None, method='multiplicative'):
		self.diffusion_field = diffusion_field
		self.diffusion_type = diffusion_type
		self.dimension = dimension
		self.mesh_file = mesh_file
		self.method = method

		self.overlap = overlap

	def set_up_system(self, number_of_partitions=None):
		'''
		Provides the set up for the solver, defining entities such
		as restriction operators and subdomain matrices.
		'''
		if(self.dimension == None):
			system = System(None)
		else:
			system = System(self.dimension) 

		self.global_A, self.global_f, self.mesh, self.V = system.create_system(expression=self.diffusion_field, diffusion_type=self.diffusion_type, mesh_file=self.mesh_file)

		self.initial_solution = np.zeros(self.global_f.shape)

		self.dofs, self.positions = generate_dof_and_positions(self.V, self.mesh)

		if(self.overlap == None):
			self.regions = non_overlapping_partition(self.mesh, self.global_A, number_of_partitions, self.V)
		else:
			self.regions = split_regions(self.dofs, self.positions, self.overlap, self.dimension)

		self.restriction_operators = construct_restrictions(self.dofs, self.regions)
		self.subdomain_matrices = generate_subdomain_matrices(self.mesh, self.V, self.dofs, self.regions, self.global_A)
	

	def solve(self, convergence=10E-10, iterations=100):
		'''
		Solves the system using either multiplicative or 
		additive schwarz decomposition methods.
		'''
		comm = MPI.COMM_WORLD
		rank = comm.Get_rank()

		if(self.method == 'multiplicative'):
			if(rank == 0):
				solution, residuals = multiplicative_schwarz_decomposition(self.global_A, self.global_f, self.subdomain_matrices, self.restriction_operators, self.initial_solution, convergence, iterations)
				return solution, residuals
			else:
				return None, None

		elif(self.method == 'additive'):
			solution, residuals = additive_schwarz_decomposition(self.global_A, self.global_f, self.subdomain_matrices, self.restriction_operators, self.initial_solution, convergence, iterations)
			return solution, residuals

	def get_matrix_and_function(self):
		'''Retrieves the global matrix and function'''
		return self.global_A, self.global_f

	def preconditioner(self):
		'''
		Uses Schwarz decomposition methods to calculate
		a preconditioner for other solvers.
		'''
		comm = MPI.COMM_WORLD
		rank = comm.Get_rank()

		if(self.method == 'multiplicative'):
			if(rank == 0):
				preconditioner = multiplicative_schwarz_preconditioner(self.global_A, self.global_f, self.subdomain_matrices, self.restriction_operators)
				return preconditioner


		elif(self.method == 'additive'):
			preconditioner = additive_schwarz_preconditioner(self.global_A, self.global_f, self.subdomain_matrices, self.restriction_operators, self.initial_solution)
			return preconditioner