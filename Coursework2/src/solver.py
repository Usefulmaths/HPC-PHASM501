from scipy.sparse.linalg import gmres, minres, cg, bicgstab, spilu, LinearOperator
from pyamg import ruge_stuben_solver, smoothed_aggregation_solver, rootnode_solver
from numpy.linalg import norm
import numpy as np

import matplotlib.pyplot as plt

class Solver:
	'''
	Solver class allows us to specific the type of solver and any preconditioner that will set up a general
	solve method that will solve a matrix equation.
	'''
	def __init__(self, solver_type, subsolver_type=None, precondition_type=None):
		self.solver_type = solver_type
		self.subsolver_type = subsolver_type
		self.precondition_type = precondition_type

	def get_residuals(self, xk):
		if(self.solver_type == 'gmres'):
			return xk
		else:
			return self.compute_residual(xk)

	def compute_residual(self, xk):
		return norm(self.matrix * xk - self.vector)

	def solve(self, system):
		'''
		Solves a matrix system, given a specific solver type.
		'''

		res = []

		matrix, vector = system.matrix, system.vector
		append_residuals = lambda residuals: res.append(residuals) if self.solver_type == 'gmres' else res.append(norm(matrix * residuals - vector))

		preconditioner = None

		if(self.solver_type == 'gmres'):
			solver = gmres
		
		elif(self.solver_type == 'minres'):
			solver = minres

		elif(self.solver_type == 'cg'):
			solver = cg

		elif(self.solver_type == 'bicgstab'):
			solver = bicgstab

		elif(self.solver_type == 'amg'):
			if(self.subsolver_type == 'ruge_stuben'):
				solver = ruge_stuben_solver(matrix)

			elif(self.subsolver_type == 'rootnode'):
				solver = rootnode_solver(matrix)

  			elif(self.subsolver_type == 'smoothed_aggregation'):
  				solver = smoothed_aggregation_solver(matrix)

  		if self.precondition_type in ['gmres', 'minres', 'cg', 'bicgstab']:
  			preconditioner = self.precondition_type

  		elif(self.precondition_type is 'smoothed_aggregation'):
  			preconditioner = smoothed_aggregation_solver(matrix).aspreconditioner(cycle='V')

  		elif(self.precondition_type is 'rootnode'):
  			preconditioner = rootnode_solver(matrix).aspreconditioner(cycle='V')

  		elif(self.precondition_type is 'ruge_stuben'):
  			preconditioner = ruge_stuben_solver(matrix).aspreconditioner(cycle='V')

  		elif(self.precondition_type is 'ilu'):
  			ilu_solver = spilu(matrix)
  			M = lambda x: ilu_solver.solve(x)
  			preconditioner = LinearOperator(matrix.shape, M)

  		if(self.solver_type is not 'amg'):
  			solution = solver(matrix, vector, M=preconditioner, callback=append_residuals, tol=1e-10)

  		else:
  			solution = solver.solve(vector, accel=preconditioner, callback=append_residuals, tol=1e-10)

		return solution, res


