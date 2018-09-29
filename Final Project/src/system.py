from dolfin import *
from scipy.sparse import csr_matrix
import numpy as np


def on_boundary(x, on_boundary):
		'''
		Callback function for boundary points
		'''
		return on_boundary

class System:

	'''
	System class takes in the dimension of a grid and has the methods necessary to create a matrix and vector
	for an equation Ax = b, given a specific diffusion type.
	'''
	def __init__(self, dimension):
		self.dimension = dimension

	def create_system(self, expression, diffusion_type, function, mesh_file=None):
		'''
		Creates a matrix A, and vector b, representing the diffusion generic diffusion equation.
		'''
		if(mesh_file == None):
			mesh = UnitSquareMesh(self.dimension - 1, self.dimension - 1)
		else:
			mesh = Mesh(mesh_file)
			
		V = FunctionSpace(mesh, 'Lagrange', 1)
		
		bc_value = Constant(0.0)
		boundary_condition = DirichletBC(V, bc_value, on_boundary)

		u = TrialFunction(V)
		v = TestFunction(V)

		# Rough diffusion
		if(diffusion_type is 'rough'):
			# expression set to zero; removes weird inteference with expression define within parameters.
			expression = 0

			# Random sigma field
			random_field = DiffusionField(self.dimension, self.dimension, element=V.ufl_element())
			zero = Expression("0", element=V.ufl_element())
			one = Expression("1", element=V.ufl_element())
			diffusion = as_matrix(((random_field, zero), (zero, one)))

			a = inner(diffusion * grad(u), grad(v)) * dx

			# Random RHS
			L = RandomRhs(self.dimension, self.dimension, element=V.ufl_element()) * v * dx
		
		elif(diffusion_type is 'smooth' or diffusion_type is 'constant' or diffusion_type is 'anisotropic'):
			diffusion = expression

			a = inner(diffusion * grad(u), grad(v)) * dx
			L = function * v * dx
		
		A = assemble(a)
		b = assemble(L)

		boundary_condition.apply(A, b)
		A = as_backend_type(A).mat()
		(indptr, indices, data) = A.getValuesCSR()
		mat = csr_matrix((data, indices, indptr), shape=A.size) 	
		rhs = b.array()

		self.mesh = mesh
		self.V = V
		self.matrix = mat
		self.vector = rhs

		return mat, np.matrix(rhs).T, mesh, V

class DiffusionField(Expression):
    def __init__(self, m, n, element):
        """
        Define a random diffusion field by
        subdividing the domain into an m x n grid.

        """
        self._rand_field = np.exp(-np.random.randn(m, n))
        self._m = m
        self._n = n
        self._ufl_element = element
        
    def eval(self, value, x):
        
        x_index = np.int(np.floor(self._m * x[0]))
        y_index = np.int(np.floor(self._n * x[1]))
        
        i = min(x_index, self._m - 1)
        j = min(y_index, self._n - 1)
        
        value[0] = self._rand_field[i, j]
        
    def value_shape(self):
        return (1, )       
        
class RandomRhs(Expression):
    
    def __init__(self, m, n, element):
        """
        Define a random right-hand side function by
        subdividing the domain into an m x n grid.

        """
        self._rand_field = np.random.randn(m, n)
        self._m = m
        self._n = n
        self._ufl_element = element
        
    def eval(self, value, x):
        
        x_index = np.int(np.floor(self._m * x[0]))
        y_index = np.int(np.floor(self._n * x[1]))
        
        i = min(x_index, self._m - 1)
        j = min(y_index, self._n - 1)
        
        value[0] = self._rand_field[i, j]
        
    def value_shape(self):
        return (1, )            

