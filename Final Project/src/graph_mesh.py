from dolfin import *
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def mesh2triang(mesh):
    '''
    Copied from Timos code for viewing the solution to the diffusion
    '''
    xy = mesh.coordinates()
    return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())


def plot_mesh(solution, mesh, V):
	'''
	Plots solution of diffusion equation.
	'''
	j = Function(V)
	j.vector()[:] = solution
	C = j.compute_vertex_values(mesh)
	plt.tripcolor(mesh2triang(mesh), C, shading='gouraud')
	plt.colorbar()
	plt.axis('equal')
	plt.show()
