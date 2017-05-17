from scipy.sparse.linalg import minres, gmres, cg
from dolfin import *
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from time import time

from graph_mesh import plot_mesh


def on_boundary(x, on_boundary):
    return on_boundary

def dofdofs(V, mesh):
    '''
    Generates the degrees of freedom (dofs) of the mesh and their positions
    '''
    gdim = mesh.geometry().dim()
    dofs = V.dofmap().dofs()

    dofs_x = V.tabulate_dof_coordinates().reshape((-1, gdim))
    return dofs, dofs_x

def construct_r(dofs, region_dofs):
    '''
    Creates the restriction operator (check that name) for each region
    '''
    grid_length = len(dofs)
    region_length = len(region_dofs)

    r = np.zeros((region_length, grid_length))
    
    for row, dof in zip(range(len(r)), region_dofs):
        r[row, dof] = 1

    return r

def generate_subdomain_matrices(mesh, V, dofs, regions):
    '''
    Puts it all together and creates the matrices for the subdomains
    '''
    restriction_matrices = []
    subdomain_matrices = []

    for region in regions:
        r = construct_r(dofs, region)

        A = r * mat.todense() * r.T

        restriction_matrices.append(r)
        subdomain_matrices.append(A)

    return subdomain_matrices, restriction_matrices

#Size of the system
dimension = 30

#Creating the mesh and diffusion function
mesh = UnitSquareMesh(dimension, dimension)
V = FunctionSpace(mesh, 'Lagrange', 1)

#Sets up the boundary conditions
bc_value = Constant(0.0)
boundary_condition = DirichletBC(V, bc_value, on_boundary)

#Trial and test functions
u = TrialFunction(V)
v = TestFunction(V)


diffusion_field = Constant(1.0)
'''
random_field = RandomDiffusionField(dimension, dimension, element = V.ufl_element())
zero = Expression("0", element=V.ufl_element())
one = Expression("1", element=V.ufl_element())
sigma = as_matrix(((random_field, zero), (zero, one)))
#f = RandomRhs(dimension, dimension, element=V.ufl_element())
'''
f = Constant(1.0)


a = inner(diffusion_field * grad(u), grad(v)) * dx
L = f * v * dx

A = assemble(a)
b = assemble(L)

boundary_condition.apply(A, b)
A = as_backend_type(A).mat()
(indptr, indices, data) = A.getValuesCSR()
mat = csr_matrix((data, indices, indptr), shape=A.size)     
rhs = b.array()



u = np.matrix([np.random.rand() for k in range(31*31)]).T
#print(rhs)


#show_mat = np.transpose(restriction_matrices[1]) * np.linalg.inv(subdomain_matrices[1]) * restriction_matrices[1]
#non_inv = restriction_matrices[1] * mat.todense() * np.transpose(restriction_matrices[1])

dofs, dofs_x = dofdofs(V, mesh)



region1 = []
region2 = []
#region3 = []
#region4 = []

for dof, dof_x in zip(dofs, dofs_x):
    if(dof_x[0] <= 0.6):
            #if(dof_x[1] >= 0.4):
        region1.append(dof)
            #if(dof_x[1] <= 0.6):
             #   region1.append(dof)
    if(dof_x[0] >= 0.4):
            #if(dof_x[1] >=0.4):
        region2.append(dof)
            #if(dof_x[1] <=0.6):
            #    region2.append(dof)
regions = [region1, region2]#, region3, region4]

subdomains, restrictions = generate_subdomain_matrices(mesh, V, dofs, regions)

corrections = np.zeros((961, 1))
for j in range(100):
    print(j)
    print(np.linalg.norm(np.matrix(rhs).T - mat * u))
    for i in range(len(regions)):
        #u = u + (np.transpose(restrictions[i]) * np.linalg.inv(subdomains[i]) * restrictions[i]).dot((np.matrix(rhs).T - mat * u))
        corrections = corrections + (np.transpose(restrictions[i]) * np.linalg.inv(subdomains[i]) * restrictions[i]).dot((np.matrix(rhs).T - mat * u))
    u = u + corrections


'''
mesh = Mesh("lom5.xml")
'''


'''
region1_no = []
region2_no = []
region1_yo = []
region2_yo = []
for dof, dof_x in zip(dofs, dofs_x):
        if(dof_x[0] <= 0.6):
            if(dof_x[1] >= 0.4):
                region1_no.append(dof)
        if(dof_x[0] >= 0.4):
            region2_no.append(dof)
        if(dof_x[0] <= 0.6):
            region1_yo.append(dof)
        if(dof_x[0] >= 0.4):
            region2_yo.append(dof)

no_subdomains, no_restrictions = generate_subdomain_matrices(mesh, V, dofs, region1_no, region2_no)
yo_subdomains, yo_restrictions = generate_subdomain_matrices(mesh, V, dofs, region1_yo, region2_yo)
time_no = []
time_yo =[]
for j in range(25):
    if j%1==0:
        print(j)
        print(np.linalg.norm(np.matrix(rhs).T - mat * u_no))
    for i in range(2):
        start_no = time()
        u_no = u_no + (np.transpose(no_restrictions[i]) * np.linalg.inv(no_subdomains[i]) * no_restrictions[i]).dot((np.matrix(rhs).T - mat * u_no))
        end_no = time()
        start_yo = time()
        u_yo = u_yo + (np.transpose(yo_restrictions[i]) * np.linalg.inv(yo_subdomains[i]) * yo_restrictions[i]).dot((np.matrix(rhs).T - mat * u_yo))
        end_yo = time()
        time_no.append(end_no-start_no)
        time_yo.append(end_yo- start_yo)

print(np.mean(time_no))
print(np.mean(time_yo))
u = u_yo - u_no
'''
j = Function(V)
j.vector()[:] = u

plot_mesh(j, mesh)