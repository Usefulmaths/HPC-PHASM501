from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import gmres, cg, minres
from graph_mesh import *

from partition import *

from mpi4py import MPI

def generate_dof_and_positions(V, mesh):
    '''
    Generates the degrees of freedom for given mesh.
    '''
    gdim = mesh.geometry().dim()
    dofs = V.dofmap().dofs()

    dof_position = V.tabulate_dof_coordinates().reshape((-1, gdim))
    return dofs, dof_position


def construct_restriction(dofs, region_dofs):
    '''
    Creates the restriction operators for a given region.
    '''
    grid_length = len(dofs)
    region_length = len(region_dofs)

    r = np.zeros((region_length, grid_length))
    
    for row, dof in zip(range(len(r)), region_dofs):
        r[row, dof] = 1

    return r

def construct_restrictions(dofs, regions_dofs):
    '''
    Creates a list of all the restriction operators.
    '''
    restriction_array = []

    for region_dof in regions_dofs:
        restriction_array.append(construct_restriction(dofs, region_dof))

    return restriction_array

def generate_subdomain_matrices(mesh, V, dofs, regions, global_matrix):
    '''
    Creates subdomain matrices and restriction operators.
    '''
    subdomain_matrices = []

    for region in regions:
        r = construct_restriction(dofs, region)
        A = r * global_matrix.todense() * r.T
        subdomain_matrices.append(A)

    return subdomain_matrices

def residual(global_matrix, rhs, approx_solution):
    '''
    Calculates the residual of a solution to a 
    linear equation.
    '''
    return np.linalg.norm(rhs - global_matrix * approx_solution)

def multiplicative_schwarz_decomposition(global_matrix, rhs, subdomains, restrictions, initial_solution, convergence=10**-10, iterations=5000):
    number_of_subdomains = len(subdomains)
    res = 10E10
    iter_num = 0
    residual_list = []

    while(res >= convergence and iter_num <= iterations):
        temp = np.zeros(rhs.shape)
        for j in range(number_of_subdomains):
            
            subdomain_f = np.matrix(restrictions[j]) * np.matrix(rhs)
            subdomain_initial_solution = np.matrix(restrictions[j]) * np.matrix(initial_solution)

            subdomain_solution, _ = gmres(np.matrix(subdomains[j]), np.matrix(subdomain_f), x0=np.matrix(subdomain_initial_solution), maxiter=1)
            subdomain_solution = np.matrix(subdomain_solution).T
        
            initial_solution = initial_solution + restrictions[j].T * subdomain_solution + restrictions[j].T * np.linalg.inv(subdomains[j]) * restrictions[j] * (rhs - global_matrix * restrictions[j].T * subdomain_solution)
               
            #initial_solution = initial_solution + np.dot((np.transpose(restrictions[j]) * np.linalg.inv(subdomains[j]) * restrictions[j]), (rhs - global_matrix * initial_solution))
        res = residual(global_matrix, rhs, initial_solution)

        residual_list.append(res)
        iter_num += 1

    return initial_solution, residual_list

def multiplicative_schwarz_preconditioner(global_matrix, rhs, subdomains, restrictions):
    number_of_subdomains = len(subdomains)
    m_i = np.zeros(global_matrix.shape)

    for i in range(number_of_subdomains):
        t_i = restrictions[i].T * np.linalg.inv(subdomains[i]) * restrictions[i]
        m_i = m_i + t_i * (np.identity(t_i.shape[0]) - global_matrix * m_i)

    return m_i

def additive_schwarz_decomposition(global_matrix, rhs, subdomains, restrictions, initial_solution, convergence=10**-10, iterations=50, number_of_cores=2):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    number_of_subdomains = len(subdomains)
    res = 10E10
    iter_num = 0
    residual_list = []

    if(rank == 0):
        subdomain_pairs = [[subdomains[index], restrictions[index]] for index in range(number_of_subdomains)]
        chunks = [[] for _ in range(size)]
        for i, chunk in enumerate(subdomain_pairs):
            chunks[i % size].append(chunk)
    else:
        subdomain_pairs = None
        chunks = None

    subdomain_pairs = comm.scatter(chunks, root=0)   

    while res >= convergence and iter_num < iterations:
        rkvalues = []
        callback = lambda x: rkvalues.append(x)

        subdomain_residuals = [] 
        for subdomain_pair in subdomain_pairs:
            residuals = np.zeros(rhs.shape) 

            subdomain_matrix = subdomain_pair[0]
            restriction_matrix = subdomain_pair[1]

            residuals = residuals + np.dot(np.transpose(restriction_matrix) * np.linalg.inv(subdomain_matrix) * restriction_matrix, rhs - global_matrix * initial_solution) 
            subdomain_residuals.append(residuals)

        comm.Barrier()

        subdomain_residuals = comm.gather(subdomain_residuals, root=0)

        if(rank == 0):
            initial_solution = initial_solution + reduce(lambda x, y: x + y, [item for sublist in subdomain_residuals for item in sublist])
            res = residual(global_matrix, rhs, initial_solution)
            residual_list.append(res)
        else:
            initial_solution = []
            res = []

        initial_solution = comm.bcast(initial_solution, root=0)
        res = comm.bcast(res, root=0)

        #print("rank", rank, iter_num)
        if rank == 0:
            #print(res)
            iter_num += 1
            if iter_num % 100:
                print(iter_num)
        else:
            residual_list = None
            iter_num = None

        residual_list = comm.bcast(residual_list, root=0)
        iter_num = comm.bcast(iter_num, root=0)
        comm.Barrier()
    
    return initial_solution, residual_list

def additive_schwarz_preconditioner(global_matrix, rhs, subdomains, restrictions, initial_solution, number_of_cores=2):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    number_of_subdomains = len(subdomains)
    iter_num = 0

    if(rank == 0):
        subdomain_pairs = [[subdomains[index], restrictions[index]] for index in range(number_of_subdomains)]
        chunks = [[] for _ in range(size)]
        for i, chunk in enumerate(subdomain_pairs):
            chunks[i % size].append(chunk)
        iter_num += 1
    else:
        subdomain_pairs = None
        chunks = None

    subdomain_pairs = comm.scatter(chunks, root=0)
    precon_subs = []
    for subdomain_pair in subdomain_pairs:
        subdomain_matrix = subdomain_pair[0]
        restriction_matrix = subdomain_pair[1]
        preconditioner_sub = (np.transpose(restriction_matrix) * np.linalg.inv(subdomain_matrix) * restriction_matrix)
        precon_subs.append(preconditioner_sub)

    comm.Barrier()

    precon_subs = comm.gather(precon_subs, root=0)
    if(rank == 0):
        preconditioner = np.zeros(global_matrix.shape)
        preconditioner = preconditioner + reduce(lambda x, y: x + y, [item for sublist in precon_subs for item in sublist])
        res = residual(global_matrix, rhs, initial_solution)
    else: 
        preconditioner = None

    preconditioner = comm.bcast(preconditioner, root=0)

    return preconditioner
