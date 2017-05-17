from dolfin import *
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

def generate_dof_and_positions(V, mesh):
    '''
    Generates the degrees of freedom for given mesh.
    '''
    gdim = mesh.geometry().dim()
    dofs = V.dofmap().dofs()

    dof_position = V.tabulate_dof_coordinates().reshape((-1, gdim))
    return dofs, dof_position

def split_regions(dofs, positions):
    region1 = []
    region2 = []

    for dof, position in zip(dofs, positions):
        if(position[0] <= 0.6):
            region1.append(dof)
        if(position[0] >= 0.4):
            region2.append(dof)

    return [region1, region2]

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
    return np.linalg.norm(rhs - global_matrix * approx_solution)

def multiplicative_schwarz_decomposition(global_matrix, rhs, subdomains, restrictions, initial_solution, convergence=10**-5, iterations=50):
    number_of_subdomains = len(subdomains)
    res = 10E10
    iter_num = 0

    while(res >= convergence and iter_num <= iterations):
        print(res)
        for j in range(number_of_subdomains):
            initial_solution = initial_solution + np.dot((np.transpose(restrictions[j]) * np.linalg.inv(subdomains[j]) * restrictions[j]), (rhs - global_matrix * initial_solution))
        res = residual(global_matrix, rhs, initial_solution)
        iter_num += 1

    return initial_solution

def multiplicative_schwarz_preconditioner(global_matrix, rhs, subdomains, restrictions, initial_solution, convergence=10**-5, iterations=50):
    number_of_subdomains = len(subdomains)
    res = 10E10
    iter_num = 0

    t1 = (restrictions[0].T * np.linalg.inv(subdomains[0]) * restrictions[0]) 
    z = t1 * rhs

    while(res >= convergence and iter_num <= iterations):
        print(res)
        print(iter_num)
        for j in range(1, number_of_subdomains):
            z = z + (restrictions[j].T * np.linalg.inv(subdomains[j]) * restrictions[j]) * (rhs - global_matrix * z)
        res = residual(global_matrix, rhs, z)
        iter_num += 1

    return z

def additive_schwarz_decomposition(global_matrix, rhs, subdomains, restrictions, initial_solution, convergence=10**-5, iterations=50, number_of_cores=2):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    number_of_subdomains = len(subdomains)
    res = 10E10
    iter_num = 0

    subdomain_pairs = [[subdomains[index], restrictions[index]] for index in range(number_of_subdomains)]

    while res >= convergence and iter_num <= iterations:
        if(rank == 0):
            subdomain_pair = subdomain_pairs
            print(res)
            iter_num += 1
        else:
            subdomain_pair = []

        subdomain_pair = comm.scatter(subdomain_pair, root=0)
        subdomain_matrix = subdomain_pair[0]
        restriction_matrix = subdomain_pair[1]

        residuals = np.zeros((1, len(global_matrix.todense()))).T
        residuals = residuals + np.dot(np.transpose(restriction_matrix) * np.linalg.inv(subdomain_matrix) * restriction_matrix, rhs - global_matrix * initial_solution)
    
        comm.Barrier()

        residuals = comm.gather(residuals, root=0)
        if(rank == 0):
            for resid in residuals:
                initial_solution = initial_solution + resid

            res = residual(global_matrix, rhs, initial_solution)
        else:
            initial_solution = []
            res = []

        initial_solution = comm.bcast(initial_solution, root=0)
        res = comm.bcast(res, root=0)

    return initial_solution

def additive_schwarz_preconditioner(global_matrix, rhs, subdomains, restrictions, initial_solution, number_of_cores=2):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    number_of_subdomains = len(subdomains)

    subdomain_pairs = [[subdomains[index], restrictions[index]] for index in range(number_of_subdomains)]

    if(rank == 0):
        subdomain_pair = subdomain_pairs
    else:
        subdomain_pair = []

    subdomain_pair = comm.scatter(subdomain_pair, root=0)
    subdomain_matrix = subdomain_pair[0]
    restriction_matrix = subdomain_pair[1]

    preconditer_sub = (np.transpose(restriction_matrix) * np.linalg.inv(subdomain_matrix) * restriction_matrix)

    comm.Barrier()

    preconditioner_parts = comm.gather(preconditer_sub, root=0)
    if(rank == 0):
        preconditioner = np.zeros(global_matrix.shape)
        for preconditioner_sub in preconditioner_parts:
            preconditioner = preconditioner + preconditioner_sub

        res = residual(global_matrix, rhs, initial_solution)
    else: 
        preconditioner = None

    preconditioner = comm.bcast(preconditioner, root=0)

    return preconditioner
