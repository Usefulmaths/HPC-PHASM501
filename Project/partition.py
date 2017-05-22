import numpy as np

from system import System
from dolfin import *
import matplotlib.pyplot as plt

import networkx as nx
import metis

def adjacency_matrix(global_A, mesh):
	adj_matrix = np.zeros(global_A.shape)

	for cell in mesh.cells():
		row = adj_matrix[cell[0]]
		if(row[cell[0]] == 0):
			adj_matrix[cell[0], cell[0]] = 1
		if(row[cell[1]] == 0):
			adj_matrix[cell[0], cell[1]] = 1
			adj_matrix[cell[1], cell[0]] = 1
		if(row[cell[2]] == 0):
			adj_matrix[cell[0], cell[2]] = 1
			adj_matrix[cell[2], cell[0]] = 1

		row = adj_matrix[cell[1]]
		if(row[cell[0]] == 0):
			adj_matrix[cell[1], cell[0]] = 1
			adj_matrix[cell[0], cell[1]] = 1
		if(row[cell[1]] == 0):
			adj_matrix[cell[1], cell[1]] = 1
		if(row[cell[2]] == 0):
			adj_matrix[cell[1], cell[2]] = 1
			adj_matrix[cell[2], cell[1]] = 1

		row = adj_matrix[cell[2]]
		if(row[cell[0]] == 0):
			adj_matrix[cell[2], cell[0]] = 1
			adj_matrix[cell[0], cell[2]] = 1
		if(row[cell[1]] == 0):
			adj_matrix[cell[2], cell[1]] = 1
			adj_matrix[cell[1], cell[2]] = 1
		if(row[cell[2]] == 0):
			adj_matrix[cell[2], cell[2]] = 1

	return adj_matrix

def non_overlapping_partition(mesh, global_A, nparts, V):
	adj_matrix = adjacency_matrix(global_A, mesh)
	G = nx.from_numpy_matrix(adj_matrix)
	partitions = metis.part_graph(G, nparts=nparts)[1]
	vertices_to_dofs = vertex_to_dof_map(V)

	regional_dofs = [[] for l in range(nparts)]
	for index, dof in enumerate(partitions):
		for i in range(nparts):
			if i == dof:
				regional_dofs[i].append(vertices_to_dofs[index])

	return regional_dofs

def split_regions(dofs, positions, overlap, dimension):
    '''
    Splits a unit square into two regions with an overlap centred
    around a vertically centred line.
    '''
    region1 = []
    region2 = []

    sorted_by_closest_node_upper = sorted(positions, key=lambda x: abs(x[0] - (0.5000001 + float(overlap)/(dimension - 1))))
    sorted_by_closest_node_lower = sorted(positions, key=lambda x: abs(x[0] - (0.4999999 - float(overlap)/(dimension - 1))))
    overlap_percentage = 2 * (0.5 - sorted_by_closest_node_lower[0][0]) * 100

    for dof, position in zip(dofs, positions):
        if(position[0] <= sorted_by_closest_node_upper[0][0]):
            region1.append(dof)

        if(position[0] >= sorted_by_closest_node_lower[0][0]):
            region2.append(dof)

    return [region1, region2]


