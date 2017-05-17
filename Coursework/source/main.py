
from solver import solver
from time import time
import numpy as np
import multiprocessing

file_name = "./example_grids/Small_Square_Triangle_Grid"
integration_order = 1	

def f(x, y):
	return 1

def sigma(x, y):
    return 1

solver(file_name, f, sigma, integration_order)