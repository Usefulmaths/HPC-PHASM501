import numpy as np

def f(x, y):
	return 1

def sigma(x, y):
	return x

def p1(index, point):
	xi = point[0]
	eta = point[1]

	if(index == 0):
		return 1 - xi - eta

	elif(index == 1):
		return xi

	elif(index == 2):
		return eta

def grad_p1(index):
	if(index == 0):
		return np.matrix([-1, -1]).T
	elif(index == 1):
		return np.matrix([1, 0]).T
	elif(index == 2):
		return np.matrix([0, 1]).T
