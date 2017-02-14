import numpy as np

def p1(index, x, y):
	if index == 0:
		return 1 - x - y
	elif index == 1:
		return x
	elif index == 2:
		return y

def grad_p1(index):
	if index == 0:
		return np.matrix([-1, -1]).T
	elif index == 1:
		return np.matrix([1, 0]).T
	elif index == 2:
		return np.matrix([0, 1]).T