'''
This file deals with all the analysis
and plots given in the final report 
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_file_names = ['../data/square_dim5_partitions.txt', '../data/square_dim10_partitions.txt', '../data/square_dim20_partitions.txt', '../data/square_dim30_partitions.txt']
names = ["5", "10", "20", "30"]

for index, file_name in enumerate(data_file_names):
	df = pd.read_csv(file_name, names=['partitions', 'iters', 'time'])
	
	partitions = df['partitions']
	iters = df['iters']
	time = df['time']

	plt.plot(partitions, iters, marker="o", label=names[index])
	#plt.plot(partitions, time, marker="o", label=names[index])

plt.legend(title="Dimensions")
plt.title("Multiplicative Schwarz - \nNumber of Partitions vs Number of Iterations")
plt.xlabel('Number of Partitions')
plt.ylabel("Iterations")
plt.show()

data_file_names = ['../data/square_dim5_partitions.txt', '../data/square_dim10_partitions.txt', '../data/square_dim20_partitions.txt', '../data/square_dim30_partitions.txt']
names = ["5", "10", "20", "30"]

for index, file_name in enumerate(data_file_names):
	df = pd.read_csv(file_name, names=['partitions', 'iters', 'time'])
	
	partitions = df['partitions']
	iters = df['iters']
	time = df['time']

	plt.plot(partitions, time, marker="o", label=names[index])
	#plt.plot(partitions, time, marker="o", label=names[index])

plt.legend(title="Dimensions")
plt.title("Multiplicative Schwarz - \nNumber of Partitions vs Time")
plt.xlabel('Number of Partitions')
plt.ylabel("Time (s)")
plt.show()

data_file_names = ['../data/additive_square_dim5_partitions_proc1.txt', '../data/additive_square_dim10_partitions_proc1.txt', '../data/additive_square_dim20_partitions_proc1.txt', '../data/additive_square_dim30_partitions_proc1.txt']
names = ["5", "10", "20", "30"]

for index, file_name in enumerate(data_file_names):
	df = pd.read_csv(file_name, names=['partitions', 'iters', 'time'])
	
	partitions = df['partitions']
	iters = df['iters']
	time = df['time']

	plt.plot(partitions, iters, marker="o", label=names[index])
	#plt.plot(partitions, time, marker="o", label=names[index])

plt.legend(title="Dimensions")
plt.title("Additive Schwarz - \nNumber of Partitions vs Number of Iterations")
plt.xlabel('Number of Partitions')
plt.ylabel("Iterations")
plt.show()

data_file_names = ['../data/additive_square_dim5_partitions_proc1.txt', '../data/additive_square_dim10_partitions_proc1.txt', '../data/additive_square_dim20_partitions_proc1.txt', '../data/additive_square_dim30_partitions_proc1.txt']
names = ["5", "10", "20", "30"]

for index, file_name in enumerate(data_file_names):
	df = pd.read_csv(file_name, names=['partitions', 'iters', 'time'])
	
	partitions = df['partitions']
	iters = df['iters']
	time = df['time']

	plt.plot(partitions, time, marker="o", label=names[index])
	#plt.plot(partitions, time, marker="o", label=names[index])

plt.legend(title="Dimensions")
plt.title("Additive Schwarz - \nNumber of Partitions vs Time")
plt.xlabel('Number of Partitions')
plt.ylabel("Time (s)")
plt.show()

data_file_name = '../data/multiplicative_square_dim30_overlap.txt'

df = pd.read_csv(data_file_name, names=['overlap', 'residual', 'time'])
print(df)
overlap = df['overlap'].round(2)
time = df['time']
residual = df['residual']

for index, res in enumerate(residual):
	res = str(res).split("@")
	res = [float(r) for r in res]
	print(res)
	plt.plot([np.log10(float(r)) for r in res], marker="o", label=str(overlap[index])+"%")

plt.xlabel("Number of Iterations")
plt.ylabel("Residual " + r"($\log_{10}$)")
plt.title("Multiplicative Schwarz \n Residual Curves for different sized overlaps")
plt.legend(title="Overlap Percentage")
plt.show()

plt.plot(overlap, time, marker='o')
plt.xlabel('Overlap Percentage (%)')
plt.ylabel('Time (s)')
plt.title('Multiplicative Schwarz \n Time curve for different sized overlaps')
plt.show()

data_file_name = '../data/additive_square_dim30_overlap.txt'

df = pd.read_csv(data_file_name, names=['overlap', 'residual', 'time'])
overlap = df['overlap'].round(2)
time = df['time']
residual = df['residual']

for index, res in enumerate(residual[:-2]):
	res = str(res).split("@")
	res = [float(r) for r in res][:25]
	print(res)
	plt.plot([r for r in res], marker="o", label=str(overlap[index])+"%")

plt.xlabel("Number of Iterations")
plt.ylabel("Residual")
plt.title("Additive Schwarz \n Residual Curves for different sized overlaps")
plt.legend(title="Overlap Percentage")
plt.show()

data_file_names = ["../data/additive_square_dim5_partitions_processor_1.txt", "../data/additive_square_dim5_partitions_processor_2.txt"]
label_names = ["Non-Parallel (1 core)", "Parallel (2 cores)"]
times = []
for index, data_file in enumerate(data_file_names):
	df = pd.read_csv(data_file, names=['partitions', 'time'])
	
	partitions = df['partitions']
	time = df['time']
	times.append(np.array(time))
	plt.plot(partitions, time, marker="o", label=label_names[index])
plt.xlabel("Number of Partitions")
plt.ylabel("Time(s)")
plt.title("Additive Schwarz \n Parallel time curves")
plt.legend()
plt.show()

quit()



data_file_names = ["../data/additive_precon_residuals", "../data/multi_precon_residuals", "../data/additive_square_dim50_partitions_processor_2.txt"]

df1 = pd.read_csv("../data/additive_precon_residuals", names=["precon", "noncon"])
df2 = pd.read_csv("../data/multi_precon_residuals", names=["precon"])
df3 = pd.read_csv("../data/additive_square_dim50_partitions_processor_2.txt", names=["residuals", "time"])
df4 = pd.read_csv("../data/square_dim50_partitions.txt", names=["residuals", "time"])

add_noncon_res = str(df1["noncon"][0]).split("@")
add_noncon_res = [float(res) for res in add_noncon_res]
add_precon_res = str(df1['precon'][0]).split("@")
add_precon_res = [float(res) for res in add_precon_res]
multi_precon_res = str(df2['precon'][0]).split("@")
multi_precon_res = [float(res) for res in multi_precon_res]

additive_solve = str(df3["residuals"][0]).split("@")
additive_solve = [float(res) for res in additive_solve]
multiplicative_solve = str(df4["residuals"][0]).split("@")
multiplicative_solve = [float(res) for res in multiplicative_solve]

plt.plot(np.log10(add_noncon_res), label="No Preconditioner")
plt.plot(np.log10(add_precon_res), label="Additive Preconditioner")
plt.plot(np.log10(multi_precon_res), label="Multiplicative Preconditioner")
plt.plot(np.log10(additive_solve), label="Parallel Additive Solver")
plt.plot(np.log10(multiplicative_solve), label="Multiplicative Solver")

plt.title("Convergence curves comparing pure Schwarz methods \nwith Schwarz preconditioned GMRES")
plt.xlabel("Number of iterations")
plt.ylabel("Residual " + r"($\log_{10}$)")
plt.legend()
plt.show()



quit()







data_file_names = ['../data/multi_precon_residuals_partitionscircle_triangle', '../data/multi_precon_residuals_partitionsl_mesh', '../data/multi_precon_residuals_partitionsholey_square', '../data/multi_precon_residuals_partitionstriangle']
names = ["Circle triangle", "Holey square", "L shape", "Triangley square"]

for index, file_name in enumerate(data_file_names):
	df = pd.read_csv(file_name, names=['partitions', 'preiters'])
	
	partitions = df['partitions']
	preiters = df['preiters']

	plt.plot(partitions, preiters, marker="o", label=names[index])

plt.legend()
plt.title("Multiplicative Schwarz Precondition - \nNumber of Partitions vs GMRES Iterations")
plt.xlabel('Number of Partitions')
plt.ylabel("Iterations")
plt.show()


data_file_names = ['../data/additive_precon_residuals_partitionscircle_triangle', '../data/additive_precon_residuals_partitionsl_mesh', '../data/additive_precon_residuals_partitionsholey_square', '../data/additive_precon_residuals_partitionstriangle']
names = ["Circle triangle", "Holey square", "L shape", "Triangley square"]

for index, file_name in enumerate(data_file_names):
	df = pd.read_csv(file_name, names=['partitions', 'preiters'])
	
	partitions = df['partitions']
	preiters = df['preiters']

	plt.plot(partitions, preiters, marker="o", label=names[index])

plt.legend()
plt.title("Additive Schwarz Precondition - \nNumber of Partitions vs GMRES Iterations")
plt.xlabel('Number of Partitions')
plt.ylabel("Iterations")
plt.show()

data_file_names = ['../data/multi_precon_residuals_overlap_constant', '../data/multi_precon_residuals_overlap_random']
names = ["Constant", "Random"]

for index, file_name in enumerate(data_file_names):
	df = pd.read_csv(file_name, names=['overlap', 'preiters'])
	
	overlap = df['overlap']
	preiters = df['preiters']

	plt.plot(overlap, preiters, marker="o", label=names[index])

plt.legend()
plt.title("Multiplicative Schwarz Precondition - \nOverlap Size vs GMRES Iterations")
plt.xlabel('Overlap Size')
plt.ylabel("Iterations")
plt.show()

data_file_names = ['../data/multi_solver_res_time_partholey_square', '../data/multi_solver_res_time_partl_mesh', '../data/multi_solver_res_time_parttriangle']
names = ["Holey square", "L shape", "Triangley square"]

for index, file_name in enumerate(data_file_names):
	df = pd.read_csv(file_name, names=['partitions', 'iters', 'time'])
	
	partitions = df['partitions']
	iters = df['iters']

	plt.plot(partitions, iters, marker="o", label=names[index])

plt.legend()
plt.title("Multiplicative Schwarz Solver - \nNumber of Partitions vs Iterations")
plt.xlabel('Number of Partitions')
plt.ylabel("Iterations")
plt.show()


for index, file_name in enumerate(data_file_names):
	df = pd.read_csv(file_name, names=['partitions', 'iters', 'time'])
	
	partitions = df['partitions']
	time = df['time']

	plt.plot(partitions, time, marker="o", label=names[index])

plt.legend()
plt.title("Multiplicative Schwarz Solver - \nNumber of Partitions vs Time")
plt.xlabel('Number of Partitions')
plt.ylabel("Time")
plt.show()

data_file_names = ['../data/additive_schwarz_solve_holey_square.txt', '../data/additive_schwarz_solve_l_mesh.txt', '../data/additive_schwarz_solve_triangle.txt']
names = ["Holey square", "L mesh", "Triangle"]

for index, file_name in enumerate(data_file_names):
	df = pd.read_csv(file_name, names=['partitions', 'processors', 'iterations', 'time'])
	
	time = df['time']
	processors = df['processors']

	plt.plot(processors, time, marker="o", label=names[index])

plt.legend()
plt.title("Parallelised Additive Schwarz Precondition - \nProcessors vs Time")
plt.xlabel('Processors')
plt.ylabel("Time (s)")
plt.show()
