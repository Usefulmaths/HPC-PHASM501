from solver import Solver
from system import System

class CompareSolvers:
    '''
    This class allows the comparison between a list of different solvers.
    '''
    def __init__(self, solvers, expression, diffusion_type):
        self.solvers = solvers
        self.expression = expression
        self.diffusion_type = diffusion_type
        
    def generate_iterations(self, dimensions):
        '''
        Generates a dictionary of iterations for each solver in the list for a 
        given list of dimensions a system will be solved over.
        '''
        iterations_dict = {}
        
        for solv in self.solvers:
            if(solv['subsolver_type'] == None and solv['precondition_type'] == None):
                key = str(solv['solver_type'])

            elif(solv['precondition_type'] == None):
                key = str(solv['solver_type']) + ': ' + str(solv['subsolver_type'])

            elif(solv['subsolver_type'] == None):
                key = str(solv['solver_type']) + ': ' + str(solv['precondition_type'])
                
            else:
                key = str(solv['solver_type']) + ': ' + str(solv['subsolver_type']) + ': ' + str(solv['precondition_type'])

            iterations_dict[key] = []
            solver = Solver(solv['solver_type'], solv['subsolver_type'], solv['precondition_type'])
            

            for dim in dimensions:
                system = System(dim)
                system.create_system(self.expression, self.diffusion_type)
                solution, res = solver.solve(system)
                
                iterations_dict[key].append(len(res))
                
        return iterations_dict

    def generate_residuals(self, dimension):
        '''
        Generates a dictionary of residuals for every solver in the solver list
        for a given dimension of grid.
        '''
        res_dict = {}
        for solv in self.solvers:
            solver = Solver(solv['solver_type'], solv['subsolver_type'], solv['precondition_type'])
            system = System(dimension)
            system.create_system(self.expression, self.diffusion_type)
            solution, res = solver.solve(system)

            if(solv['subsolver_type'] == None and solv['precondition_type'] == None):
                key = str(solv['solver_type'])

            elif(solv['precondition_type'] == None):
                key = str(solv['solver_type']) + ': ' + str(solv['subsolver_type'])

            elif(solv['subsolver_type'] == None):
                key = str(solv['solver_type']) + ': ' + str(solv['precondition_type'])

            else:
                key = str(solv['solver_type']) + str(solv['subsolver_type']) + str(solv['precondition_type'])
            
            res_dict[key] = res

        return res_dict