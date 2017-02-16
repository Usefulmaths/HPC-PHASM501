'''
This module deals with the importing and exporting of vtk file formats.
Specifically, the importing of grid data and the exporting of grid solution.
'''
from shutil import copyfile

# In pylint gives erros for not existing, but does exist.
from vtk import vtkUnstructuredGridReader


def import_vtk_file(file_name):
    '''Returns a vtkDataObject containing information about mesh.'''
    reader = vtkUnstructuredGridReader()
    reader.SetFileName(file_name)
    reader.Update()

    return reader.GetOutput()

def export_vtk_file(file_name, number_of_points, solution):
    '''Exports solution to diffusion into a vtk file.'''
    copyfile(file_name, file_name + "_solution.vtk")

    with open(file_name + '_solution.vtk', 'a') as output_file:
        output_file.write('\nPOINT_DATA ' + str(number_of_points) + "\n")
        output_file.write('SCALARS u double 1\n')
        output_file.write('LOOKUP_TABLE default\n')
        for scalar in solution:
            output_file.write(str(scalar.item(0)) + '\n')
