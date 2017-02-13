'''
This module deals with the importing and exporting of vtk file formats.
Specifically, the importing of grid data and the exporting of grid solution.
'''

from vtk import vtkUnstructuredGridReader

def import_vtk_file(file_name):
    '''Returns a vtkDataObject containing information about mesh.'''
    reader = vtkUnstructuredGridReader()
    reader.SetFileName(file_name)
    reader.Update()

    return reader.GetOutput()

def export_vtk_file(file_name):
    '''Exports solution to diffusion into a vtk file.'''
    return file_name
