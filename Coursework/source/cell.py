'''
Contains Cell Class
'''
class Cell(object):
    '''
    Cell object contains global point indices that make
    up the cell, the cell type, and the cell id.
    '''
    def __init__(self, points, cell_type, cell_id):
        self.points = points
        self.cell_type = cell_type
        self.cell_id = cell_id
    def get_points(self):
        '''Returns point ids that make up cell'''
        return self.points

    def get_cell_type(self):
        '''Returns the cell type'''
        return self.cell_type

    def get_cell_id(self):
        '''Returns the cell id'''
        return self.get_cell_id
