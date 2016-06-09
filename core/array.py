import numpy as np
from pymses.utils import constants as C
from pynbody import array

class SimArray(array.SimArray):
    '''
    Extends pynbody SimArray to add field_latex feature
    '''
    def get_pymses_unit(self):
        '''
        Returns pymses compatible units
        '''
        unit = 1.
        compontents = str(self.units).split(' ')
        for c in compontents:
            if '**' in c:
                dims = c.split('**')
                pymses_unit = np.power(C.Unit(dims[0]), float(dims[1]))
                unit *= pymses_unit
            else:
                unit *= C.Unit(c)
        return unit

    def set_field_name(self, field_name):
        self._field_name = field_name

    def get_field_name(self):
        return self._field_name

    def field_latex(self):
        return r"%s [%s]" % (self.get_field_name(), self.units.latex())

    def in_units(self, units):
        copy = self.copy()
        copy.convert_units(units)
        return copy