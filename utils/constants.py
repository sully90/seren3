from pynbody.array import SimArray
from pymses.utils import constants as C

def from_pymses_constant(const):
    unit_string = const._decompose_base_units().replace("^", "**").replace(".", " ")
    return SimArray(const.coeff, unit_string)