from pynbody.array import SimArray
from pymses.utils import constants as C

def from_pymses_constant(const):
    return SimArray(const.coeff, const._decompose_base_units())