import numpy as np
from util.geometry_util import trivial_basis

def z_static(point_count):
    constr = np.zeros((point_count, point_count * 3))
    constr[np.arange(point_count), np.arange(point_count) * 3 + 2] = 1
    return constr

