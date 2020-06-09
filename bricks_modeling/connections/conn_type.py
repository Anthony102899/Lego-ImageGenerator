import enum
from bricks_modeling.connections.connpointtype import ConnPointType
from bricks_modeling.connections.connpoint import CPoint
import numpy as np

class ConnType(enum.Enum):
   HOLE_PIN = 1
   HOLE_AXLE = 2
   CROSS_AXLE = 3
   BLOCK = 4

def compute_conn_type(c_point1: CPoint, c_point2: CPoint):
    if np.linalg.norm(c_point1.pos - c_point2.pos) < 1e-9:
        if np.linalg.norm(np.cross(c_point1.orient, c_point2.orient)) < 1e-9:
            if {c_point1.type, c_point2.type} == {ConnPointType.HOLE, ConnPointType.PIN}:
                return ConnType.HOLE_PIN
            else:
                #TODO: add more connection types
                print("unsupported connection type")
                return None
    elif abs(np.linalg.norm(c_point1.pos - c_point2.pos) - 1.0) < 0: # detect the case of contact
        # TODO: add connection types of brick inter-blocking
        print("unsupported connection type")
        return None
    else:
        return None