import enum

import numpy as np

from bricks_modeling.connections.connpoint import CPoint
from bricks_modeling.connections.connpointtype import ConnPointType


class ConnType(enum.Enum):
    HOLE_PIN = 1  # to insert a pin into a circular hole
    HOLE_AXLE = 2  # to insert an axle into a circular hole
    CROSS_AXLE = 3  # to insert an axle into a cross-shaped hole
    BLOCK = 4  # movement constraint bt inter-blocking
    STUD_TUBE = 5  # to insert a stud on a regular brick into a tube


def compute_conn_type(c_point1: CPoint, c_point2: CPoint):
    if np.linalg.norm(c_point1.pos - c_point2.pos) < 1e-9:
        if np.linalg.norm(np.cross(c_point1.orient, c_point2.orient)) < 1e-9:
            if {c_point1.type, c_point2.type} == {
                ConnPointType.HOLE,
                ConnPointType.PIN,
            }:
                return ConnType.HOLE_PIN
            elif {c_point1.type, c_point2.type} == {
                ConnPointType.CROSS_HOLE,
                ConnPointType.AXLE,
            }:
                return ConnType.CROSS_AXLE
            elif {c_point1.type, c_point2.type} == {
                ConnPointType.HOLE,
                ConnPointType.AXLE,
            }:
                return ConnType.HOLE_AXLE
            elif {c_point1.type, c_point2.type} == {
                ConnPointType.STUD,
                ConnPointType.TUBE,
            }:
                return ConnType.STUD_TUBE
            else:
                print("unsupported connection type!!!")
                print(c_point1.type.name, c_point2.type.name)
                return None
    elif (
        abs(np.linalg.norm(c_point1.pos - c_point2.pos) - 1.0) < 0
    ):  # detect the case of inter-blocking
        # TODO: add connection types of brick inter-blocking
        print("unsupported connection type")
        return None
    else:
        return None
