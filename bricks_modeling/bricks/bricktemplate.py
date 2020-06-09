from bricks_modeling.connections.connpoint import CPoint
from bricks_modeling.connections.connpointtype import ConnPointType
import numpy as np

class BrickTemplate():

    def __init__(self, c_points, ldraw_id: str):
        self.c_points = c_points
        self.id = ldraw_id

    def __eq__(self, other):
        if isinstance(other, BrickTemplate):
            return self.id == other.id
        return False


if __name__ == "__main__":
    cpoints = [CPoint(np.array([0, 0, -1]), np.array([0,1,0]), ConnPointType.AXLE),
                CPoint(np.array([0, 0, 0]), np.array([0, 1, 0]), ConnPointType.AXLE),
                CPoint(np.array([0, 0, 1]), np.array([0, 1, 0]), ConnPointType.AXLE)]
    brick = BrickTemplate(cpoints, ldraw_id="32523.dat")
    input("")
