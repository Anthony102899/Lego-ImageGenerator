import numpy as np
from bricks.ConnPointType import ConnPointType

#### abstraction of a point on a LEGO bricks, which can be either a hole, a pin, or an axle,
class CPoint():
    def __init__(self, pos, orient, type: ConnPointType):
        # local position of this connection point
        self.pos = pos
        # local orientation of this connection point
        self.orient = orient
        # type (hole, pin, axle, or axle hole)
        self.type = type

if __name__ == "__main__":
    point = CPoint(np.array([0,0,0]), np.array([0,1,0]), ConnPointType.AXLE)
    print(point)