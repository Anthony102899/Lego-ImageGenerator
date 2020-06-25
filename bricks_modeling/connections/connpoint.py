import numpy as np

from bricks_modeling.connections.connpointtype import ConnPointType, typeToBrick
from util.geometry_util import rot_matrix_from_vec_a_to_b

#### abstraction of a point on a LEGO bricks, which can be either a hole, a pin, or an axle,
class CPoint:
    def __init__(self, pos, orient, type: ConnPointType):
        # local position of this connection point
        self.pos: np.ndarray = np.array(pos, dtype=np.float64)
        # local orientation of this connection point
        self.orient: np.ndarray = np.array(orient, dtype=np.float64)
        # type (hole, pin, axle, or axle hole)
        self.type = type

    def to_ldraw(self) -> str:
        rot_mat = rot_matrix_from_vec_a_to_b(typeToBrick[self.type][1], self.orient)
        offset = rot_mat @ np.array(typeToBrick[self.type][2])
        text = (
            f"1 5 {self.pos[0] + offset[0]} {self.pos[1] + offset[1]} {self.pos[2] + offset[2]} "
            + f"{rot_mat[0][0]} {rot_mat[0][1]} {rot_mat[0][2]} "
            + f"{rot_mat[1][0]} {rot_mat[1][1]} {rot_mat[1][2]} "
            + f"{rot_mat[2][0]} {rot_mat[2][1]} {rot_mat[2][2]} "
            + typeToBrick[self.type][0]
        )
        return text


if __name__ == "__main__":
    point = CPoint(np.array([0, 0, 0]), np.array([0, 1, 0]), ConnPointType.AXLE)
    print(point)
