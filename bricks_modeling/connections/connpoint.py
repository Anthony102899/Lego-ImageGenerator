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
        scale_mat = np.identity(3)
        for i in range(3):
            if typeToBrick[self.type][1][i] != 0:
                scale_mat[i][i] *= typeToBrick[self.type][3]
        matrix = scale_mat
        rot_mat = rot_matrix_from_vec_a_to_b(typeToBrick[self.type][1], self.orient)
        matrix = matrix @ rot_mat
        offset = rot_mat @ np.array(typeToBrick[self.type][2])
        text = (
            f"1 5 {self.pos[0] + offset[0]} {self.pos[1] + offset[1]} {self.pos[2] + offset[2]} "
            + f"{matrix[0][0]} {matrix[0][1]} {matrix[0][2]} "
            + f"{matrix[1][0]} {matrix[1][1]} {matrix[1][2]} "
            + f"{matrix[2][0]} {matrix[2][1]} {matrix[2][2]} "
            + typeToBrick[self.type][0]
        )
        return text


if __name__ == "__main__":
    point = CPoint(np.array([0, 0, 0]), np.array([0, 1, 0]), ConnPointType.AXLE)
    print(point)
