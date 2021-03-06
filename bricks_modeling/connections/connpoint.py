import numpy as np
from numpy import linalg as LA
from bricks_modeling.connections.connpointtype import ConnPointType, typeToBrick, typeToBoundingBox
from util.geometry_util import rot_matrix_from_vec_a_to_b, gen_lateral_vec, rot_matrix_from_two_basis


#### abstraction of a point on a LEGO bricks, which can be either a hole, a pin, or an axle,
class CPoint:
    def __init__(self, pos, orient, bi_orient, type: ConnPointType):
        # local position of this connection point
        self.pos: np.ndarray = np.array(pos, dtype=np.float64)
        # local orientation of this connection point
        self.orient: np.ndarray = np.array(orient, dtype=np.float64)
        # axis-aligned vector that perpendicular to the orient
        self.bi_orient: np.ndarray = np.array(bi_orient, dtype=np.float64)
        # type (hole, pin, axle, or axle hole)
        self.type = type

    def __eq__(self, other):
        """Overrides the default implementation"""
        if (
                isinstance(other, CPoint)
                and self.type == other.type
                and LA.norm(self.pos - other.pos) < 1e-4
        ):
            return True
        return False

    def get_cuboid(self):
        heights = typeToBoundingBox[self.type]
        dimension = heights[0] * self.orient + heights[1] * self.bi_orient + heights[2] * np.cross(self.orient,
                                                                                                   self.bi_orient)
        return {"Origin": self.pos, "Rotation": self._get_rotation_matrix(), "Dimension": dimension}

    def to_ldraw(self) -> str:
        scale_mat = np.identity(3)
        for i in range(3):
            scale_mat[i][i] *= typeToBrick[self.type][3][i]

        rot_mat = self._get_rotation_matrix()
        # rot_mat = rot_matrix_from_vec_a_to_b(typeToBrick[self.type][1], self.orient)
        matrix = rot_mat
        matrix = matrix @ scale_mat
        offset = rot_mat @ np.array(typeToBrick[self.type][2])
        text = (
                f"1 5 {self.pos[0] + offset[0]} {self.pos[1] + offset[1]} {self.pos[2] + offset[2]} "
                + f"{matrix[0][0]} {matrix[0][1]} {matrix[0][2]} "
                + f"{matrix[1][0]} {matrix[1][1]} {matrix[1][2]} "
                + f"{matrix[2][0]} {matrix[2][1]} {matrix[2][2]} "
                + typeToBrick[self.type][0]
        )
        return text

    def _get_rotation_matrix(self):
        return rot_matrix_from_two_basis(typeToBrick[self.type][1], gen_lateral_vec(typeToBrick[self.type][1]),
                                         self.orient, self.bi_orient)


if __name__ == "__main__":
    point = CPoint(np.array([0, 0, 0]), np.array([0, 1, 0]), np.array([1, 0, 0]), ConnPointType.AXLE)
    print(point)
