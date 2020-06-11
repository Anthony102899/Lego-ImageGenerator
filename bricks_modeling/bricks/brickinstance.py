import numpy as np

from bricks_modeling.bricks.bricktemplate import BrickTemplate
from bricks_modeling.connections.connpoint import CPoint
import util.geometry_util as geo_util


class BrickInstance:
    def __init__(self, template: BrickTemplate, trans_matrix, color=15):
        self.template = template
        self.trans_matrix = trans_matrix
        self.color = color

    def to_ldraw(self):
        text = (
            f"1 {self.color} {self.trans_matrix[0][3]} {self.trans_matrix[1][3]} {self.trans_matrix[2][3]} "
            + f"{self.trans_matrix[0][0]} {self.trans_matrix[0][1]} {self.trans_matrix[0][2]} "
            + f"{self.trans_matrix[1][0]} {self.trans_matrix[1][1]} {self.trans_matrix[1][2]} "
            + f"{self.trans_matrix[2][0]} {self.trans_matrix[2][1]} {self.trans_matrix[2][2]} "
            + f"{self.template.id}.dat"
        )
        return text

    def rotate(self, rot_mat):
        self.trans_matrix[:3, :3] = np.dot(rot_mat, self.trans_matrix[:3, :3])

    def translate(self, trans_vec):
        self.trans_matrix[:3, 3:4] = self.trans_matrix[:3, 3:4] + np.reshape(
            trans_vec, (3, 1)
        )

    def get_rotation(self):
        return self.trans_matrix[:3, :3]

    def get_translation(self):
        return self.trans_matrix[:3, 3]

    def reset_transformation(self):
        self.trans_matrix = np.identity(4, dtype=float)

    def get_current_conn_points(self):
        conn_points = []

        for cp in self.template.c_points:
            conn_point_orient = geo_util.vec_local2world(self.trans_matrix[:3, :3], cp.orient)
            conn_point_position = geo_util.point_local2world(self.trans_matrix[:3, :3], self.trans_matrix[:3, 3], cp.pos)
            conn_points.append(CPoint(conn_point_position, conn_point_orient, cp.type))

        return conn_points
