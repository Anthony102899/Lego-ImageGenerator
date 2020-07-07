import numpy as np

from bricks_modeling.bricks.bricktemplate import BrickTemplate
from bricks_modeling.connections.connpoint import CPoint
import util.geometry_util as geo_util
import util.cuboid_geometry as cu_geo
import itertools as iter


class BrickInstance:
    def __init__(self, template: BrickTemplate, trans_matrix, color=15):
        self.template = template
        self.trans_matrix = trans_matrix
        self.color = color

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, BrickInstance) and self.template.id == other.template.id:
            if (
                np.max(self.trans_matrix - other.trans_matrix)
                - np.min(self.trans_matrix - other.trans_matrix)
                < 1e-6
            ):
                return True
            else:
                self_c_points = self.get_current_conn_points()
                other_c_points = other.get_current_conn_points()
                for i in range(len(self_c_points)):
                    if self_c_points[i] not in other_c_points:
                        return False
                return True
        else:
            return False

    # TODO: haven't test yet
    # return one of the spatial relation: {seperated, connected, collision, same(fully overlaped)}
    def collide(self, other):
        self_c_points = self.get_current_conn_points()
        other_c_points = other.get_current_conn_points()
        for p_self, p_other in iter.product(self_c_points, other_c_points):
            if cu_geo.collosion_detect(p_self.get_cuboid(), p_other.get_cuboid()):
                return True
        return False

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
            conn_point_orient = geo_util.vec_local2world(
                self.trans_matrix[:3, :3], cp.orient
            )
            conn_point_bi_orient = geo_util.vec_local2world(
                self.trans_matrix[:3, :3], cp.bi_orient
            )
            conn_point_position = geo_util.point_local2world(
                self.trans_matrix[:3, :3], self.trans_matrix[:3, 3], cp.pos
            )
            conn_points.append(
                CPoint(
                    conn_point_position,
                    conn_point_orient,
                    conn_point_bi_orient,
                    cp.type,
                )
            )

        return conn_points

if __name__ == "__main__":
    from bricks_modeling.file_IO.model_reader import read_bricks_from_file
    from bricks_modeling.file_IO.model_writer import write_bricks_to_file
    from bricks_modeling.connectivity_graph import ConnectivityGraph
    from util.debugger import MyDebugger

    # test the equal function
    debugger = MyDebugger("test")
    bricks = read_bricks_from_file(r"F:\workspace\lego-solver\debug\2020-07-07_16-50-00_test\test1.ldr")
    i= 0
    j = 1
    print(f"{i},{j}: ",bricks[i] == bricks[j])