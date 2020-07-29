import numpy as np
import open3d as o3d
from os import path
import trimesh
from bricks_modeling.bricks.bricktemplate import BrickTemplate
from bricks_modeling.connections.connpoint import CPoint
from bricks_modeling.connections.conn_type import compute_conn_type
import util.geometry_util as geo_util
import util.cuboid_geometry as cu_geo
import itertools as iter
import json

def get_concave(
    brick_database=[
        "regular_cuboid.json",
        "regular_plate.json",
        "regular_slope.json",
        "regular_other.json",
        "regular_circular.json"]):
    data = []
    for data_base in brick_database:
        database_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "bricks_modeling", "database", data_base)
        with open(database_file) as f:
            temp = json.load(f)
            data.extend(temp)
    concave = []
    for brick in data:
        if len(brick) > 2:
            if brick["concave"] == 1:
                concave.append(brick["id"])
    return concave

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

    # TODO: 11477, connect AND collide
    # return one of the spatial relation: {seperated, connected, collision, same(fully overlaped)}
    def collide(self, other):
        concave = get_concave()
        self_c_points = self.get_current_conn_points()
        other_c_points = other.get_current_conn_points()
        concave_connect = 0
        for p_self, p_other in iter.product(self_c_points, other_c_points):
            if cu_geo.cub_collision_detect(p_self.get_cuboid(), p_other.get_cuboid()):
                if not compute_conn_type(p_self, p_other) == None:
                    if self.template.id in concave or other.template.id in concave:
                        concave_connect = 1
                        continue
                    return 0
                return 1
        if concave_connect:
            return 0
        return -1

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

    def get_translation_for_mesh(self):
        return self.trans_matrix[:3, 3]/2.5

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

    def get_mesh(self, color_dict):
        obj_file_path = path.join(path.dirname(path.dirname(__file__)), "database", "obj",f'{self.template.id + ".obj"}')
        mesh = o3d.io.read_triangle_mesh(
            obj_file_path
        )
        mesh.compute_vertex_normals()
        if str(self.color) in color_dict.keys():
            mesh.paint_uniform_color(color_dict[str(self.color)])
        elif not str(self.color).isdigit():  # color stored in hex
            rgb_color = trimesh.visual.color.hex_to_rgba(self.color[3:])
            mesh.paint_uniform_color(list(map(lambda comp: comp / 255, rgb_color[:3])))
        else:
            print("warning, no such color in ldview, print red")
            mesh.paint_uniform_color([1, 0, 0])
        mesh.rotate(self.get_rotation().tolist(), [0, 0, 0])
        mesh.translate([i / 2.5 for i in self.get_translation().tolist()])
        return mesh

if __name__ == "__main__":
    from bricks_modeling.file_IO.model_reader import read_bricks_from_file
    from bricks_modeling.file_IO.model_writer import write_bricks_to_file
    from bricks_modeling.connectivity_graph import ConnectivityGraph
    from util.debugger import MyDebugger

    # test the equal function
    #debugger = MyDebugger("test")
    bricks = read_bricks_from_file(r"./solvers/generation_solver/test.ldr")
    for i in range(len(bricks)):
        for j in range(len(bricks)):
            #print(f"{i}=={j}: ",bricks[i] == bricks[j])
            if not i==j and i > j:
                print(f"{i}collide with{j}: ", bricks[i].collide(bricks[j]),"\n")