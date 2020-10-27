import numpy as np
import open3d as o3d
import os
import trimesh
from bricks_modeling.bricks.bricktemplate import BrickTemplate
from bricks_modeling.connections.connpoint import CPoint
from bricks_modeling.connections.conn_type import compute_conn_type
from bricks_modeling.database.ldraw_colors import color_phraser
import util.geometry_util as geo_util
import itertools as iter
import json
from util.geometry_util import get_random_transformation
from bricks_modeling.file_IO.util import to_ldr_format
from bricks_modeling import config
import util.cuboid_collision as cuboid_col

class BrickInstance:
    def __init__(self, template: BrickTemplate, trans_matrix, color=15):
        self.template = template
        self.trans_matrix = trans_matrix
        self.color = color
    
    def get_bbox(self):
        bbox = []
        brick_id = self.template.id
        brick_rot = self.get_rotation()
        brick_trans = self.get_translation()
        if os.path.exists(os.path.join(config.col_folder, f"{brick_id}.col")):
            for line in open(os.path.join(config.col_folder, f"{brick_id}.col")):
                line = (line.split(" "))[:17]
                line = [float(x) for x in line]
                init_orient = (np.array(line[2:11])).reshape((3,3))
                init_origin = np.array(line[11:14])
                init_dim = init_orient @ np.array(line[14:17])  # in (x,y,z) format

                origin = brick_rot @ init_origin + brick_trans
                rotation = brick_rot @ init_orient
                dim = init_dim * 2 + 1
                bbox.append({"Origin": origin, "Rotation": rotation, "Dimension": dim})
            return bbox
        else:
            return []

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, BrickInstance) and self.template.id == other.template.id:
            if (
                np.max(self.trans_matrix - other.trans_matrix)
                - np.min(self.trans_matrix - other.trans_matrix)
                < 1e-6
            ): # tranformation matrix the same
                return True
            else:
                self_c_points = self.get_current_conn_points()
                other_c_points = other.get_current_conn_points()
                for i in range(len(self_c_points)):
                    if self_c_points[i] not in other_c_points: # cpoint is not the same
                        return False
                if len(self_c_points) == 1:
                    return False
                return True
        else:
            return False

    def connect(self, other):
        for p_self, p_other in iter.product(self.get_current_conn_points(), other.get_current_conn_points()):
            if not compute_conn_type(p_self, p_other) == None:
                return True
        return False
    
    def collide(self, other):
        self_bbox = self.get_bbox()
        other_bbox = other.get_bbox()
        for bb1, bb2 in iter.product(self_bbox, other_bbox):
            if cuboid_col.cub_collision_detect(bb1, bb2):
                return True
        return False

    def to_ldraw(self):
        return to_ldr_format(self.color, self.trans_matrix, f"{self.template.id}.dat")

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

    def get_mesh(self):
        color_dict = color_phraser()
        obj_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "obj",f'{self.template.id + ".obj"}')
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
        mesh.scale(2.5, center=(0, 0, 0))
        mesh.rotate(self.get_rotation().tolist(), [0, 0, 0])
        mesh.translate([i for i in self.get_translation().tolist()])
        return mesh

if __name__ == "__main__":
    from bricks_modeling.file_IO.model_reader import read_bricks_from_file
    from bricks_modeling.file_IO.model_writer import write_bricks_to_file
    from bricks_modeling.connectivity_graph import ConnectivityGraph
    
    bricks = read_bricks_from_file("./debug/0 3024+54200 2.ldr") 
    for i in range(len(bricks)):
        for j in range(len(bricks)):
            if not i == j and i > j:
                collide = bricks[i].collide(bricks[j])
                connect = bricks[i].connect(bricks[j]) and (not collide)
                print(f"{i}=={j}: ",bricks[i] == bricks[j])
                print(f"{i} collide with {j}: ", collide,"\n")
                print(f"{i} connect with {j}: ", connect,"\n")