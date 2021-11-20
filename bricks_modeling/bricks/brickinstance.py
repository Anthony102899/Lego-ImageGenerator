
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


"""
    This file is about bricks collision & connectivity & translation & rotation
"""
# Todo: what's the meaning of corner position and what's the meaning of argument four_point
# resolved: corner position is a pair of opposite vertices of a bounding box
# four_point means we need to use four corners to represent the bounding box
# return a list of bbox corners
def get_corner_pos(brick, four_point=False):
    bbox_ls = brick.get_col_bbox()
    cub_corner = []
    if four_point:
        corner_transform = np.array([[1, 1, 1], [1, 1, -1], [-1, 1, -1], [-1, 1, 1]])
    else:
        corner_transform = np.array([[1, 1, 1], [-1, -1, -1]])
    for bbox in bbox_ls:
        cuboid_center = np.array([bbox["Dimension"][0] / 2, bbox["Dimension"][1] / 2, bbox["Dimension"][2] / 2])
        if four_point:
            cuboid_corner_relative = (np.tile(cuboid_center, (4, 1))) * corner_transform
        else:
            cuboid_corner_relative = (np.tile(cuboid_center, (2, 1))) * corner_transform
        cub_corners_pos = np.array(bbox["Rotation"] @ cuboid_corner_relative.transpose()).transpose() + np.array(bbox["Origin"])
        cub_corner.append(cub_corners_pos[0])
        cub_corner.append(cub_corners_pos[1])
        if four_point:
            cub_corner.append(cub_corners_pos[2])
            cub_corner.append(cub_corners_pos[3])
    return cub_corner

class BrickInstance:
    def __init__(self, template: BrickTemplate, trans_matrix, color=15):
        self.template = template
        self.trans_matrix = trans_matrix
        self.color = color
    
    def get_col_bbox(self):
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
    
    def get_brick_bbox(self):
        corner_pos = np.array(get_corner_pos(self))
        max_x = np.amax(corner_pos[:,0])
        min_x = np.amin(corner_pos[:,0])
        max_y = np.amax(corner_pos[:,1])
        min_y = np.amin(corner_pos[:,1])
        max_z = np.amax(corner_pos[:,2])
        min_z = np.amin(corner_pos[:,2])
        origin = [(max_x + min_x) / 2, (max_y + min_y) / 2, (max_z + min_z) / 2]
        dim = [max_x - min_x, max_y - min_y, max_z - min_z]
        return {"Origin": origin, "Rotation": np.identity(3), "Dimension": dim}

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
                """self_c_points = self.get_current_conn_points()
                other_c_points = other.get_current_conn_points()
                for i in range(len(self_c_points)):
                    if self_c_points[i] not in other_c_points: # cpoint is not the same
                        return False
                if len(self_c_points) == 1:
                    return False
                return True"""
                """self.template.use_vertices_edges2D()
                other.template.use_vertices_edges2D()
                template_peri = self.template.perimeter * 25
                mesh_size = 25
                polygon_1 = PolygonInstance(self.trans_matrix, np.multiply(self.template.edges2D, mesh_size))
                polygon_2 = PolygonInstance(other.trans_matrix, np.multiply(other.template.edges2D, mesh_size))
                template_peri -= compute_polygon_touch_length(polygon_1, polygon_2)
                # print(template_peri)
                if template_peri < 1e-6:
                    print("Removed")
                    return True
                return False"""
                self.template.use_vertices_edges2D()
                other.template.use_vertices_edges2D()
                
                v_1 = np.multiply(self.template.vertices2D, 25)
                v_2 = np.multiply(other.template.vertices2D, 25)
                v_1 = self.trans_matrix.dot(np.insert(v_1, 3, 1, 1).T).T[:, [0, 2]]
                v_2 = other.trans_matrix.dot(np.insert(v_2, 3, 1, 1).T).T[:, [0, 2]]

                for vec1 in v_1:
                    flag = False
                    for vec2 in v_2:
                        if np.linalg.norm(vec1 - vec2)< 1e-6:
                            flag = True
                            break
                    if flag:
                        continue
                    else:
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
        self_brick_bbox = self.get_brick_bbox()
        other_brick_bbox = other.get_brick_bbox()
        if not cuboid_col.cub_collision_detect(self_brick_bbox, other_brick_bbox):
            return False

        self_cp_bbox = self.get_col_bbox()
        other_cp_bbox = other.get_col_bbox()
        for bb1, bb2 in iter.product(self_cp_bbox, other_cp_bbox):
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
        mesh.scale(25, center=(0, 0, 0))
        mesh.rotate(self.get_rotation().tolist(), [0, 0, 0])
        mesh.translate([i for i in self.get_translation().tolist()])
        return mesh


def is_parallel(vec1, vec2):
    vec1 = np.round(vec1 / np.linalg.norm(vec1), 2)
    vec2 = np.round(vec2 / np.linalg.norm(vec2), 2)
    if np.equal(vec1, vec2).all() or np.equal(vec1, -1 * vec2).all():
        return True
    else:
        return False


def parallel_relative(vec1, vec2):
    rate = np.linalg.norm(vec1) / np.linalg.norm(vec2)
    if rate == 0:
        return 0
    vec1 = np.round(vec1 / np.linalg.norm(vec1), 2)
    vec2 = np.round(vec2 / np.linalg.norm(vec2), 2)
    if np.equal(vec1, vec2).all():
        return rate
    else:
        return -rate


def compute_edge_touch_length(edges_1, edges_2):
    a, b = edges_1
    c, d = edges_2

    if not is_parallel(b - a, d - c):
        return 0

    B = b - a
    C = c - a
    D = d - a
    BC = c - b
    BD = d - b
    BC_B = parallel_relative(BC, B)
    BD_B = parallel_relative(BD, B)

    if np.equal(a, c).all():
        if not is_parallel(D, B):
            return 0
    else:
        if not is_parallel(C, B):
            return 0

    if BD_B >= 0:
        if BC_B >= 0:
            return 0

        if BC_B < 0:
            if abs(BC_B) <= 1:
                return np.linalg.norm(BC)

            if abs(BC_B) > 1:
                return np.linalg.norm(B)

    if BD_B < 0 and abs(BD_B) < 1:
        if BC_B >= 0:
            return np.linalg.norm(BD)

        if BC_B < 0:
            if abs(BC_B) <= 1:
                return np.linalg.norm(C - D)

            if abs(BC_B) > 1:
                return np.linalg.norm(D)

    if BD_B < 0 and abs(BD_B) >= 1:
        if BC_B >= 0:
            return np.linalg.norm(B)

        if BC_B < 0:
            if abs(BC_B) <= 1:
                return np.linalg.norm(C)

            if abs(BC_B) > 1:
                return 0


class PolygonInstance:

    def __init__(self, transform_matrix, edges_list):
        edges_list = np.insert(edges_list, 3, values=1, axis=2)
        self.edges = edges_list
        for i in range(len(edges_list)):
            self.edges[i] = transform_matrix.dot(edges_list[i].T).T
        self.edges = self.edges[:, :, [0, 2]]
        # self.perimeter = np.sum(np.linalg.norm(self.edges[:, 1] - self.edges[:, 0], axis=1))


def compute_polygon_touch_length(polygon_1, polygon_2):
    result = 0
    for edges_1 in polygon_1.edges:
        for edges_2 in polygon_2.edges:
            result += compute_edge_touch_length(edges_1, edges_2)
    return result


if __name__ == "__main__":
    from bricks_modeling.file_IO.model_reader import read_bricks_from_file
    from bricks_modeling.file_IO.model_writer import write_bricks_to_file
    from bricks_modeling.connectivity_graph import ConnectivityGraph
    
    bricks = read_bricks_from_file("")

    for i in range(len(bricks)):
        for j in range(len(bricks)):
            if not i == j and i > j:
                collide = bricks[i].collide(bricks[j])
                connect = bricks[i].connect(bricks[j]) and (not collide)
                print(f"{i}=={j}: ",bricks[i] == bricks[j])
                print(f"{i} collide with {j}: ", collide,"\n")
                print(f"{i} connect with {j}: ", connect,"\n")