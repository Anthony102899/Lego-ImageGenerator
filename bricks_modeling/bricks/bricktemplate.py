import numpy as np
import os
from bricks_modeling.connections.connpoint import CPoint
from bricks_modeling.connections.connpointtype import ConnPointType

from typing import Set


class BrickTemplate:
    def __init__(self, c_points: CPoint, ldraw_id: str):
        self.c_points = c_points
        self.id = ldraw_id
        self.vertices2D = self.get_vertices_2d()
        self.edges2D = self.get_edges_2d()

    def get_vertices_2d(self):
        obj_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "obj",
                                     f'{self.id + ".obj"}')
        vertices = []
        min_y = 0.0
        stud_vertices = []
        for line in open(obj_file_path, "r"):
            if line.startswith('v'):
                values = line.split(" ")
                if float(values[2]) < min_y:
                    min_y = float(values[2])
                if round(float(values[2]), 1) == 0.0:
                    v = [float(x) for x in values[1: 4]]
                    vertices.append(v)
        if min_y < 0:
            for line in open(obj_file_path, "r"):  # detect vertices of studs, which will not be count as 2d vertices
                if line.startswith('v'):
                    values = line.split(" ")
                    if float(values[2]) == min_y:
                        v = [float(x) for x in values[1: 4]]
                        v[1] = 0
                        stud_vertices.append(v)
        result = []
        for v in vertices:
            if result.count(v) == 0 and stud_vertices.count(v) == 0:
                result.append(v)
        return result

    def get_edges_2d(self):  # get the edges of brick by reading obj files
        obj_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database", "obj",
                                     f'{self.id + ".obj"}')
        indices = []
        pair_list = []
        faces = []
        edges = []
        line_count = 0
        for line in open(obj_file_path, "r"):  # read vertices. if vertex is in vertices2D, save the index of it.
            if line.startswith('v'):
                line_count = line_count + 1
                values = line.split(" ")
                if round(float(values[2]), 1) == 0.0:
                    v = [float(x) for x in values[1: 4]]
                    if self.vertices2D.count(v) != 0:
                        vertex_index_pair = (line_count, v)
                        indices.append(vertex_index_pair[0])
                        pair_list.append(vertex_index_pair)
        for line in open(obj_file_path, "r"):  # read faces. if a face is consist of 3 saved vertices, save the face.
            if line.startswith('f'):
                values = line.split(" ")
                face = [int(x) for x in values[1: 4]]
                flag = 1
                for x in face:
                    if indices.count(x) == 0:
                        flag = 0
                        break
                if flag == 1:
                    faces.append(face)
        for face in faces:  # get the coordinates of vertices for each tri-faces.
            vertex_coordinates = []
            tmp_edge_list = []
            for index in face:
                for pair in pair_list:
                    if pair[0] == index:  # find the coordinate for the vertex.
                        coordinate = pair[1]
                        vertex_coordinates.append(coordinate)
            for i in range(len(vertex_coordinates)):
                for j in range(i+1, len(vertex_coordinates)):
                    tmp_edge_list.append([vertex_coordinates[i], vertex_coordinates[j]])
            for edge in tmp_edge_list:
                flag = 1
                i = 0
                for exist_edge in edges:
                    if (exist_edge[0].__eq__(edge[0]) and exist_edge[1].__eq__(edge[1])) or (exist_edge[0].__eq__(edge[1]) and exist_edge[1].__eq__(edge[0])):
                        flag = 0
                        del edges[i]
                        break
                    i = i + 1
                if flag == 1:
                    edges.append(edge)
        return edges


    def __eq__(self, other):
        if isinstance(other, BrickTemplate):
            return self.id == other.id
        return False

    def deg1_cpoint_indices(self) -> Set[int]:
        """
        return a set of the indices of the c_points that have exactly one c_point having 1 lego distance to it

        Note: '1 lego distance' is hard-coded as 20 here temporarily, which stands for a beam.
        """
        from itertools import combinations

        deg_count = [0 for _ in range(len(self.c_points))]
        lego_dist = 20
        tol = 1e-6

        point_positions = [pt.pos for pt in self.c_points]
        # iterate over all pairs of conn_points of the instance
        for (i, p), (j, q) in combinations(enumerate(point_positions), 2):
            if (
                -tol < np.linalg.norm(p - q) - lego_dist < tol
            ):  # if distance within tolerance
                deg_count[i] += 1
                deg_count[j] += 1

        deg1set = {ind for ind, count in enumerate(deg_count) if count == 1}
        return deg1set


if __name__ == "__main__":
    """cpoints = [
        CPoint(np.array([0, 0, -1]), np.array([0, 1, 0]), ConnPointType.AXLE),
        CPoint(np.array([0, 0, 0]), np.array([0, 1, 0]), ConnPointType.AXLE),
        CPoint(np.array([0, 0, 1]), np.array([0, 1, 0]), ConnPointType.AXLE),
    ]
    brick = BrickTemplate(cpoints, ldraw_id="32523.dat")
    input("")"""
    brick = BrickTemplate([], ldraw_id="43723")

    print(brick.edges2D)
