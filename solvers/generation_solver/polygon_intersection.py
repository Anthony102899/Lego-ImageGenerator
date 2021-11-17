import numpy as np
from bricks_modeling.bricks.bricktemplate import BrickTemplate
from shapely.geometry import Polygon

"""EDGE_TEMPLATE = np.array([
    [[0, 1, -1.2], [0.8, 1, -1.2]],
    [[0.8, 1, -1.2], [0.8, 1, 1.2]],
    [[0.8, 1, 1.2], [0, 1, 1.2]],
    [[0, 1, 1.2], [-0.8, 1, 1.2]],
    [[-0.8, 1, 1.2], [0, 1, -1.2]]
])"""
EDGE_TEMPLATE = np.array(BrickTemplate([], ldraw_id="43723").edges2D)

TRANSFORM_MATRIX_1 = np.identity(4)

TRANSFORM_MATRIX_2 = np.array([
    [-1, 0, 0, 0.8],
    [0, 1, 0, 0],
    [0, 0, -1, 2.4],
    [0, 0, 0, 1]
])

class PolygonInstance:

    def __init__(self, transform_matrix, edges_list):
        edges_list = np.insert(edges_list, 3, values=1, axis=2)
        self.edges = edges_list
        for i in range(len(edges_list)):
            self.edges[i] = transform_matrix.dot(edges_list[i].T).T
        self.edges = self.edges[:, :, [0, 2]]
        self.perimeter = np.sum(np.linalg.norm(self.edges[:, 1] - self.edges[:, 0], axis=1))


def compute_polygon_touch_length(polygon_1, polygon_2):
    result = 0
    for edges_1 in polygon_1.edges:
        for edges_2 in polygon_2.edges:
            result += compute_edge_touch_length(edges_1, edges_2)
    return result

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

if __name__ == "__main__":
    polygon_1 = PolygonInstance(TRANSFORM_MATRIX_1, EDGE_TEMPLATE)
    polygon_2 = PolygonInstance(TRANSFORM_MATRIX_2, EDGE_TEMPLATE)
    print(compute_polygon_touch_length(polygon_1, polygon_2))
    pass
