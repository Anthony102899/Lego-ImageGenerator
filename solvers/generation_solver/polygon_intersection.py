from bricks_modeling.bricks.bricktemplate import BrickTemplate
from bricks_modeling.bricks.brickinstance import get_corner_pos
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.connectivity_graph import ConnectivityGraph
from visualization.model_visualizer import visualize_3D
import numpy as np


"""EDGE_TEMPLATE = np.array([
    [[0, 1, -1.2], [0.8, 1, -1.2]],
    [[0.8, 1, -1.2], [0.8, 1, 1.2]],
    [[0.8, 1, 1.2], [0, 1, 1.2]],
    [[0, 1, 1.2], [-0.8, 1, 1.2]],
    [[-0.8, 1, 1.2], [0, 1, -1.2]]
])"""
template = BrickTemplate([], ldraw_id="43723")
template.use_vertices_edges2D()
EDGE_TEMPLATE = np.array(template.edges2D)

TRANSFORM_MATRIX_1 = np.identity(4)

# This matrix has been proven correct
TRANSFORM_MATRIX_2 = np.array([
    [0.9487, 0, 0.3162, 1.17952],
    [0, 1, 0, 0],
    [-0.3162, 0, 0.9487, -0.1914],
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


def exam():
    bricks = read_bricks_from_file("./['43723'] base=12 n=290 r=1.ldr")
    structure_graph = ConnectivityGraph(bricks)
    # graph = AdjacencyGraph(bricks)
    for brick in bricks:
        brick.template.use_vertices_edges2D()
        print(brick.template.edges2D)

    mesh_size = 25
    polygon_1 = PolygonInstance(bricks[0].trans_matrix, np.multiply(bricks[0].template.edges2D, mesh_size))
    polygon_2 = PolygonInstance(bricks[1].trans_matrix, np.multiply(bricks[1].template.edges2D, mesh_size))
    print(compute_polygon_touch_length(polygon_1, polygon_2))

    points = [b.get_translation() for b in structure_graph.bricks]

    edges = [e["node_indices"] for e in structure_graph.connect_edges]
    visualize_3D(points, lego_bricks=bricks, edges=edges, show_axis=True)


if __name__ == "__main__":
    """polygon_1 = PolygonInstance(TRANSFORM_MATRIX_1, EDGE_TEMPLATE)
    polygon_2 = PolygonInstance(TRANSFORM_MATRIX_2, EDGE_TEMPLATE)
    print(compute_polygon_touch_length(polygon_1, polygon_2))
    pass"""
    exam()
