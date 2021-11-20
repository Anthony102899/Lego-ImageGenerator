import os
import pickle

from bricks_modeling.bricks.bricktemplate import BrickTemplate
import bricks_modeling.file_IO.model_reader
from bricks_modeling.connectivity_graph import ConnectivityGraph
from visualization.model_visualizer import visualize_3D
import numpy as np
from matplotlib.patches import Polygon as MatPolygon
from shapely.geometry import Polygon, LineString, MultiLineString
import matplotlib.pyplot as plt



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
        # self.perimeter = np.sum(np.linalg.norm(self.edges[:, 1] - self.edges[:, 0], axis=1))


class Vertex:

    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.next = []

    def add_neighbor(self, vertex):
        self.next.append(vertex)


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


def plot_polygons(bricks):
    fig, ax = plt.subplots()
    for i in range(1, len(bricks)):
        brick = bricks[i]
        brick.template.use_vertices_edges2D()
        mesh_size = 25
        polygon = PolygonInstance(brick.trans_matrix, np.multiply(brick.template.edges2D, mesh_size))
        vertices = brick.trans_matrix.dot(
            np.insert(np.multiply(brick.template.vertices2D, mesh_size), 3, values=1, axis=1).T).T[:, [0, 2]]
        sorted_vertices = get_sorted_vertices(polygon, vertices)

        p = MatPolygon(sorted_vertices, facecolor='r')
        ax.add_patch(p)

    brick = bricks[0]
    brick.template.use_vertices_edges2D()
    mesh_size = 25
    polygon = PolygonInstance(brick.trans_matrix, np.multiply(brick.template.edges2D, mesh_size))
    vertices = brick.trans_matrix.dot(
        np.insert(np.multiply(brick.template.vertices2D, mesh_size), 3, values=1, axis=1).T).T[:, [0, 2]]
    sorted_vertices = get_sorted_vertices(polygon, vertices)

    p = MatPolygon(sorted_vertices, facecolor='k')
    ax.add_patch(p)

    ax.set_xlim([-500, 500])
    ax.set_ylim([0, 1000])
    plt.show()



def exam():
    bricks = bricks_modeling.read_bricks_from_file("./['43723'] base=12 n=290 r=1.ldr")
    structure_graph = ConnectivityGraph(bricks)
    # graph = AdjacencyGraph(bricks)
    for brick in bricks:
        brick.template.use_vertices_edges2D()
        print(brick.template.edges2D)

    mesh_size = 25
    polygon_1 = PolygonInstance(bricks[0].trans_matrix, np.multiply(bricks[0].template.edges2D, mesh_size))
    polygon_2 = PolygonInstance(bricks[1].trans_matrix, np.multiply(bricks[1].template.edges2D, mesh_size))
    vertices_1 = bricks[0].trans_matrix.dot(
        np.insert(np.multiply(bricks[0].template.vertices2D, mesh_size), 3, values=1, axis=1).T).T[:, [0, 2]]
    vertices_2 = bricks[1].trans_matrix.dot(
        np.insert(np.multiply(bricks[1].template.vertices2D, mesh_size), 3, values=1, axis=1).T).T[:, [0, 2]]
    sorted_vertices_1 = get_sorted_vertices(polygon_1, vertices_1)
    sorted_vertices_2 = get_sorted_vertices(polygon_2, vertices_2)

    p1 = MatPolygon(sorted_vertices_1, facecolor='k')
    p2 = MatPolygon(sorted_vertices_2, facecolor='k')
    fig, ax = plt.subplots()
    ax.add_patch(p1)
    ax.add_patch(p2)
    ax.set_xlim([0, 1000])
    ax.set_ylim([0, 1000])
    plt.show()

    poly1 = Polygon(sorted_vertices_1)
    poly2 = Polygon(sorted_vertices_2)

    intersection = poly1.intersection(poly2)
    print(intersection)
    if isinstance(intersection, Polygon):
        if not intersection.is_empty:
            print("collide!")
    elif isinstance(intersection, LineString) or isinstance(intersection, MultiLineString):
        print("touch!")
    else:
        print("No relationship")

    print(compute_polygon_touch_length(polygon_1, polygon_2))

    points = [b.get_translation() for b in structure_graph.bricks]

    edges = [e["node_indices"] for e in structure_graph.connect_edges]
    visualize_3D(points, lego_bricks=bricks, edges=edges, show_axis=True)


def get_sorted_vertices(polygon, vertices):
    vertices_dict = {}
    for vertex in vertices:
        vertices_dict[str(vertex)] = Vertex(vertex)
    for edge in polygon.edges:
        vertices_dict[str(edge[0])].add_neighbor(vertices_dict[str(edge[1])])
        vertices_dict[str(edge[1])].add_neighbor(vertices_dict[str(edge[0])])
    initial = vertices_dict[str(vertices[0])]
    current = initial
    sorted_vertices = []
    previous = None
    for i in range(len(vertices)):
        sorted_vertices.append(current.coordinates)
        neighbor_pass = False
        for neighbor in current.next:
            if not neighbor_pass and (previous == None or neighbor == previous):
                neighbor_pass = True
                continue
            if neighbor == initial:
                continue
            previous = current
            current = neighbor
            break
    return np.array(sorted_vertices)


def collide_connect_2D(brick_1, brick_2):
    if brick_1.trans_matrix[1][3] != brick_2.trans_matrix[1][3]:
        return 0
    brick_1.template.use_vertices_edges2D()
    brick_2.template.use_vertices_edges2D()
    mesh_size = 25
    polygon_1 = PolygonInstance(brick_1.trans_matrix, np.multiply(brick_1.template.edges2D, mesh_size))
    polygon_2 = PolygonInstance(brick_2.trans_matrix, np.multiply(brick_2.template.edges2D, mesh_size))
    vertices_1 = brick_1.trans_matrix.dot(
        np.insert(np.multiply(brick_1.template.vertices2D, mesh_size), 3, values=1, axis=1).T).T[:, [0, 2]]
    vertices_2 = brick_2.trans_matrix.dot(
        np.insert(np.multiply(brick_2.template.vertices2D, mesh_size), 3, values=1, axis=1).T).T[:, [0, 2]]
    sorted_vertices_1 = get_sorted_vertices(polygon_1, vertices_1)
    sorted_vertices_2 = get_sorted_vertices(polygon_2, vertices_2)

    """p1 = MatPolygon(sorted_vertices_1, facecolor='k')
    p2 = MatPolygon(sorted_vertices_2, facecolor='k')
    fig, ax = plt.subplots()
    ax.add_patch(p1)
    ax.add_patch(p2)
    ax.set_xlim([70, 300])
    ax.set_ylim([100, 300])
    plt.show()"""

    poly1 = Polygon(sorted_vertices_1)
    poly2 = Polygon(sorted_vertices_2)

    intersection = poly1.intersection(poly2)
    if isinstance(intersection, Polygon):
        if not intersection.is_empty:
            # print("collide!")
            return -1
        else:
            return 0
    else:
        return round(compute_polygon_touch_length(polygon_1, polygon_2), 2)
    """elif isinstance(intersection, LineString) or isinstance(intersection, MultiLineString):
            # print("touch!")
            return compute_polygon_touch_length(polygon_1, polygon_2)"""


def group_display(bricks, color, depict_base=False, base=None):
    fig, ax = plt.subplots()
    """brick = bricks[0]
    brick.template.use_vertices_edges2D()
    mesh_size = 25
    polygon = PolygonInstance(brick.trans_matrix, np.multiply(brick.template.edges2D, mesh_size))
    vertices = brick.trans_matrix.dot(
        np.insert(np.multiply(brick.template.vertices2D, mesh_size), 3, values=1, axis=1).T).T[:, [0, 2]]
    sorted_vertices = get_sorted_vertices(polygon, vertices)

    p = MatPolygon(sorted_vertices, facecolor='b')
    ax.add_patch(p)"""
    if depict_base:
        for brick in base:
            brick.template.use_vertices_edges2D()
            mesh_size = 25
            polygon = PolygonInstance(brick.trans_matrix, np.multiply(brick.template.edges2D, mesh_size))
            vertices = brick.trans_matrix.dot(
                np.insert(np.multiply(brick.template.vertices2D, mesh_size), 3, values=1, axis=1).T).T[:, [0, 2]]
            sorted_vertices = get_sorted_vertices(polygon, vertices)

            p = MatPolygon(sorted_vertices, facecolor='b')
            ax.add_patch(p)

    for i in range(0, len(bricks)):
        brick = bricks[i]
        brick.template.use_vertices_edges2D()
        mesh_size = 25
        polygon = PolygonInstance(brick.trans_matrix, np.multiply(brick.template.edges2D, mesh_size))
        vertices = brick.trans_matrix.dot(
            np.insert(np.multiply(brick.template.vertices2D, mesh_size), 3, values=1, axis=1).T).T[:, [0, 2]]
        sorted_vertices = get_sorted_vertices(polygon, vertices)

        p = MatPolygon(sorted_vertices, facecolor=color)
        ax.add_patch(p)

    """ brick = bricks[1]
    brick.template.use_vertices_edges2D()
    mesh_size = 25
    polygon = PolygonInstance(brick.trans_matrix, np.multiply(brick.template.edges2D, mesh_size))
    vertices = brick.trans_matrix.dot(
        np.insert(np.multiply(brick.template.vertices2D, mesh_size), 3, values=1, axis=1).T).T[:, [0, 2]]
    sorted_vertices = get_sorted_vertices(polygon, vertices)

    p = MatPolygon(sorted_vertices, facecolor='r')
    ax.add_patch(p)"""

    ax.set_xlim([-500, 500])
    ax.set_ylim([0, 1000])
    plt.show()


def align_detection():
    path = os.path.dirname(__file__) + "/connectivity/['43722', '43723'] base=24 t=1281.55.pkl"
    structure_graph = pickle.load(open(path, "rb"))
    positive_align = []
    negative_align = []
    for brick in structure_graph.bricks:
        if brick.trans_matrix[0][3] == 1 and brick.trans_matrix[2][3] == 1:
            positive_align.append(brick)
        if brick.trans_matrix[0][3] == -1 and brick.trans_matrix[2][3] == -1:
            negative_align.append(brick)
    group_display(positive_align, 'r')
    group_display(negative_align, 'k')





if __name__ == "__main__":
    """polygon_1 = PolygonInstance(TRANSFORM_MATRIX_1, EDGE_TEMPLATE)
    polygon_2 = PolygonInstance(TRANSFORM_MATRIX_2, EDGE_TEMPLATE)
    print(compute_polygon_touch_length(polygon_1, polygon_2))
    pass"""
    #exam()
    align_detection()
