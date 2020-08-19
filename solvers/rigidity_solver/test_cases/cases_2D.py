import numpy as np

# a rigid triangle
def case_1():
    points = np.array([[0, 0], [1, 0], [0, 1]])
    fixed_points_index = []
    edges = [(0, 1), (1, 2), (2, 0)]
    abstract_edges = []
    return points, fixed_points_index, edges, abstract_edges

# a hinge
def case_2():
    points = np.array([[0, 0], [1, 0], [0, 2]])
    fixed_points_index = []
    edges = [(0, 1), (0, 2)]
    abstract_edges = []
    return points, fixed_points_index, edges, abstract_edges

# a hinge with one edge fiexed
def case_3():
    points = np.array([[0, 0], [1, 0], [0, 2]])
    fixed_points_index = [0, 1]
    edges = [(0, 1), (0, 2)]
    abstract_edges = []
    return points, fixed_points_index, edges, abstract_edges

# a triangle with one point not connected
def case_4():
    points = np.array([[0, 0], [1, 0], [0, 1],[0, 1]])
    fixed_points_index = []
    edges = [(0, 1), (1, 2), (0, 3)]
    abstract_edges = []
    return points, fixed_points_index, edges, abstract_edges

# a triangle with one point not connected and one edge fixed
def case_5():
    points = np.array([[0, 0], [1, 0], [0, 1],[0, 1]])
    fixed_points_index = [0, 1]
    edges = [(0, 1), (1, 2), (0, 3)]
    abstract_edges = []
    return points, fixed_points_index, edges, abstract_edges

# a triangle with one point not connected and one edge fixed. The dangling point can slide over adjacent edge
def case_6():
    points = np.array([[0, 0], [1, 0], [0, 1],[0, 1]])
    fixed_points_index = [0, 1]
    edges = [(0, 1), (1, 2), (0, 3)]
    abstract_edges = [(2, 3, 1.0, 0.0)]
    return points, fixed_points_index, edges, abstract_edges