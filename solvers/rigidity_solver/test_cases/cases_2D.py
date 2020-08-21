import numpy as np

# a rigid triangle
def case_1():
    points = np.array([[-1, 0], [1, 0], [0, 1.732]])
    fixed_points_index = []
    edges = [(0, 1), (1, 2), (2, 0)]
    abstract_edges = []
    return points, fixed_points_index, edges, abstract_edges

def case_1_1():
    points = np.array([[-1, 0], [1, 0], [0, 0.2]])
    fixed_points_index = []
    edges = [(0, 1), (1, 2), (2, 0)]
    abstract_edges = []
    return points, fixed_points_index, edges, abstract_edges

def case_1_2():
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    fixed_points_index = []
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
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

# a bar-shaped truss structure
def case_7():
    length = 8
    points = [[i, 0] for i in range(length)]
    points += [[i, 1] for i in range(length)]

    edges = [(i, i+length) for i in range(length)]
    edges += [(i, i + 1) for i in range(length-1)]
    edges += [(i+length, i+length + 1) for i in range(length - 1)]
    edges += [(i, i+length+1) for i in range(length - 1)]

    fixed_points_index = [0, length-1, length, length*2 -1]
    # fixed_points_index = []
    abstract_edges = []

    return np.array(points), fixed_points_index, edges, abstract_edges

# a rigid U shape
def case_8():
    points = np.array([[0, 1], [0, -1], [1,-0.5], [1, 0.5], [10,-1], [10,1]])
    fixed_points_index = []
    edges = [(0, 1), (1, 2), (2, 0), (2,4), (1,4), (1,3), (0,3), (0,5), (3,5)]
    abstract_edges = []
    return points, fixed_points_index, edges, abstract_edges

# another rigid U shape
def case_8_1():
    truss_length = 2

    points = [[0, 1], [0, -1], [1,-0.5], [1, 0.5], [10,-1], [10,1]]
    origin_length = len(points)

    points += [[-(i+1)*0.5,  1] for i in range(truss_length)]
    points += [[-(i+1)*0.5, -1] for i in range(truss_length)]

    edges = [(0, 1), (1, 2), (2, 0), (2, 4), (1, 4), (1, 3), (0, 3), (0, 5), (3, 5),
             (6,0), (6+truss_length,0), (6,1), (6+truss_length,1)]
    edges += [(i, i + truss_length) for i in range(origin_length, origin_length + truss_length)]
    edges += [(i, i + 1) for i in range(origin_length, origin_length + truss_length - 1)]
    edges += [(i + truss_length, i + truss_length + 1) for i in range(origin_length, origin_length + truss_length - 1)]
    edges += [(i, i + truss_length + 1) for i in range(origin_length, origin_length + truss_length - 1)]
    edges += [(i+1, i + truss_length) for i in range(origin_length, origin_length + truss_length - 1)]

    fixed_points_index = []
    abstract_edges = []

    return np.array(points), fixed_points_index, edges, abstract_edges