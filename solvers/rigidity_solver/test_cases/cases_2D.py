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

# a rigid square
def case_2():
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    fixed_points_index = []
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    abstract_edges = []
    return points, fixed_points_index, edges, abstract_edges

# a hinge
def case_3():
    points = np.array([[0, 0], [1, 0], [0, 2]])
    fixed_points_index = []
    edges = [(0, 1), (0, 2)]
    abstract_edges = []
    return points, fixed_points_index, edges, abstract_edges

# a hinge with one edge fixed
def case_3_1():
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
def case_4_1():
    points = np.array([[0, 0], [1, 0], [0, 1],[0, 1]])
    fixed_points_index = [0, 1]
    edges = [(0, 1), (1, 2), (0, 3)]
    abstract_edges = []
    return points, fixed_points_index, edges, abstract_edges

# a triangle with one point not connected and one edge fixed. The dangling point can slide over adjacent edge
def case_5():
    points = np.array([[0, 0], [1, 0], [0, 1],[0, 1]])
    fixed_points_index = [0, 1]
    edges = [(0, 1), (1, 2), (0, 3)]
    abstract_edges = [(2, 3, 1.0, 0.0)]
    return points, fixed_points_index, edges, abstract_edges

def case_5_1():
    points = np.array([[0, 0], [1, 0], [0, 1], [0, 1]])
    fixed_points_index = [0, 1, 3]
    edges = [(0, 1), (1, 2), (0, 3)]
    abstract_edges = [(2, 3, 1.0, 0.0)]
    return points, fixed_points_index, edges, abstract_edges

# a bar-shaped truss structure
def case_6():
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

# a triangular shaped truss structure
def case_7():
    points = np.array([[0, 0], [1, 0], [2, 0], [4, 0], [5, 0], [6, 0], [1, 1], [2, 2], [3, 3], [4, 2], [5, 1]])
    fixed_points_index = [0, 5]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 5), (1, 6), (2, 7),
             (3, 9), (4, 10), (2, 6), (2, 8), (3, 8), (4, 9)]
    abstract_edges = []
    return points, fixed_points_index, edges, abstract_edges

# a pump mechanism
def case_8():
    points = np.array([[0, 0], [1, -1], [1, 1], [1, 1]])
    fixed_points_index = [1, 3]
    edges = [(0, 1), (0, 2)]
    abstract_edges = [(2, 3, 1.0, 0.0)]
    return points, fixed_points_index, edges, abstract_edges

# a robot leg
def case_9():
    points = np.array([[0, 0], [4, 0], [2, -1], [-1, -1], [1, -2], [3, -1], [0, -2], [-2, -4], [2, -2.5], [3.5, -4]])
    fixed_points_index = [0,1]
    edges = [(0, 1), (0, 2), (1, 2), (0, 3), (2, 4), (1, 5), (3, 7), (6, 7), (4, 6), (4, 8), (8, 9), (5, 9), (3, 4),
             (4, 5)]
    abstract_edges = []
    return points, fixed_points_index, edges, abstract_edges

def case_10():
    height = 3
    points = [[1,1],[0,2],[2,2],[2,0],[0,6],[2,6],[1,4],[0,10],[2,10],[1,8],[0,14],[2,14],[1,12],[0,18],[2,18],[1,16],[0,22],[2,22],[1,20],[0,26],[2,26],[1,24],
                       [6,0],[6,2],[4,1],[10,0],[10,2],[8,1],[14,0],[14,2],[12,1],[18,0],[18,2],[16,1],[22,0],[22,2],[20,1],[26,0],[26,2],[24,1]]
    fixed_points_index = []
    edges = [(0,1),(0,2),(0,3),(1,2),(2,3),(1,4),(4,5),(2,5),(1,6),(4,6),(5,6),(2,6),(4,7),(7,8),(5,8),(4,9),(7,9),(8,9),(5,9),(7,10),(10,11),(8,11),(7,12),(10,12),(11,12),(8,12),
             (10,13),(13,14),(14,11),(15,10),(15,13),(15,14),(15,11),(13,16),(16,17),(17,14),(16,18),(17,18),(13,18),(14,18),(16,19),(19,20),(20,17),(21,16),(21,19),(21,20),(21,17),
             (22,23),(2,23),(3,22),(24,2),(24,22),(24,23),(24,3),(23,26),(26,25),(22,25),(27,23),(27,26),(27,25),(27,22),(26,29),(29,28),(25,28),(30,26),(30,29),(30,28),(30,25),
             (29,32),(32,31),(31,28),(33,29),(33,32),(33,31),(33,28),(32,35),(34,35),(31,34),(36,32),(36,35),(36,34),(36,31),(35,38),(38,37),(37,34),(39,35),(39,38),(39,34),(39,37)]
    points_base_lower_slope = len(points)
    if height > 0:
        points += [[6 + 4 * i, 6 + 4 * (height - 2) - 4 * i] for i in range(height - 1)]
        points_base_upper_slope = len(points)
        points += [[4 + 4 * i, 4 + 4 * height  - 4 * i] for i in range(height + 1)]
        points_base_middle_base = len(points)
        points +=[[5 + 4 * i, 5 + 4 * (height - 1)  - 4 * i] for i in range(height)]
    if height == 1:
        edges += [(5, 23)]
        edges +=[(5,points_base_middle_base),(23,points_base_middle_base)]
    if height > 1:
        edges += [(8 + 3 * (height - 2), points_base_lower_slope)]
        edges += [(i, i + 1) for i in range(points_base_lower_slope, points_base_upper_slope-1)]
        edges += [(points_base_upper_slope - 1, 26 + 3 * (height - 2))]
        edges += [(8 + 3 * (height - 2),points_base_middle_base), (26 + 3 * (height - 2), len(points) - 1)]
        edges += [(points_base_lower_slope, points_base_middle_base), (points_base_upper_slope - 1, len(points) - 1)]
        edges += [(points_base_middle_base + 1 + i, points_base_lower_slope + i) for i in range(height - 2)]
        edges += [(points_base_middle_base + 1 + i, points_base_lower_slope + 1 + i) for i in range(height - 2)]

    if height > 0:
        edges += [(8 + 3 * (height - 1), points_base_upper_slope)]
        edges += [(i, i + 1) for i in range(points_base_upper_slope, points_base_middle_base - 1)]
        edges += [(26 + 3 * (height - 1), points_base_middle_base - 1)]

        edges += [(8 + 3 * (height - 2), points_base_upper_slope)]
        edges += [(points_base_middle_base - 1, 26 + 3 * (height - 2))]
        edges += [(points_base_lower_slope + i, points_base_upper_slope + 1 + i) for i in range(height - 1)]
        edges += [(points_base_middle_base + i, points_base_upper_slope + 1 + i) for i in range(height)]
        edges += [(points_base_middle_base + i, points_base_upper_slope + i) for i in range(height)]

    abstract_edges = []
    return np.array(points), fixed_points_index, edges, abstract_edges


# a rigid U shape
def case_11():
    points = np.array([[0, 1], [0, -1], [1,-0.5], [1, 0.5], [10,-1], [10,1]])
    fixed_points_index = []
    edges = [(0, 1), (1, 2), (2, 0), (2,4), (1,4), (1,3), (0,3), (0,5), (3,5)]
    abstract_edges = []
    return points, fixed_points_index, edges, abstract_edges

# another rigid U shape
def case_11_1():
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
def case_two_edges():
    points = np.array([[-1, 0], [1, 0], [2, 0], [3, 0]])
    edges =  [(0, 1), (2, 3)]
    fixed_points_index = []
    return points, fixed_points_index, edges, []

def case_seperate_parts():
    # points = np.array([[0, 0],[-1, 0], [0, 1]])
    points = np.array([[-1, 0], [0, 0], [0, 1]])
    points = np.append(points, np.array([[1, 0], [2, 0], [1, 1], [2, 1]]), axis=0)
    fixed_points_index = []
    edges =  [(0, 1), (1, 2), (2, 0)]
    edges += [(3, 4), (4, 5), (5, 3)]
    edges += [(5, 6), (4, 6)]
    abstract_edges = []
    return points, fixed_points_index, edges, abstract_edges