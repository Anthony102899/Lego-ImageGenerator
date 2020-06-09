from bricks.BrickTemplate import BrickTemplate
import numpy as np

from bricks.ConnPoint import CPoint

class Line(): # Line is an abstract line in a parts, such as a T sytle beam has a horizontal line and a vertial line


    def __init__(self):
        self.conn_points = []
        self.direction = np.array([0, 0, 0])

    def add_points(self, point:CPoint):
        self.conn_points.append(point)

    def get_end_points(self):
        start_point = (self.conn_points[0]).pos
        end_point = start_point
        length = 0
        if len(self.conn_points) == 1:
            return start_point, end_point
        for cp in self.conn_points:
            distance_from_start_point = np.linalg.norm(cp.pos - start_point)
            distance_from_end_point = np.linalg.norm(cp.pos - end_point)
            if distance_from_end_point > length:
                start_point = cp.pos
                length = distance_from_end_point
            elif distance_from_start_point >length:
                end_point = cp.pos
                length = distance_from_start_point
        return start_point, end_point

def hash(pos): #to hash a position of a point
    return (1000000 * pos[0][0] + 1000 * pos[0][1] + pos[0][2])

class Points(): #The end points of abstract lines
    def __init__(self):
        self.points = []
        self.current_index = -1
        self.points_to_index = {}

    def add(self, pos):
        if hash(pos) in self.points_to_index.keys():
            print("position already added")
        else:
            self.points.append(pos)
            self.current_index += 1
            self.points_to_index[hash(pos)] = self.current_index
            print(f"{pos} added as index {self.points_to_index[hash(pos)]}")

    def get_point_index(self,pos):
        return self.points_to_index[hash(pos)]

    def print(self):
        for point in self.points:
            print(f"Point{self.get_point_index(point)} with position {point}")

def hash_for_edge(edge):# To hash an edge to get its index

    return 1000 * edge[0] + edge[1]

class Edges():
    def __init__(self):
        self.edges = []
        self.current_index = -1
        self.edegs_to_index = {}

    def add(self, start_point, end_point, points:Points):
        self.current_index += 1
        edge = [points.points_to_index[hash(start_point)], points.points_to_index[hash(end_point)]]
        self.edges.append(edge)

        self.edegs_to_index[hash_for_edge(edge)] = self.current_index

    def get_edge_idnex(self, pos1, pos2,points:Points):
        #print(pos1)
        #print(pos2)
        if hash_for_edge([points.get_point_index(pos1),points.get_point_index(pos2)]) in self.edegs_to_index.keys():
            return self.edegs_to_index[hash_for_edge([points.get_point_index(pos1),points.get_point_index(pos2)])]
        else:
            return self.edegs_to_index[hash_for_edge([points.get_point_index(pos2),points.get_point_index(pos1)])]

    def get_edge_index_by_edge(self, edge):
        if hash_for_edge(edge) in self.edegs_to_index.keys():
            return self.edegs_to_index[hash_for_edge(edge)]
        else:
            return self.edegs_to_index[hash_for_edge(edge)]

    def print(self, points:Points):
        for edge in self.edges:
            #print(edge)
            print(f"Edge{self.get_edge_index_by_edge(edge)} is with points{edge[0]} and {edge[1]}")


class Pins():
    def __init__(self):
        self.pins = []

    def add(self, edge1, edge2, pin, points: Points, edges: Edges):
        pin_index = points.get_point_index(pin)
        edge1_index = edges.get_edge_idnex(edge1[0],edge1[1], points)
        edge2_index = edges.get_edge_idnex(edge2[0],edge2[1], points)
        self.pins.append([pin_index, edge1_index, edge2_index])

    def print(self):
        for pin in self.pins:
            print(f"pin is at point {pin[0]}, edge {pin[1]} insert in edge {pin[2]}")

class Axles():
    def __init__(self):
        self.axles = []

    def add(self, edge1, edge2, axle, points: Points, edges: Edges):
        axle_index = points.get_point_index(axle)
        edge1_index = edges.get_edge_idnex(edge1[0],edge1[1], points)
        edge2_index = edges.get_edge_idnex(edge2[0],edge2[1], points)
        self.axles.append([axle_index, edge1_index, edge2_index])

    def print(self):
        for pin in self.axles:
            print(f"axle is at point {pin[0]}, edge {pin[1]} insert in edge {pin[2]}")

class Anchors():
    def __init__(self):
        self.anchors = []

    def add(self, edge1, edge2, anchor, points:Points, edges:Edges):
        anchor_index = points.get_point_index(anchor)
        edge1_index = edges.get_edge_idnex(edge1[0],edge1[1],points)
        edge2_index = edges.get_edge_idnex(edge2[0],edge2[1],points)
        self.anchors.append([anchor_index, edge1_index, edge2_index])

    def print(self):
        for anchor in self.anchors:
            print(f"anchors is at point {anchor[0]}, joining edge {anchor[1]} and edge {anchor[2]}")
