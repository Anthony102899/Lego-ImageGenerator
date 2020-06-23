import json
from typing import List, Tuple

from bricks_modeling.connectivity_graph import ConnectivityGraph
from util.debugger import MyDebugger
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.bricks.brickinstance import BrickInstance

import autograd.numpy as np
from autograd import grad, jacobian

from tqdm import tqdm

import torch

"""
June 11, 2020

prototyping......

Microsoft is a real jerk...
It must be a torture for most of developers to work with Windows and confront their ill-designed standard.

"""

model_name = "hinged_L"
bricks: List[BrickInstance] = read_bricks_from_file(
    f"./data/LEGO_models/full_models/{model_name}.ldr"
)
graph = ConnectivityGraph(bricks)
num_nodes = len(graph.bricks)
num_edges = len(graph.edges)

def build_rotation_for_node(vars, node_ind):
    phi = vars[node_ind * 6 + 3: node_ind * 6 + 6]
    sin_x, sin_y, sin_z = np.sin(phi)
    cos_x, cos_y, cos_z = np.cos(phi)
    yaw = np.array([
        [cos_x, -sin_x, 0],
        [sin_x, cos_x, 0],
        [0, 0, 1]
    ], dtype=np.double)
    pitch = np.array([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y]
    ], dtype=np.double)
    roll = np.array([
        [1, 0, 0],
        [0, cos_z, -sin_z],
        [0, sin_z, cos_z]
    ], dtype=np.double)

    vars_rotation = yaw @ pitch @ roll
    return vars_rotation

def build_translation_for_node(vars, node_ind):
    return vars[node_ind * 6: node_ind * 6 + 3]


def build_initial_transform(brick):
    rotation = np.array(brick.get_rotation(), dtype=np.double)
    translation = np.array(brick.get_translation(), dtype=np.double)

    _transform = lambda point: rotation @ point + translation

    return _transform

def norm_sq(arr):
    return np.dot(arr, arr)

def edge_cost(edge_ind, vars):
    edge = graph.edges[edge_ind]
    nind1, nind2 = edge["node_indices"]
    R1 = build_rotation_for_node(vars, nind1)
    R2 = build_rotation_for_node(vars, nind2)
    T1 = build_translation_for_node(vars, nind1)
    T2 = build_translation_for_node(vars, nind2)

    B1 = build_initial_transform(graph.bricks[nind1])
    B2 = build_initial_transform(graph.bricks[nind2])

    local1 = edge["properties"]["contact_point_1"]
    local2 = edge["properties"]["contact_point_2"]

    world1 = R1 @ B1(local1) + T1
    world2 = R2 @ B2(local2) + T2

    return norm_sq(world1 - world2)


def cost(vars):
    # vars -> length 6 * num_nodes
    ec = np.array([edge_cost(i, vars) for i in range(num_edges)])

    return ec.sum()

if __name__ == "__main__":
    print("gradient solver")

    # variables = np.zeros((num_nodes * 6, ), dtype=np.double)

    variables_set = np.identity(num_nodes * 6, dtype=np.double) / 100
    grad_cost = grad(cost)

    for vars in variables_set:
        c = cost(vars)

        gradient = grad_cost(vars)
        print(gradient)


