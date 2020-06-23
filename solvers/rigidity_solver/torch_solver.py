import json
from typing import List, Tuple

from bricks_modeling.connectivity_graph import ConnectivityGraph
from util.debugger import MyDebugger
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.bricks.brickinstance import BrickInstance

import numpy as np
import torch
from torch import Tensor

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

def build_transformation_from_node(vars, node_ind):
    brick = graph.bricks[node_ind]

    d   = vars[node_ind * 6    : node_ind * 6 + 3]
    phi = vars[node_ind * 6 + 3: node_ind * 6  + 6]
    sin_x, sin_y, sin_z = torch.sin(phi)
    cos_x, cos_y, cos_z = torch.cos(phi)

    yaw = torch.tensor([
        [cos_x, -sin_x, 0],
        [sin_x, cos_x, 0],
        [0, 0, 1]
    ], dtype=torch.double)
    pitch = torch.tensor([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y]
    ], dtype=torch.double)
    roll = torch.tensor([
        [1, 0, 0],
        [0, cos_z, -sin_z],
        [0, sin_z, cos_z]
    ], dtype=torch.double)

    vars_rotation = yaw.matmul(pitch).matmul(roll)
    rotation = vars_rotation.matmul(torch.tensor(brick.get_rotation(), dtype=torch.double))
    translation = d + torch.tensor(brick.get_translation(), dtype=torch.double)
    print(node_ind, translation, d)

    _transform = lambda point: rotation.matmul(point) + translation

    return _transform

def norm_sq(tensor: Tensor):
    return torch.dot(tensor, tensor)



def cost(vars):
    # vars -> length 6 * num_nodes
    c = torch.tensor(0.0, dtype=torch.double)

    transforms = [build_transformation_from_node(vars, ind) for ind in range(num_nodes)]

    def cost_for_edge(edge):
        ind1, ind2 = edge["node_indices"]
        local1 = torch.tensor(edge["properties"]["contact_point_1"], dtype=torch.double)
        local2 = torch.tensor(edge["properties"]["contact_point_2"], dtype=torch.double)

        toworld1 = transforms[ind1]
        toworld2 = transforms[ind2]

        world1 = toworld1(local1)
        world2 = toworld2(local2)
        return norm_sq(world1 - world2)

    costs = np.array([cost_for_edge(edge) for edge in graph.edges])
    c = costs.sum()

    return c

if __name__ == "__main__":
    print("torch solver")

    for i, brick in enumerate(graph.bricks):
        print(i, brick.get_translation())

    variables_set = np.identity(num_nodes * 6, dtype=np.double) / 100

    i = 7
    row = variables_set[i]

    vars = torch.tensor(row, requires_grad=True)
    c = cost(vars)
    c.backward()
    gradient = vars.grad.detach().numpy()
    print(i, c.detach().numpy(), gradient)
    print("------------")
