import numpy as np
import torch
from collections import namedtuple

Part = namedtuple("Part", "points, edges, index_offset")
Joint = namedtuple("Joint", "pivot, part1_ind, part2_ind, translation, rotation_center")

def empty(_):
    return None


def define():
    data = np.array([
        [0, 0],
        [0, 10],
        [10, 10],
        [10, 0],
    ])

    # mutable
    parameter_nodes = {
        "left-down": torch.tensor(data[0], dtype=torch.double),
        "left-up": torch.tensor(data[1], dtype=torch.double),
        "right-up": torch.tensor(data[2], dtype=torch.double),
        "right-down": torch.tensor(data[3], dtype=torch.double),
    }
    parameter_scalars = {}
    immutable = {}

    # for param in parameter_nodes.values():
    #     param.requires_grad_(True)

    node_connectivity = {
        "left": ("left-up", "left-down"),
        "right": ("right-up", "right-down"),
        "up": ("left-up", "right-up"),
    }



    part_map = {}


    joints = [
        Joint(lambda nm: nm["left-up"], "left", "up", empty, lambda nm: nm["left-up"]),
        Joint(lambda nm: nm["right-up"], "right", "up", empty, lambda nm: nm["right-up"]),
    ]


    return locals()
