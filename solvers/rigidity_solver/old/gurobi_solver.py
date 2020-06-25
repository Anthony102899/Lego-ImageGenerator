import json
import numpy as np
from typing import List, Tuple

from bricks_modeling.connectivity_graph import ConnectivityGraph
from util.debugger import MyDebugger
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.bricks.brickinstance import BrickInstance

from util.geometry_util import point_local2world, vec_local2world

import gurobipy as grb
from gurobipy import GRB

# for prototyping
if __name__ == "__main__":
    bricks: List[BrickInstance] = read_bricks_from_file(
        "./data/LEGO_models/full_models/cube7.ldr"
    )
    graph = ConnectivityGraph(bricks)

    num_nodes = len(graph.bricks)
    num_edges = len(graph.edges)

    model = grb.Model("lego")

    budget = 0.1

    energy = grb.QuadExpr()

    items = (
        ["d_x", "d_y", "d_z"]
        + [f"sin_phi_{m}" for m in "xyz"]
        + [f"cos_phi_{m}" for m in "xyz"]
    )
    lb = ([-GRB.INFINITY for _ in range(3)] + [-1.0 for _ in range(6)]) * num_nodes
    ub = ([GRB.INFINITY for _ in range(3)] + [1.0 for _ in range(6)]) * num_nodes

    from itertools import product

    name = [f"{i}_{item}" for i, item in product(range(num_nodes), items)]

    vars = model.addVars(num_nodes, items, lb=lb, ub=ub)

    sqaure_sum_exprs = [
        vars[i, f"sin_phi_{m}"] * vars[i, f"sin_phi_{m}"]
        + vars[i, f"cos_phi_{m}"] * vars[i, f"cos_phi_{m}"]
        == 1
        for i in range(num_nodes)
        for m in "xyz"
    ]

    x = model.addVar()
    model.addConstr(x * x * x <= 1)

    square_sum_constrs = model.addConstrs(expr for expr in sqaure_sum_exprs)

    def _dot(a, b):
        return sum(m * n for m, n in zip(a, b))

    def _matmul(a, b):
        bT = list(zip(*b))
        return [[_dot(a[i], bT[j]) for j in range(len(b))] for i in range(len(a))]

    def node_rotate(node_ind: int, point: np.ndarray) -> List[grb.LinExpr]:
        cos_x, sin_x = vars[node_ind, "cos_phi_x"], vars[node_ind, "sin_phi_x"]
        cos_y, sin_y = vars[node_ind, "cos_phi_y"], vars[node_ind, "sin_phi_y"]
        cos_z, sin_z = vars[node_ind, "cos_phi_z"], vars[node_ind, "sin_phi_z"]

        yaw = [[cos_x, -sin_x, 0], [sin_x, cos_x, 0], [0, 0, 1]]
        pitch = [[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]]
        roll = [[1, 0, 0], [0, cos_z, -sin_z], [0, sin_z, cos_z]]

        # rotation = _matmul(_matmul(yaw, pitch), roll)

        rotation = [
            [
                cos_x * cos_y,
                cos_x * sin_y * sin_z - sin_x * cos_z,
                cos_x * sin_y * cos_z + sin_x * sin_z,
            ],
            [
                sin_x * cos_y,
                sin_x * sin_y * sin_z + cos_x * cos_z,
                sin_x * sin_y * cos_z - cos_x * sin_z,
            ],
            [-sin_y, cos_y * sin_z, cos_y * cos_z],
        ]

        prod = [_dot(row, point) for row in rotation]
        return prod

    for edge in graph.edges:
        ind_a, ind_b = edge["node_indices"][0], edge["node_indices"][1]
        contact_pt_a = edge["properties"]["contact_point_1"]
        contact_pt_b = edge["properties"]["contact_point_2"]

        world_a: List[grb.LinExpr] = node_rotate(ind_a, contact_pt_a)
        world_b: List[grb.LinExpr] = node_rotate(ind_b, contact_pt_b)

        edge_energy: grb.QuadExpr = sum(
            [(m - n) * (m - n) for m, n in zip(world_a, world_b)]
        )
        energy += edge_energy

        # world_pt_a = _vars_point_local2world(contact_pt_a, ind_a)
        # world_pt_b = _vars_point_local2world(contact_pt_b, ind_b)
        #
        # distance_sq = (world_pt_a - world_pt_b) @ (world_pt_a - world_pt_b)

    model.optimize()

    # world_pt_a = point_local2world(
    #     brick_a.get_rotation(),
    #     brick_a.get_translation(),
    #     contact_pt_a
    # )

    model.addConstr(energy, GRB.LESS_EQUAL, budget)
