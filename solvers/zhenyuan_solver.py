import json
import numpy as np
from typing import List

from bricks_modeling.connectivity_graph import ConnectivityGraph
from util.debugger import MyDebugger
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.bricks.brickinstance import BrickInstance

if __name__ == "__main__":
    bricks: List[BrickInstance] = read_bricks_from_file("./data/LEGO_models/full_models/cube7.ldr")
    graph = ConnectivityGraph(bricks)

    print(graph.to_json())

    # for brick in bricks:
    #     print(brick.get_current_conn_points()[0].pos)
    #
    graph_json = json.loads(graph.to_json())

    for edge in graph.edges:
        ind_a, ind_b = edge["node_indices"][0], edge["node_indices"][1]
        brick_a, brick_b = bricks[ind_a], bricks[ind_b]
        contact_pt_a = edge["properties"]["contact_point_1"]
        contact_pt_b = edge["properties"]["contact_point_2"]

        world_a = brick_a.point_to_world(contact_pt_a)
        world_b = brick_b.point_to_world(contact_pt_b)
        print("a", world_a, "b", world_b)

    #     world_contact_pt_a = transform_point(
    #         np.array(n_a["orientation"]), n_a["translation"], contact_pt_a)
    #     world_contact_pt_b = transform_point(
    #         np.array(n_b["orientation"]), n_b["translation"], contact_pt_b)
    #     print(world_contact_pt_a, world_contact_pt_b)
