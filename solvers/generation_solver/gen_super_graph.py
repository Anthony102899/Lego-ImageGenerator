from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from solvers.generation_solver.tile_graph import find_brick_placements
from util.debugger import MyDebugger
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.bricks.brick_factory import get_all_brick_templates
from bricks_modeling.bricks.brickinstance import BrickInstance
import numpy as np
import copy
from scipy.spatial.transform import Rotation as R
from bricks_modeling.connectivity_graph import ConnectivityGraph
from util.geometry_util import get_random_transformation
from solvers.generation_solver.adjacency_graph import AdjacencyGraph
from solvers.generation_solver.gurobi_solver import GurobiSolver

brick_IDs = ["3004",
             # "4070", # cuboid
              "4287", # slope
             # "3070", # plate
             # "3062", # round
             ]

def get_brick_templates(brick_IDs):
    brick_templates, template_ids = get_all_brick_templates()
    bricks = []
    for id in brick_IDs:
        assert id in template_ids
        brick_idx = template_ids.index(id)
        brickInstance = BrickInstance(
            brick_templates[brick_idx], np.identity(4, dtype=float)
        )
        bricks.append(brickInstance)

    return bricks

if __name__ == "__main__":
    debugger = MyDebugger("test")
    brick_set = get_brick_templates(brick_IDs)
    seed_brick = copy.deepcopy(brick_set[0])

    seed_brick.trans_matrix = get_random_transformation()
    num_rings = 1
    bricks = find_brick_placements(
        num_rings, base_tile=seed_brick, tile_set=brick_set
    )

    print(f"number of tiles neighbours in ring{num_rings}:", len(bricks))
    write_bricks_to_file(
        bricks, file_path=debugger.file_path(f"test{num_rings}.ldr")
    )

    structure_graph = AdjacencyGraph(bricks)

    solver = GurobiSolver()
    results, time_used = solver.solve(nodes_num=len(structure_graph.bricks),
                                      node_areas=[1 for b in structure_graph.bricks],
                                      edges=structure_graph.overlap_edges)

    selected_bricks = []
    for i in range(len(structure_graph.bricks)):
        if results[i] == 1:
            selected_bricks.append(structure_graph.bricks[i])

    write_bricks_to_file(
        selected_bricks, file_path=debugger.file_path(f"selected_test{num_rings}.ldr")
    )

    print("done!")

