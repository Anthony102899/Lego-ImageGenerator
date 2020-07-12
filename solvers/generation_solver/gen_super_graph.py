from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from solvers.generation_solver.tile_graph import find_brick_placements
from util.debugger import MyDebugger
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.bricks.brick_factory import get_all_brick_templates
from bricks_modeling.bricks.brickinstance import BrickInstance
import numpy as np
import copy
import json
import os
import pickle5 as pickle
from util.geometry_util import get_random_transformation
from solvers.generation_solver.adjacency_graph import AdjacencyGraph
from solvers.generation_solver.gurobi_solver import GurobiSolver

brick_IDs = ["3004",
              "4070", # cuboid
             # "4287", # slope
             # "3070", # plate
             # "3062", # round
             ]
def get_volume(
    brick_database=[
        "regular_cuboid.json",
        "regular_plate.json",
        "regular_slope.json",
        "regular_other.json",
        "regular_circular.json"]):
    data = []
    for data_base in brick_database:
        database_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "bricks_modeling", "database", data_base)
        with open(database_file) as f:
            temp = json.load(f)
            data.extend(temp)
    volume = {}
    for brick in data:
        if len(brick) > 2:
            volume.update({brick["id"]: brick["volume"]})
    return volume

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

def generate_new(brick_set, num_rings, debugger):
    seed_brick = copy.deepcopy(brick_set[0])
    bricks = find_brick_placements(
        num_rings, base_tile=seed_brick, tile_set=brick_set
    )
    print(f"number of tiles neighbours in ring{num_rings}:", len(bricks))
    write_bricks_to_file(
        bricks, file_path=debugger.file_path(f"test{brick_IDs} {num_rings}.ldr")
    )
    structure_graph = AdjacencyGraph(bricks)  
    pickle.dump(structure_graph, open(os.path.join(os.path.dirname(__file__), f'connectivity/{brick_IDs} {num_rings}.pkl'), "wb"))
    return bricks, structure_graph

def read_bricks(path, debugger):
    bricks = read_bricks_from_file(path)
    _, filename=os.path.split(path)
    structure_graph = AdjacencyGraph(bricks) 
    pickle.dump(structure_graph, open(os.path.join(os.path.dirname(__file__), f'connectivity/{filename}.pkl'), "wb"))
    return bricks, structure_graph

if __name__ == "__main__":
    debugger = MyDebugger("test")
    volume = get_volume()

    """ option1: generate a new graph """    
    #brick_set = get_brick_templates(brick_IDs)
    #bricks, structure_graph = generate_new(brick_set, num_rings=3, debugger=debugger)

    """ option2: load an existing ldr file """
    bricks, structure_graph = read_bricks(os.path.join(os.path.dirname(__file__), "super_graph/['3004', '4287'] 4.ldr"), debugger)

    """ option3: load a pkl file """
    #structure_graph = pickle.load(open("./connectivity/['3004', '3062'] 3.pkl", "rb"))

    solver = GurobiSolver()
    results, time_used = solver.solve(nodes_num=len(structure_graph.bricks),
                                      node_volume=[volume[b.template.id] for b in structure_graph.bricks],
                                      edges=structure_graph.overlap_edges)

    selected_bricks = []
    for i in range(len(structure_graph.bricks)):
        if results[i] == 1:
            selected_bricks.append(structure_graph.bricks[i])

    write_bricks_to_file(
        selected_bricks, file_path=debugger.file_path(f"selected_test.ldr")
    )

    print("done!")

