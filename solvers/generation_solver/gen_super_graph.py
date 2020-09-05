import os
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from solvers.generation_solver.tile_graph import find_brick_placements
from solvers.generation_solver.crop_model import crop_brick
from util.debugger import MyDebugger
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.bricks.brick_factory import get_all_brick_templates
from bricks_modeling.bricks.brickinstance import BrickInstance
import numpy as np
import trimesh
import copy
import json
import os
import time
import pickle5 as pickle
from util.geometry_util import get_random_transformation
from solvers.generation_solver.adjacency_graph import AdjacencyGraph
from solvers.generation_solver.gurobi_solver import GurobiSolver
from solvers.generation_solver.minizinc_solver import MinizincSolver

brick_IDs = [#"3005",
             #"4733",
             "3023",
             "3024",
             "54200",
             "3069",
             "4081b",
             "4073",
             #"3070",
             #"59900",
             #"3004",
             #"11477",
             #"4070", # cuboid
             #"4287", # slope
             #"3070", # plate
             #"3062", # round
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

def crop(debugger, tile_set, tilename):
    obj_path = os.path.join(os.path.dirname(__file__), "super_graph/bunny.obj")
    mesh = trimesh.load(obj_path)
    flip = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    mesh.apply_transform(flip)
    scale = float(input("Enter scale of obj: "))
    start_time = time.time()
    result = crop_brick(mesh, tile_set, scale)
    print(f"resulting LEGO model has {len(result)} bricks")

    _, filename=os.path.split(obj_path)
    filename = (filename.split("."))[0]
    write_bricks_to_file(result, file_path=debugger.file_path(f"{filename} s={int(scale)} n={len(result)} {tilename} t={round(time.time() - start_time, 2)}.ldr"))
    return result

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
    start_time = time.time()
    seed_brick = copy.deepcopy(brick_set[0])
    bricks = find_brick_placements(
        num_rings, base_tile=seed_brick, tile_set=brick_set, initial_time=start_time
        )
    print(f"generate finished in {round(time.time() - start_time, 2)}")
    print(f"number of tiles neighbours in ring{num_rings}:", len(bricks))
    write_bricks_to_file(
        bricks, file_path=debugger.file_path(f"{brick_IDs} n={len(bricks)} r={num_rings} t={round(time.time() - start_time, 2)}.ldr")
        )
    start_time = time.time()
    structure_graph = AdjacencyGraph(bricks)  
    pickle.dump(structure_graph, open(os.path.join(os.path.dirname(__file__), f'connectivity/{brick_IDs} n={len(bricks)} r={num_rings} t={round(time.time() - start_time, 2)}.pkl'), "wb"))
    return bricks, structure_graph

def read_bricks(path, debugger):
    bricks = read_bricks_from_file(path)
    _, filename = os.path.split(path)
    filename = (filename.split("t="))[0]
    start_time = time.time()
    structure_graph = AdjacencyGraph(bricks) 
    pickle.dump(structure_graph, open(os.path.join(os.path.dirname(__file__), f'connectivity/{filename} t={round(time.time() - start_time, 2)}.pkl'), "wb"))
    return bricks, structure_graph

if __name__ == "__main__":
    volume = get_volume()
    debugger = MyDebugger("test")
    model_file = "./solvers/generation_solver/solve_model.mzn"

    mode = int(input("Enter mode: "))
    if mode == 1:
        """ option1: generate a new graph """
        brick_set = get_brick_templates(brick_IDs)
        num_rings = int(input("Enter ring: "))
        bricks, structure_graph = generate_new(brick_set, num_rings=num_rings, debugger=debugger)
        filename = str(brick_IDs) + str(num_rings)
    else:
        if mode == 2: 
            """ option2: load an existing ldr file """
            path = "super_graph/" + input("Enter path in super_graph: ")
            bricks, structure_graph = read_bricks(os.path.join(os.path.dirname(__file__), path), debugger)
        elif mode == 3:
            """ option3: load a pkl file """
            path1 = "solvers/generation_solver/connectivity/"
            path = path1 + input("Enter path in connectivity: ")
            structure_graph = pickle.load(open(path, "rb"))
        _, filename = os.path.split(path)
        filename = (filename.split(" t="))[0]
    
    start_time = time.time()
    solver = MinizincSolver(model_file, "gurobi")
    results, time_used = solver.solve(structure_graph=structure_graph,
                                      node_volume=[volume[b.template.id] for b in structure_graph.bricks],
                                      flag=[int(f) for f in np.ones(len(structure_graph.bricks))])
    
    selected_bricks = []
    for i in range(len(structure_graph.bricks)):
        if results[i] == 1:
            selected_bricks.append(structure_graph.bricks[i])

    write_bricks_to_file(
        selected_bricks, file_path=debugger.file_path(f"selected {filename} n={len(selected_bricks)} t={round(time.time() - start_time, 2)}.ldr")
    )

    print("done!")
