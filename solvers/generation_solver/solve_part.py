import os
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from solvers.generation_solver.tile_graph import find_brick_placements
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
from solvers.generation_solver.minizic_part import MinizincSolverp

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
    model_file = "./solvers/generation_solver/solve_model.mzn"
    debugger = MyDebugger("part")

    path1 = "solvers/generation_solver/connectivity/"
    path = path1 + "tmp/4081- nShort_bus s=1.7 n=214 m2 n=2048 t=87.59 t=142.46.pkl"
    #path = path1 + input("Enter path in connectivity: ")
    structure_graph = pickle.load(open(path, "rb"))
    overlap_edges = np.array(structure_graph.overlap_edges)
    _, filename = os.path.split(path)
    filename = (filename.split(" t="))[0]

    brick_num = len(structure_graph.bricks)
    start = 0
    step = (int)(brick_num / 8)
    end = step
    if brick_num < 50:
        end = brick_num

    all_select = []
    while end <= brick_num:
        start_time = time.time()
        solver = MinizincSolverp(model_file, "gurobi")
        results, time_used = solver.solve(structure_graph=structure_graph,
                                          node_volume=[volume[b.template.id] for b in structure_graph.bricks],
                                          flag=[int(f) for f in np.ones(len(structure_graph.bricks))],
                                          start=start, end=end)
        selected_bricks = []
        for i in range(len(results)):
            if results[i] == 1:
                selected_bricks.append(structure_graph.bricks[i + start])
                all_select.append(structure_graph.bricks[i + start])

        write_bricks_to_file(
            selected_bricks, file_path=debugger.file_path(f"{start}-{end} {filename} n={len(selected_bricks)}.ldr"))

        start = start + (int)(2 * step / 3)
        end = end + (int)(2 * step / 3)
        if start < brick_num and end > brick_num:
            end = brick_num
    write_bricks_to_file(
            all_select, file_path=debugger.file_path(f"all {filename} n={len(selected_bricks)}.ldr"))

    print("done!")