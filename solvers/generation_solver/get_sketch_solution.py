from solvers.generation_solver.minizinc_sketch import MinizincSolver
from util.debugger import MyDebugger
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from solvers.generation_solver.crop_sketch import Crop
from solvers.generation_solver.adjacency_graph import AdjacencyGraph
import numpy as np
import pickle5 as pickle

if __name__ == "__main__":
    model_file = "./solvers/generation_solver/solve_sketch.mzn"
    folder_path = "solvers/generation_solver/connectivity/"
    graph_path = "Google-Photos b=2 n=1689 ['51739', '2420', '2450', '30357', '26601'] .pkl"#input("Enter path in connectivity: ")
    path = folder_path + graph_path
    crop_path = folder_path + "crop_" + graph_path

    solver = MinizincSolver(model_file, "gurobi")
    structure_graph = pickle.load(open(path, "rb"))
    crop = pickle.load(open(crop_path, "rb"))
    base_count = crop.base_count
    result_crop = crop.result_crop
    filename = crop.filename
    platename = crop.platename
    node_sd = [0.0001 for i in range(base_count)] + [round(np.sum(np.std(i[1], axis = 0))) + 0.0001 for i in result_crop]
    results, time_used = solver.solve(structure_graph=structure_graph, node_volume=node_sd, base_count=base_count)
    
    selected_bricks = []
    for i in range(len(structure_graph.bricks)):
        if results[i] == 1:
            selected_bricks.append(structure_graph.bricks[i])

    debugger = MyDebugger("solve")
    write_bricks_to_file(
        selected_bricks, file_path=debugger.file_path(f"selected {filename} {platename} n={len(selected_bricks)}.ldr"))

    print("done!")