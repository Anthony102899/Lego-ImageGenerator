import os
from solvers.generation_solver.minizinc_sketch import MinizincSolver
from util.debugger import MyDebugger
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from solvers.generation_solver.adjacency_graph import AdjacencyGraph
import solvers.generation_solver.sketch_util as util
import numpy as np
import cv2
import math
import pickle5 as pickle

if __name__ == "__main__":
    model_file = "./solvers/generation_solver/solve_sketch.mzn"
    folder_path = "solvers/generation_solver/connectivity/"
    graph_name = input("Enter adj path in connectivity: ")
    path = folder_path + graph_name
    layer = int(input("Enter layer: "))
    layer_names = input("Enter names in each layer, separated by space: ")
    layer_names = layer_names.split(" ")

    solver = MinizincSolver(model_file, "gurobi")
    structure_graph = pickle.load(open(path, "rb"))

    crop_path = folder_path + "crop " + layer_names[0] + " " + graph_name
    crop = pickle.load(open(crop_path, "rb"))
    base_count = crop.base_count
    base_bricks = structure_graph.bricks[:base_count]
    filename = crop.filename
    cpoints = np.array([len(base.get_current_conn_points()) / 2 for base in base_bricks])
    base_int = int(math.sqrt(np.sum(cpoints)))

    area = util.get_area()
    area = [0 for i in range(base_count)] + [area[b.template.id] for b in structure_graph.bricks[base_count:]]
    area_max = np.amax(np.array(area))
    area_normal = [round(i / area_max, 3) for i in area]
    weight = util.get_weight()
    weight = [weight[b.template.id] for b in structure_graph.bricks]

    selected_bricks = base_bricks
    for l in range(0, layer):
        node_sd = [1 / i for i in crop.result_sd]
        sd_max = np.amax(np.array(node_sd))
        if not sd_max == 0:
            sd_normal = [round(i / sd_max, 3)  if i > 0 else i for i in node_sd]
        node_color = crop.result_color

        results = solver.solve(structure_graph=structure_graph,
                                node_sd=sd_normal,
                                node_area=area_normal,
                                node_weight=weight,
                                base_count=base_count,
                                scale=1)
        selected_bricks_layer = []
        for i in range(base_count, len(structure_graph.bricks)):
            if results[i] == 1:
                colored_brick = util.color_brick(structure_graph.bricks[i], np.array(node_color))
                selected_bricks_layer.append(colored_brick)

        if layer > 1:
            crop_path = folder_path + "crop " + layer_names[l + 1] + " " + graph_name
            crop = pickle.load(open(crop_path, "rb"))
            selected_bricks_layer = util.move_layer(selected_bricks_layer, l)

        selected_bricks += selected_bricks_layer

    debugger = MyDebugger("solve")
    write_bricks_to_file(
        selected_bricks, file_path=debugger.file_path(f"{filename} {crop.platename} n={len(selected_bricks_layer)}.ldr"))
        
    print("done!")