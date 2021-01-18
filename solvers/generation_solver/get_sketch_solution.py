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
    graph_path = input("Enter adj path in connectivity: ")
    path = folder_path + graph_path
    crop_path = folder_path + input("Enter crop path in connectivity: ")

    solver = MinizincSolver(model_file, "gurobi")
    structure_graph = pickle.load(open(path, "rb"))

    crop = pickle.load(open(crop_path, "rb"))
    base_count = crop.base_count
    filename = crop.filename
    cpoints = np.array([len(base.get_current_conn_points()) / 2 for base in structure_graph.bricks[:base_count]])
    base_int = int(math.sqrt(np.sum(cpoints)))

    img_name = filename + ".png"
    img_path = os.path.join(os.path.dirname(__file__), "super_graph" + img_name)
    img = cv2.imread(img_path)

    node_sd = [1 / i for i in crop.result_sd]
    sd_max = np.amax(np.array(node_sd))
    if not sd_max == 0:
        sd_normal = [round(i / sd_max, 3)  if i > 0 else i for i in node_sd]

    node_color = crop.result_color

    area = util.get_area()
    area = [0 for i in range(base_count)] + [area[b.template.id] for b in structure_graph.bricks[base_count:]]
    area_max = np.amax(np.array(area))
    area_normal = [round(i / area_max, 3) for i in area]

    weight = util.get_weight()
    weight = [weight[b.template.id] for b in structure_graph.bricks]

    max_v = - 1e5
    selected_bricks = []
    scale_with_max_v = -1
    debugger = MyDebugger("solve")
    for scale in range(1, 122, 20):
        results = solver.solve(structure_graph=structure_graph,
                               node_sd=sd_normal,
                               node_area=area_normal,
                               node_weight=weight,
                               base_count=base_count,
                               scale=scale)
        selected_bricks_scale = []
        for i in range(len(structure_graph.bricks)):
            if results[i] == 1:
                if i < base_count:
                    selected_bricks_scale.append(structure_graph.bricks[i])
                else:
                    colored_brick = color_brick(structure_graph.bricks[i], np.array(node_color))
                    selected_bricks_scale.append(colored_brick)

        selected_scale_without_base = selected_bricks_scale[base_count:]
        value_of_solution = 0 # TODO
        if value_of_solution > max_v:
            max_v = value_of_solution
            scale_with_max_v = scale
            selected_bricks = selected_bricks_scale.copy()
    print("Minimum difference obtained at scale = ", scale_with_max_v)
    debugger = MyDebugger("solve")
    write_bricks_to_file(
        selected_bricks, file_path=debugger.file_path(f"selected {filename} {crop.platename} n={len(selected_bricks)}.ldr"))

    print("done!")