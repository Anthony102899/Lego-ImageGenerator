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
from multiprocessing import Pool
from functools import partial

# return color or sd or -1
def crop_ls(rgbs, sd):
    if len(rgbs) == 0:
        if sd:
            return -1
        return []
    if sd:
        return float(round(np.sum(np.std(rgbs, axis=0)), 4) + 0.0001)
    return np.average(rgbs, axis = 0)

# return *node_sd* and *node_color*
def ls_from_layout(img, plate_set, base_int):
    with Pool(20) as p:
        rgbs_ls = p.map(partial(util.get_cover_rgb, img=img, base_int=base_int), plate_set)
        node_sd = p.map(partial(crop_ls, sd=True), rgbs_ls)
        node_color = p.map(partial(crop_ls, sd=False), rgbs_ls)
    return node_sd, node_color

if __name__ == "__main__":
    layer = int(input("Enter layer: "))
    layer_names = input("Enter names in each layer, separated by space: ")
    layer_names = layer_names.split(" ")
    degree = int(input("Enter rotation angle: "))
    scale = int(input("Enter scalling factor: "))

    model_file = "./solvers/generation_solver/solve_sketch.mzn"
    solver = MinizincSolver(model_file, "gurobi")

    folder_path = "solvers/generation_solver/connectivity/"
    graph_name = input("Enter adj path in connectivity: ")
    path = folder_path + graph_name
    plate_name = graph_name.split("base=")[0]

    structure_graph = pickle.load(open(path, "rb"))
    plate_set = structure_graph.bricks
    base_count = util.count_base(plate_set)
    base_bricks = plate_set[:base_count]
    sketch_bricks = plate_set[base_count:]
    print("#bricks in plate: ", len(plate_set))
    cpoints = np.array([len(base.get_current_conn_points()) / 2 for base in base_bricks])
    base_int = int(math.sqrt(np.sum(cpoints)))

    area = util.get_area()
    area = [0 for i in range(base_count)] + [area[b.template.id] for b in sketch_bricks]
    area_max = np.amax(np.array(area))
    area_normal = [round(i / area_max, 3) for i in area]
    weight = util.get_weight()
    weight = [weight[b.template.id] for b in plate_set]

    selected_bricks = base_bricks
    for l in range(layer):
        img_path = "super_graph/images/" + layer_names[l]
        img_path = os.path.join(os.path.dirname(__file__), img_path)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        _, img_name = os.path.split(img_path)
        img_name = (img_name.split("."))[0]

        if not scale == 1 and degree == 0:
            img = util.center_crop(img, scale)
            img = util.rotate_image(img, degree)
            cv2.imwrite(os.path.join(os.path.dirname(__file__), f"super_graph/images/{img_name}_{degree}_{scale}.png"), img)
        
        # resize image to fit the brick
        img = cv2.resize(img, (base_int * 20 + 1, base_int * 20 + 1))

        node_sd, node_color = ls_from_layout(img, sketch_bricks, base_int)
        node_sd = [0.0001 for i in range(base_count)] + node_sd
        node_sd = [1 / i for i in node_sd]
        sd_max = np.amax(np.array(node_sd))
        if not sd_max == 0:
            sd_normal = [round(i / sd_max, 3)  if i > 0 else i for i in node_sd]
        
        node_color = [i for i in node_color if len(i) == 3]
        node_color = np.average(node_color, axis = 0)

        results = solver.solve(structure_graph=structure_graph,
                                node_sd=sd_normal,
                                node_area=area_normal,
                                node_weight=weight,
                                base_count=base_count,
                                scale=1)
        selected_bricks_layer = []
        for i in range(base_count, len(plate_set)):
            if results[i] == 1:
                colored_brick = util.color_brick(plate_set[i], np.array(node_color))
                selected_bricks_layer.append(colored_brick)

        if not l == 0:
            selected_bricks_layer = util.move_layer(selected_bricks_layer, l + 1)
        selected_bricks += selected_bricks_layer
    
    debugger = MyDebugger("solve")
    write_bricks_to_file(
        selected_bricks, file_path=debugger.file_path(f"{img_name} {plate_name} n={len(selected_bricks)}.ldr"))
        
    print("done!")
