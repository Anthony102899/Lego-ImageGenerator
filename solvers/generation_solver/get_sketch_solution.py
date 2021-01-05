import os
from solvers.generation_solver.minizinc_sketch import MinizincSolver
from util.debugger import MyDebugger
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from solvers.generation_solver.crop_sketch import Crop, proj_bbox
from solvers.generation_solver.adjacency_graph import AdjacencyGraph
import numpy as np
import json
import cv2
import math
from shapely.geometry import Point
import pickle5 as pickle

def hex_to_rgb(hexx):
    value = hexx.lstrip('0x2')
    lv = len(value)
    if lv == 0:
        return np.array([0,0,0])
    rgb = [int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)]
    return np.array(rgb)

# return difference between input and brickset (solution) (without base)
def calculate_v(brick_set,img, base_int, polygon_ls):
    dif_sum = [0, 0, 0]
    for i in range(len(brick_set)):
        brick = brick_set[i]
        polygon = polygon_ls[i]
        brick_color = hex_to_rgb(brick.color)
        mini, minj, maxi, maxj = polygon.bounds
        dif = [0, 0, 0]
        for x in range(math.floor(mini), math.ceil(maxi) + 1):
            for y in range(math.floor(minj), math.ceil(maxj) + 1):
                point = Point(x, y)
                if polygon.contains(point):
                    try:
                        img_color = (img[y, x])[::-1]
                        dif = [dif[i] + abs(brick_color[i] - img_color[i]) * 1e-6 for i in range(3)]
                    except:
                        continue
        dif_sum = [round(dif_sum[i] + dif[i], 3) for i in range(3)]
    return - np.sum(dif_sum)

# *brick* is the lower one
def calculate_overlap_v(brick, brick2, img, base_int):
    polygon1 = proj_bbox(brick)
    polygon2 = proj_bbox(brick2)
    dif_polygon = polygon1.difference(polygon2)
    return calculate_v([brick], img, base_int, [dif_polygon])

# return an integer in [0,1]
def cal_border(brickset, base_int):
    standard = base_int * 4 - 4
    maxx = base_int * 20 - 10
    count = 0
    for brick in brickset:
        cpoints = brick.get_current_conn_points()
        cpoints_pos = [[cp.pos[0], cp.pos[2]] for cp in cpoints]
        for z in range(10, base_int * 20 -9, 10):
            if [10, z] in cpoints_pos or [maxx, z] in cpoints_pos:
                count += 1
            if z < 20 or z > maxx - 10:
                continue
            if [z, 10] in cpoints_pos or [z, maxx] in cpoints_pos:
                count += 1
    return count / standard

def get_area(
    brick_database=[
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
    area = {}
    for brick in data:
        if "area" in brick.keys():
            area.update({brick["id"]: brick["area"]})
    return area

if __name__ == "__main__":
    model_file = "./solvers/generation_solver/solve_sketch.mzn"
    folder_path = "solvers/generation_solver/connectivity/"
    graph_path = input("Enter path in connectivity: ")
    path = folder_path + graph_path
    crop_path = folder_path + "crop_" + graph_path

    solver = MinizincSolver(model_file, "gurobi")
    structure_graph = pickle.load(open(path, "rb"))

    crop = pickle.load(open(crop_path, "rb"))
    base_count = crop.base_count
    node_sd = crop.result_crop
    filename = crop.filename
    cpoints = np.array([len(base.get_current_conn_points()) / 2 for base in structure_graph.bricks[:base_count]])
    base_int = int(math.sqrt(np.sum(cpoints)))

    img_name = filename + ".png"
    img_path = os.path.join(os.path.dirname(__file__), "super_graph" + img_name)
    img = cv2.imread(img_path)

    area = get_area()
    area = [0 for i in range(base_count)] + [area[b.template.id] for b in structure_graph.bricks[base_count:]]
    area_max = np.amax(np.array(area))
    area_normal = [round(i / area_max, 3) for i in area]

    max_v = - 1e5
    selected_bricks = []
    scale_with_max_v = -1
    for scale in range(20, 61, 10):
        results = solver.solve(structure_graph=structure_graph, node_sd=node_sd, node_area=area_normal, base_count=base_count, scale=scale)
        selected_bricks_scale = []
        for i in range(len(structure_graph.bricks)):
            if results[i] == 1:
                selected_bricks_scale.append(structure_graph.bricks[i])

        selected_scale_without_base = selected_bricks_scale[base_count:]
        value_of_solution = calculate_v(selected_scale_without_base,img, base_int, [proj_bbox(brick) for brick in selected_scale_without_base]) + \
                            0.5 * cal_border(selected_scale_without_base, base_int)
        if value_of_solution > max_v:
            max_v = value_of_solution
            scale_with_max_v = scale
            selected_bricks = selected_bricks_scale.copy()

    print("Minimum difference obtained at scale = ", scale_with_max_v)
    debugger = MyDebugger("solve")
    write_bricks_to_file(
        selected_bricks, file_path=debugger.file_path(f"selected {filename} {crop.platename} n={len(selected_bricks)}.ldr"))

    print("done!")