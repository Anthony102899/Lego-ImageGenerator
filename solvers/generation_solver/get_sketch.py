import copy
import json
import logging
import sys
sys.path.append("../../../lego-solver")
import os
from solvers.generation_solver.minizinc_sketch import MinizincSolver
from solvers.generation_solver.polygon_intersection import plot_polygons, collide_connect_2D, group_display
from util.debugger import MyDebugger
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from solvers.generation_solver.adjacency_graph import AdjacencyGraph
import solvers.generation_solver.sketch_util as util
from solvers.generation_solver.img_interface import show_interface
import numpy as np
import cv2
import math
import pickle
import time
from multiprocessing import Pool
from functools import partial
from scipy import stats

# return color or sd or -1
def crop_ls(rgbs, sd):
    if len(rgbs) == 0:
        if sd:
            return -1;
        return []
    length_rgbs = float(len(rgbs))
    mode, frequency = stats.mode(rgbs)
    if sd:
        """if (np.divide(frequency, length_rgbs) < 0.5).any():
            return -0.01"""
        """diff_rgbs = np.subtract(rgbs, mode)
        sd_rgbs = np.sqrt(np.sum(np.power(diff_rgbs, 2), axis=0))
        return float(round(np.prod(sd_rgbs), 4) + 0.0001)"""
        frequency_diff = np.subtract(frequency, length_rgbs)
        frequency_diff_rate = np.divide(frequency_diff, length_rgbs)
        return float(np.sum(np.square(frequency_diff_rate)))
    return mode
    # return np.average(rgbs, axis = 0)

# return *node_sd* and *node_color*
# Add new parameter here
def new_ls_from_layout(img, plate_set, base_int, file_name):
    """rgbs_ls = util.get_cover_rgb(img=img, base_int=base_int, brick=plate_set)
    node_sd = crop_ls(rgbs_ls, sd=True)
    node_color = crop_ls(rgbs_ls, sd=False)"""
    set_size = len(plate_set)
    map = np.array(json.load(open(os.path.dirname(__file__) + f"/json/{file_name.split('.')[0]}.json")))
    with Pool(5) as p:
        rgbs_ls = p.map_async(partial(util.new_get_cover_rgb, img=img, base_int=base_int, map=map), plate_set)
        while not rgbs_ls.ready():
            print(time.strftime("INFO: %Y-%m-%d %H:%M:%S ", time.localtime()),
                  "rgbs process: ", "#"*(50-int(50*float(rgbs_ls._value.count(None))/set_size)),
                  "-"*int(50*float(rgbs_ls._value.count(None))/set_size),
                  " ", 100-int(100*float(rgbs_ls._value.count(None))/set_size), "%")
            time.sleep(2)
        print(time.strftime("INFO: %Y-%m-%d %H:%M:%S ", time.localtime()),
              "rgbs process: ", "#"*50,
              "   100%")
        print("-"*100)
        task_size = len(rgbs_ls.get()) # include only node_sd
        node_sd = p.map_async(partial(crop_ls, sd=True), rgbs_ls.get())
        node_color = p.map_async(partial(crop_ls, sd=False), rgbs_ls.get())
        while not node_sd.ready():
            sd_left = 0
            if not node_sd.ready():
                sd_left = node_sd._value.count(None)
            not_ready_rate = float(sd_left) / task_size
            print(time.strftime("INFO: %Y-%m-%d %H:%M:%S ", time.localtime()),
                  "node sd and color process: ", "#"*(50-int(50*not_ready_rate)),
                  "-"*int(50*not_ready_rate),
                  " ", 100-int(100*not_ready_rate), "%")
            time.sleep(2)
        print(time.strftime("INFO: %Y-%m-%d %H:%M:%S ", time.localtime()),
              "node sd and color process: ", "#" * 50,
              "   100%")
        node_color.wait()
    return node_sd.get(), node_color.get()


def ls_from_layout(img, plate_set, base_int):
    """rgbs_ls = util.get_cover_rgb(img=img, base_int=base_int, brick=plate_set)
    node_sd = crop_ls(rgbs_ls, sd=True)
    node_color = crop_ls(rgbs_ls, sd=False)"""
    set_size = len(plate_set)
    with Pool(5) as p:
        rgbs_ls = p.map_async(partial(util.get_cover_rgb, img=img, base_int=base_int), plate_set)
        while not rgbs_ls.ready():
            print(time.strftime("INFO: %Y-%m-%d %H:%M:%S ", time.localtime()),
                  "rgbs process: ", "#"*(50-int(50*float(rgbs_ls._value.count(None))/set_size)),
                  "-"*int(50*float(rgbs_ls._value.count(None))/set_size),
                  " ", 100-int(100*float(rgbs_ls._value.count(None))/set_size), "%")
            time.sleep(2)
        print(time.strftime("INFO: %Y-%m-%d %H:%M:%S ", time.localtime()),
              "rgbs process: ", "#"*50,
              "   100%")
        print("-"*100)
        task_size = len(rgbs_ls.get()) # include only node_sd
        node_sd = p.map_async(partial(crop_ls, sd=True), rgbs_ls.get())
        node_color = p.map_async(partial(crop_ls, sd=False), rgbs_ls.get())
        while not node_sd.ready():
            sd_left = 0
            if not node_sd.ready():
                sd_left = node_sd._value.count(None)
            not_ready_rate = float(sd_left) / task_size
            print(time.strftime("INFO: %Y-%m-%d %H:%M:%S ", time.localtime()),
                  "node sd and color process: ", "#"*(50-int(50*not_ready_rate)),
                  "-"*int(50*not_ready_rate),
                  " ", 100-int(100*not_ready_rate), "%")
            time.sleep(2)
        print(time.strftime("INFO: %Y-%m-%d %H:%M:%S ", time.localtime()),
              "node sd and color process: ", "#" * 50,
              "   100%")
        node_color.wait()
    return node_sd.get(), node_color.get()


def inspect(structure_graph=None, bricks_only=False, bricks=None, basenum=8, depictbase=False, base=None):
    positive_align = []
    negative_align = []
    reverse_positive_align = []
    reverse_negative_align = []
    if not bricks_only:
        bricks = structure_graph.bricks
    print("Total number is " + str(len(bricks)))
    for i in range(basenum, len(bricks)):
        brick = bricks[i]
        if round(brick.trans_matrix[0][0], 2) == 1 and round(brick.trans_matrix[2][2], 2) == -1:
            positive_align.append(brick)
        if round(brick.trans_matrix[0][0], 2) == -1 and round(brick.trans_matrix[2][2], 2) == 1:
            negative_align.append(brick)
        if round(brick.trans_matrix[0][2], 2) == -1 and round(brick.trans_matrix[2][0], 2) == -1:
            reverse_positive_align.append(brick)
        if round(brick.trans_matrix[0][2], 2) == 1 and round(brick.trans_matrix[2][0], 2) == 1:
            reverse_negative_align.append(brick)

    group_display(positive_align, 'r', depict_base=depictbase, base=base)
    group_display(negative_align, 'k', depict_base=depictbase, base=base)
    group_display(reverse_positive_align, 'r', depict_base=depictbase, base=base)
    group_display(reverse_negative_align, 'k', depict_base=depictbase, base=base)
    print(len(positive_align) + len(negative_align) + len(reverse_positive_align) + len(reverse_negative_align))


def pickle_collision_finding(structure_graph):
    bricks = structure_graph.bricks
    for i in range(len(bricks)):
        brick = bricks[i]
        if brick.template.id == '43722' and brick.trans_matrix[0][3] == 250 \
                and brick.trans_matrix[2][3] == 400 and brick.trans_matrix[0][2] == -1 \
                and brick.trans_matrix[2][0] == -1:
            for j in range(len(bricks)):
                brick = bricks[j]
                if brick.template.id == '43723' and brick.trans_matrix[0][3] == 230 \
                        and brick.trans_matrix[2][3] == 400 and round(brick.trans_matrix[0][2], 2) == 1 \
                        and round(brick.trans_matrix[2][0], 2) == 1:
                    for col_edge in structure_graph.connect_edges:
                        if col_edge[0] == i:
                            print(col_edge)
                    result = collide_connect_2D(bricks[i], bricks[j])
                    print("found")


if __name__ == "__main__":
    graph_name, img_num, layer_names, layer_nums, background_rgb, degree, scale, width_dis, height_dis = show_interface()
    background_bool = 1
    if len(background_rgb) == 0:
        background_bool = 0

    folder_path = os.path.dirname(__file__) + "/connectivity/"
    path = folder_path + graph_name
    plate_name = graph_name.split("base=")[0]

    model_file = os.path.dirname(__file__) + "/solve_sketch.mzn"
    solver = MinizincSolver(model_file, "gurobi")

    structure_graph = pickle.load(open(path, "rb"))
    # inspect(structure_graph)
    # pickle_collision_finding(structure_graph)

    plate_set = structure_graph.bricks
    base_count = util.count_base_number(plate_set)
    base_bricks = plate_set[:base_count]
    sketch_bricks = plate_set[base_count:]
    cpoints = np.array([len(base.get_current_conn_points()) / 2 for base in base_bricks])
    base_int = int(math.sqrt(np.sum(cpoints)))

    area = util.get_area()
    area = [0 for i in range(base_count)] + [area[b.template.id] for b in sketch_bricks]
    area_max = np.amax(np.array(area))
    area_normal = [round(i / area_max, 3) for i in area]
    weight = util.get_weight()
    weight = [weight[b.template.id] for b in plate_set]
    ldr_color = util.read_ldr_color()

    selected_bricks = base_bricks
    for k in range(img_num):
        layer = int(layer_nums[k])
        img_name = layer_names[k]
        print("Layer number ", layer, " Image name: ", img_name)
        img_path = os.path.dirname(__file__) + "/inputs/images/" + img_name
        img_path = os.path.join(os.path.dirname(__file__), img_path)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_name = (img_name.split("."))[0]

        if not (scale == 1 and degree == 0):
            img = util.rotate_image(util.scale_image(img, scale), degree)
        if not (width_dis == 0 and height_dis == 0):
            img = util.translate_image(img, width_dis, height_dis)
        img = cv2.resize(img, (base_int * 20 + 1, base_int * 20 + 1))

        node_sd, node_color = ls_from_layout(img, sketch_bricks, base_int)
        node_sd = [0.1 for i in range(base_count)] + node_sd
        node_sd = [i for i in node_sd]
        sd_max = np.amax(np.array(node_sd))
        if not sd_max == 0:
            sd_normal = [round(i / sd_max, 3)  if i > 0 else i for i in node_sd]

        # node_color = [i for i in node_color if len(i) == 3]
        node_color = [i for i in node_color]
        # node_color = np.average(node_color, axis = 0)
        # ldr_code = util.nearest_color(node_color, ldr_color)
        ldr_code = [util.nearest_color(color, ldr_color) if len(color) != 0 else 15 for color in node_color]

        # Remove out-of-boundary bricks
        map_array = np.full(len(node_sd), -1)  # A map mapping old index to new index
        head = 0
        filtered_bricks = []
        filtered_overlap_edges = []
        filtered_connect_edges = []
        filtered_node_sd = []
        filtered_node_area = []
        filtered_node_weight = []
        filtered_ldr_code = []
        for i in range(len(node_sd)):
            if node_sd[i] < 0 and i not in range(base_count):  # out-of-boundary
                continue
            else:  # keep in-boundary
                filtered_bricks.append(structure_graph.bricks[i])
                filtered_node_sd.append(sd_normal[i])
                filtered_node_area.append(area_normal[i])
                filtered_node_weight.append(weight[i])
                if i in range(base_count):
                    filtered_ldr_code.append(15)
                else:
                    filtered_ldr_code.append(ldr_code[i-base_count])
                map_array[i] = head
                head += 1
        for overlap_edge in structure_graph.overlap_edges:
            if map_array[overlap_edge[0]] != -1 and map_array[overlap_edge[1]] != - 1:
                filtered_overlap_edges.append((map_array[overlap_edge[0]], map_array[overlap_edge[1]]))
        for connect_edge in structure_graph.connect_edges:
            if map_array[connect_edge[0]] != -1 and map_array[connect_edge[1]] != - 1:
                filtered_connect_edges.append((map_array[connect_edge[0]], map_array[connect_edge[1]], connect_edge[2]))
        # Use temp filtered graph
        filtered_structure_graph = copy.deepcopy(structure_graph)
        filtered_structure_graph.bricks = filtered_bricks
        filtered_structure_graph.connect_edges = filtered_connect_edges
        filtered_structure_graph.overlap_edges = filtered_overlap_edges
        # Create another set to store the original bricks
        results = solver.solve(structure_graph=filtered_structure_graph,
                                node_sd=filtered_node_sd,
                                node_area=filtered_node_area,
                                node_weight=filtered_node_weight,
                                base_count=base_count)
        selected_bricks_layer = []
        for i in range(base_count, len(filtered_bricks)):
            if results[i] == 1:
                colored_brick = None
                # colored_brick = util.color_brick(plate_set[i], ldr_code, rgb=False)
                if i < base_count:
                    colored_brick = util.color_brick(filtered_bricks[i], 15, rgb=False)
                else:
                    colored_brick = util.color_brick(filtered_bricks[i], filtered_ldr_code[i-base_count], rgb=False)
                selected_bricks_layer.append(colored_brick)

        if background_bool:
            selected_bricks_layer = util.move_layer(selected_bricks_layer, layer + 1)
        else:
            selected_bricks_layer = util.move_layer(selected_bricks_layer, layer)
        selected_bricks += selected_bricks_layer

    if background_bool:
        background = os.path.dirname(__file__) + "/inputs/" + "back " + graph_name.split(".pkl")[0] + ".ldr"
        background = read_bricks_from_file(background)
        selected_bricks += util.move_brickset(background, background_rgb, 0, 0, ldr_color)

    img_ls = img_name.split("_")
    if len(img_ls) > 1:
        img_name = '_'.join(map(str, img_ls[:-1]))

    debugger = MyDebugger(f"{img_name}")
    if scale == 1 and degree == 0:
        write_bricks_to_file(
            selected_bricks, file_path=debugger.file_path(f"{img_name} b={base_int} {plate_name}.ldr"))
    else:
        write_bricks_to_file(
            selected_bricks, file_path=debugger.file_path(f"{img_name} b={base_int} d={degree} s={scale} {plate_name}.ldr"))

    print("done!")
