import copy
import math
import os
import pickle
import cv2
import numpy as np

import solvers.generation_solver.sketch_util as util
from solvers.generation_solver.adjacency_graph import AdjacencyGraph
from solvers.generation_solver.get_sketch import ls_from_layout
from solvers.generation_solver.img_interface import show_interface
from solvers.generation_solver.minizinc_sketch import MinizincSolver
from solvers.generation_solver.sample_constant import *

class PrecomputedModel:
    def __init__(self, structure_graph, node_sd, node_area, node_weight, base_count, filtered_bricks,
                 filtered_ldr_code, background_bool):
        self._structure_graph = structure_graph
        self._node_sd = node_sd
        self._node_area = node_area
        self._node_weight = node_weight
        self._base_count = base_count
        self._filtered_bricks = filtered_bricks
        self._filtered_ldr_code = filtered_ldr_code
        self._background_bool = background_bool

    def get_structure_graph(self):
        return self._structure_graph

    def get_node_sd(self):
        return self._node_sd

    def get_node_area(self):
        return self._node_area

    def get_node_weight(self):
        return self._node_weight

    def get_base_count(self):
        return self._base_count

    def get_filtered_bricks(self):
        return self._filtered_bricks

    def get_filtered_ldr_code(self):
        return self._filtered_ldr_code

    def get_background_bool(self):
        return self._background_bool

    def dump_to_pickle(self, filename):
        dir_name = "_".join(filename.split("_")[:-1])
        folder = os.path.exists(os.path.join(os.path.dirname(__file__), f"precompute_models/{dir_name}"))
        if not folder:
            os.makedirs(os.path.join(os.path.dirname(__file__), f"precompute_models/{dir_name}"))
        pickle.dump(self,
                    open(os.path.join(os.path.dirname(__file__), f'precompute_models/{dir_name}/{filename}.pkl'), "wb"))

    def display_debug_info(self):
        print(f"Node Standard Deviation Size: {len(self._node_sd)}")
        print(f"Node Area Size: {len(self._node_area)}")
        print(f"Node Weight Size: {len(self._node_weight)}")
        print(f"Base Count: {self._base_count}")
        print(f"Filtered Bricks Size: {len(self._filtered_bricks)}")
        print(f"Filtered LDR Code Size: {len(self._filtered_ldr_code)}")
        print(f"LDR Set: {self.ldr_stat()}")

    def ldr_stat(self):
        ldr_set = []
        for ldr in self._filtered_ldr_code:
            if ldr not in ldr_set:
                ldr_set.append(ldr)
        return ldr_set


class Precompute:
    def __init__(self):
        graph_name, img_num, layer_names, layer_nums, background_rgb, degree, scale, width_dis, height_dis = \
            show_interface()
        background_bool = 1
        if len(background_rgb) == 0:
            background_bool = 0

        folder_path = os.path.dirname(__file__) + "/connectivity/"
        path = folder_path + graph_name
        plate_name = graph_name.split("base=")[0]

        structure_graph = pickle.load(open(path, "rb"))

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
            # Todo: Here I use new images
            # img_path = os.path.dirname(__file__) + "/inputs/images/" + img_name
            img_path = os.path.dirname(__file__) + "/new_inputs/" + "_".join(img_name.split("_")[:-1]) + "/" + img_name
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
                sd_normal = [round(i / sd_max, 3) if i > 0 else i for i in node_sd]

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
                        filtered_ldr_code.append(ldr_code[i - base_count])
                    map_array[i] = head
                    head += 1
            for overlap_edge in structure_graph.overlap_edges:
                if map_array[overlap_edge[0]] != -1 and map_array[overlap_edge[1]] != - 1:
                    filtered_overlap_edges.append((map_array[overlap_edge[0]], map_array[overlap_edge[1]]))
            for connect_edge in structure_graph.connect_edges:
                if map_array[connect_edge[0]] != -1 and map_array[connect_edge[1]] != - 1:
                    filtered_connect_edges.append(
                        (map_array[connect_edge[0]], map_array[connect_edge[1]], connect_edge[2]))
            # Use temp filtered graph
            filtered_structure_graph = copy.deepcopy(structure_graph)
            filtered_structure_graph.bricks = filtered_bricks
            filtered_structure_graph.connect_edges = filtered_connect_edges
            filtered_structure_graph.overlap_edges = filtered_overlap_edges

            precompute_model = PrecomputedModel(filtered_structure_graph, filtered_node_sd, filtered_node_area,
                                                filtered_node_weight, base_count, filtered_bricks, filtered_ldr_code,
                                                background_bool)

            if not sample_constant.ACTIVATE_EXTEND_SAMPLE:
                filename = f"{img_name} b={base_count} p={plate_name} No Ext"
            else:
                filename = f"{img_name} b={base_count} p={plate_name} g={sample_constant.EXTEND_SAMPLE_GRANULARITY} " \
                           f"t={sample_constant.EXTEND_SAMPLE_THRESHOLD}"
            precompute_model.dump_to_pickle(filename)


if __name__ == "__main__":
    precompute_starter = Precompute()
