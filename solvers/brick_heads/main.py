from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.connectivity_graph import ConnectivityGraph
from util.debugger import MyDebugger
import os
import itertools as iter
import json
from solvers.brick_heads.part_selection import get_part_files, select_nearest_face_color
import copy

parts = ["hair", "clothes", "glasses", "left_arm","right_arm", "beard"]

template_path = "./solvers/brick_heads/template.ldr"
parts_dir = "./solvers/brick_heads/parts/"
input_dir = f"./solvers/brick_heads/input_images/"


def get_skin_files(selected_files, json_data):
    skin_files = []

    skin_color = json_data["skin"]
    color_id = select_nearest_face_color(skin_color)

    for file in selected_files:
        skined_file = file[0] + "_skin"
        file_path = parts_dir + skined_file + ".ldr"
        if os.path.exists(file_path):
            skin_files.append((skined_file, color_id))

    return skin_files

def gen_LEGO_figure(json_data):
    selected_files = []

    for i in range(len(parts)):
        part_selection = get_part_files(parts[i], json_data)
        selected_files += part_selection

    skin_files = get_skin_files(selected_files, json_data)
    selected_files += skin_files

    # start reading bricks
    total_bricks = []
    template_bricks = read_bricks_from_file(
        template_path, read_fake_bricks=True
    )
    total_bricks += template_bricks

    for file in selected_files:
        bricks = read_bricks_from_file(parts_dir + file[0] + ".ldr", read_fake_bricks=True)
        ldraw_color = file[1]
        if ldraw_color is not None:
            for b in bricks:
                b.color = ldraw_color
        total_bricks += bricks

    return total_bricks

def gen_all_inputs():
    files = []

    with open(input_dir + "5_Hepburn.json") as f:
        json_data = json.load(f)

    for gender in [0, 1]:
        for hair in [1,2,3]:
            for glasses in [-1, 1]:
                for beard in [-1,1]:
                    for bang in [0,1]:
                        new_json = copy.deepcopy(json_data)
                        new_json["gender"] = gender
                        new_json["hair"][1] = hair
                        new_json["glasses"] = glasses
                        new_json["beard"] = beard
                        new_json["hair"][2][1] = bang
                        files.append((new_json, f"{gender}_{hair}_{glasses}_{beard}_{bang}"))
    return files

def ouptut_all_inputs():
    fake_inputs = gen_all_inputs()
    for file in fake_inputs:

        bricks = gen_LEGO_figure(file[0])

        write_bricks_to_file(
            bricks, file_path=debugger.file_path(f"{file[1]}.ldr"), debug=False
        )

if __name__ == "__main__":
    debugger = MyDebugger("brick_heads")
    # ouptut_all_inputs()

    # files = ["1_lkf", "2_gxs", "3_ymh", "4_taylor", "5_Hepburn", "6_James"]
    files = ["6_James"]

    for input_figure in files:
        with open(input_dir + f"{input_figure}.json") as f:
            json_data = json.load(f)

        bricks = gen_LEGO_figure(json_data)

        write_bricks_to_file(
            bricks, file_path=debugger.file_path(f"complete_{input_figure}.ldr"), debug=False
        )


