from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.connectivity_graph import ConnectivityGraph
from util.debugger import MyDebugger
import os
import itertools as iter
import json
from solvers.brick_heads.part_selection import get_part_files, select_nearest_face_color

parts = ["hair", "clothes", "glasses", "left_arm","right_arm", "beard"]

template_path = "./solvers/brick_heads/template.ldr"
parts_dir = "./solvers/brick_heads/parts/"
input_dir = f"./solvers/brick_heads/input_images/"


def get_skin_files(selected_files):
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

    skin_files = get_skin_files(selected_files)
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


if __name__ == "__main__":
    debugger = MyDebugger("brick_heads")

    # files = ["1_lkf", "2_gxs", "3_ymh", "4_taylor", "5_Hepburn", "6_James"]
    files = ["6_James"]

    for input_figure in files:
        with open(input_dir + f"{input_figure}.json") as f:
            json_data = json.load(f)

        bricks = gen_LEGO_figure(json_data)

        write_bricks_to_file(
            bricks, file_path=debugger.file_path(f"complete_{input_figure}.ldr"), debug=False
        )


