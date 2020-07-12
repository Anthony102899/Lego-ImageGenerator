from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.connectivity_graph import ConnectivityGraph
from util.debugger import MyDebugger
import os
import itertools as iter
import json
from bricks_modeling.database import ldraw_colors

parts = ["hair", "clothes", "glasses", "left_arm","right_arm", "beard"]

template_path = "./solvers/brick_heads/template.ldr"
parts_dir = "./solvers/brick_heads/parts/"
input_dir = f"./solvers/brick_heads/input_images/"

skin_color_map = {
    1 : 511, # white
    2 : 78,   # yellow
    3 : 484,   # black or 10484?
}

def select_nearest_color(rgb):
    all_colors = ldraw_colors.read_colors()
    best_id = -1
    closest_dist = 1e8
    for l_rgb, color_id in all_colors.items():
        current_dist = (rgb[0]-l_rgb[0])**2+(rgb[1]-l_rgb[1])**2+(rgb[2]-l_rgb[2])**2
        if current_dist < closest_dist:
            best_id = color_id
            closest_dist = current_dist

    return best_id

def get_LEGO_parts(body_id, json_data):
    if parts[body_id] == "hair":
        gender = "F" if json_data["gender"] == 0 else "M"
        length = json_data["hair"][1]
        if sum(json_data["hair"][2])==0: # no bang
            return [f"{gender}-Hair-{length}"]
        else:
            return [f"{gender}-Hair-{length}", f"{gender}-Hair-{length}_lh"]
    elif parts[body_id] == "right_arm":
        return ["right_arm_0"]
    elif parts[body_id] == "left_arm":
        return ["left_arm_0"]
    elif parts[body_id] == "clothes":
        return ["clothes"]
    elif parts[body_id] == "glasses":
        if json_data["glasses"] == -1:
            return ["eyes_0"]
        else:
            return ["eyes_glasses"]
    elif parts[body_id] == "beard":
        return ["mustache_yes" if json_data["beard"] == 1 else "mustache_no"]
    else:
        print("error id:", body_id)
        input()

def gen_LEGO_figure(input_figure):
    total_bricks = read_bricks_from_file(
        template_path, read_fake_bricks=True
    )

    with open(input_dir + f"{input_figure}.json") as f:
        data = json.load(f)

    skin_color = data["skin"]
    color_id = skin_color_map[skin_color]

    for i in range(len(parts)):
        part_selection = get_LEGO_parts(i, data)
        nearest_color = None
        if parts[i] == "hair" or parts[i] == "clothes":
            part_color = data[parts[i]][0]
            nearest_color = select_nearest_color(part_color)

        for part_file in part_selection:
            absolute_path = parts_dir + part_file
            if os.path.exists(absolute_path + ".ldr"):
                bricks = read_bricks_from_file(absolute_path + ".ldr", read_fake_bricks=True)
                if nearest_color is not None:
                    for b in bricks:
                        b.color = nearest_color
                total_bricks += bricks
            if os.path.exists(absolute_path + "_skin.ldr"):
                bricks = read_bricks_from_file(absolute_path + "_skin.ldr", read_fake_bricks=True)
                for b in bricks:
                    b.color = color_id
                total_bricks += bricks

    write_bricks_to_file(
        total_bricks, file_path=debugger.file_path(f"complete_{input_figure}.ldr"), debug=False
    )


if __name__ == "__main__":
    debugger = MyDebugger("brick_heads")

    # files = ["kaifu", "yuminhong", "taylor", "hepburn", "gxs"]
    files = ["taylor"]

    for input_figure in files:
        gen_LEGO_figure(input_figure)


