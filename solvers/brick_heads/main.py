from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.connectivity_graph import ConnectivityGraph
from util.debugger import MyDebugger
import os
import itertools as iter
import json

parts = ["hairs", "clothes", "glasses", "left_arm","right_arm", "mustache", "face"]

part_candidates = {
    parts[0]: ["hair_girl_short.ldr", "hair_girl_medium.ldr","hair_girl_long.ldr","hair_boy_short.ldr", "hair_boy_medium.ldr","hair_boy_long.ldr"],
    parts[1]: ["clothes.ldr"],
    parts[2]: ["eyes_glasses.ldr"],
    parts[3]:  ["right_arm_0.ldr"],
    parts[4] : ["left_arm_0.ldr"],
    parts[5] : ["mustache_yes.ldr", "mustache_no.ldr"],
    parts[6] : ["face.ldr"] # TODO: add the file of face
}

def get_LEGO_parts(body_id, json_data):
    if parts[body_id] == "hairs":
        gender = json_data["gender"]
        length = json_data["hair"][1]
        return 3*gender + length-1
    elif parts[body_id] == "right_arm":
        return 0
    elif parts[body_id] == "left_arm":
        return 0
    elif parts[body_id] == "clothes":
        return 0
    elif parts[body_id] == "glasses":
        return 0
    elif parts[body_id] == "mustache":
        return 0
    elif parts[body_id] == "face":
        return 0
    else:
        print("error id:", body_id)

def gen_LEGO_figure(input_figure):
    total_bricks = read_bricks_from_file(
        "./solvers/brick_heads/brickheads/template.ldr", read_fake_bricks=True
    )

    with open(f"./solvers/brick_heads/input_images/{input_figure}.json") as f:
        data = json.load(f)

    for i in range(len(parts)):
        part_selection = get_LEGO_parts(i, data)
        bricks = read_bricks_from_file(part_dir + part_candidates[parts[i]][part_selection], read_fake_bricks=True)
        total_bricks += bricks

    write_bricks_to_file(
        total_bricks, file_path=debugger.file_path(f"complete_{input_figure}.ldr"), debug=False
    )


part_dir = "./solvers/brick_heads/brickheads/parts/"


if __name__ == "__main__":
    debugger = MyDebugger("brick_heads")

    # files = ["kaifu", "yuminhong", "taylor", "hepburn", "gxs"]
    files = ["taylor"]

    for input_figure in files:
        gen_LEGO_figure(input_figure)


