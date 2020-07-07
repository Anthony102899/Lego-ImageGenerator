from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.connectivity_graph import ConnectivityGraph
from util.debugger import MyDebugger
import os
import itertools as iter
import json

parts = ["hairs", "right_arm", "left_arm", "face_bottom", "eyes", "cloth"]

part_candidates = {
    parts[0]: ["hair_short_2.ldr", "hair_short_1.ldr", "hair_long_1.ldr", "hair_long_2.ldr"],
    parts[1]: ["right_arm_0.ldr"],
    parts[2] : ["left_arm_0.ldr"],
    parts[3] : ["face_bottom_0.ldr", "face_bottom_1.ldr", "face_bottom_2.ldr", "face_bottom_3.ldr"],
    parts[4] : ["eyes_0.ldr","eyes_glasses_2.ldr" ,"eyes_glasses_3.ldr"],
    parts[5] : ["cloth_up_0.ldr"]
}

def get_LEGO_parts(body_id, json_data):
    if parts[body_id] == "hairs":
        if json_data["hair"] > 2:
            return 2
        else:
            return 0
    elif parts[body_id] == "right_arm":
        return 0
    elif parts[body_id] == "left_arm":
        return 0
    elif parts[body_id] == "face_bottom":
        if json_data["mouth"] > 1:
            return 0
        else:
            return 1
    elif parts[body_id] == "eyes":
        if json_data["glasses"] > 0:
            return 1
        else:
            return 0
    elif parts[body_id] == "cloth":
        return 0
    else:
        print("error id:", body_id)

def gen_LEGO_figure(input_figure):
    total_bricks = read_bricks_from_file(
        "./data/brickheads/template.ldr", read_fake_bricks=True
    )

    with open(f"./solvers/brick_heads/input/{input_figure}.json") as f:
        data = json.load(f)

    for i in range(len(parts)):
        part_selection = get_LEGO_parts(i, data)
        bricks = read_bricks_from_file(part_dir + part_candidates[parts[i]][part_selection], read_fake_bricks=True)
        total_bricks += bricks

    write_bricks_to_file(
        total_bricks, file_path=debugger.file_path(f"complete_{input_figure}.ldr"), debug=False
    )


part_dir = "./data/brickheads/parts/"


if __name__ == "__main__":
    debugger = MyDebugger("brick_heads")

    files = ["kaifu", "yuminhong", "taylor", "hepburn", "gxs"]

    for input_figure in files:
        gen_LEGO_figure(input_figure)


