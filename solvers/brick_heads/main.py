from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.connectivity_graph import ConnectivityGraph
from util.debugger import MyDebugger
import os
import itertools as iter
import json
import solvers.brick_heads.part_selection as p_select
import copy
import numpy as np
import csv
from typing import Tuple,List
import solvers.brick_heads.bach_render_images as render

def gen_LEGO_figure(
    gender: int,
    hair: int,
    hair_color: Tuple,
    bang: int,
    skin_color: int,
    eye: int,
    jaw: int,
    hands: int,
    clothes_style: int,
):
    # Template
    template_bricks = p_select.gen_template()

    # Hair
    hair_bricks = p_select.gen_hair(gender, hair, hair_color, bang, skin_color)

    # Eye
    eye_bricks = p_select.gen_eyes(eye, skin_color)

    # Jaw
    jaw_bricks = p_select.gen_jaw(jaw, skin_color)

    # Hands
    hands_bricks = p_select.gen_hands(hands, skin_color)

    return template_bricks + hair_bricks + eye_bricks + jaw_bricks + hands_bricks

def ouptut_all_inputs():
    fake_inputs = gen_all_inputs()
    for file in fake_inputs:

        bricks = gen_LEGO_figure(file[0])

        write_bricks_to_file(
            bricks, file_path=debugger.file_path(f"{file[1]}.ldr"), debug=False
        )



if __name__ == "__main__":
    debugger = MyDebugger("brick_heads")
    csv_path = r"/Users/apple/Dropbox/deecamp/data.csv"

    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        csv_matrix = list(csv_reader)

    for model in csv_matrix[2:3]:
        if model[1] == "":
            continue
        bricks = gen_LEGO_figure(
            gender = int(model[1]),
            hair = int(model[2]),
            hair_color = (int(model[3].split(",")[0]), int(model[3].split(",")[1]), int(model[3].split(",")[2])),
            bang = int(model[4]),
            skin_color= int(model[5]),
            eye= int(model[6]),
            jaw= int(model[7]),
            hands= int(model[8]),
            clothes_style= int(model[9]),
        )

        write_bricks_to_file(
            bricks,
            file_path=debugger.file_path(f"complete_{int(model[0])}.ldr"),
            debug=False,
        )
    render.render_ldrs(debugger._debug_dir_name)