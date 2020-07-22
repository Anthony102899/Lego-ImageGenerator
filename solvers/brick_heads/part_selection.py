from bricks_modeling.database import ldraw_colors
from typing import List
import solvers.brick_heads.config as conf
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
import os

def get_skin_color(skin_color: int):
    skin_color_map = {
        2: 511,  # white
        1: 78,  # yellow
        0: 484,  # black or 10484?
    }
    return skin_color_map[skin_color]


def nearest_color_id(rgb, given_list=None):
    if given_list is None:
        all_colors = ldraw_colors.read_colors()
    else:
        all_colors = given_list

    best_id = -1
    closest_dist = 1e8
    for l_rgb, color_id in all_colors.items():
        current_dist = (
            (rgb[0] - l_rgb[0]) ** 2
            + (rgb[1] - l_rgb[1]) ** 2
            + (rgb[2] - l_rgb[2]) ** 2
        )
        if current_dist < closest_dist:
            best_id = color_id
            closest_dist = current_dist

    return best_id

def gen_template():
    template_file = "template"
    template_bricks = get_bricks_from_files([template_file])
    return template_bricks

def gen_hair(gender:int, hair:int, hair_color:str, bang:int, skin_color:int):
    gender = "M" if gender == 0 else "F"
    hair_file = f"hair/{gender}-Hair-{hair}"
    hair_lh_file = f"hair/{gender}-Hair-{hair}_lh" if bang>0 else "hair/Hair_no_lh"
    hair_files = [hair_file, hair_lh_file] if hair_lh_file is not None else [hair_file]
    hair_skin_files = get_skin_files(hair_files)

    hair_color_id = nearest_color_id(hair_color)
    skin_color_id = get_skin_color(skin_color)

    hair_bricks = get_bricks_from_files(hair_files, hair_color_id)
    hair_skin_bricks = get_bricks_from_files(hair_skin_files, skin_color_id)

    return hair_bricks + hair_skin_bricks

def gen_eyes(eye:int, skin_color:int):
    eye_file = "eyes/eyes_0" if eye == 0 else "eyes/eyes_glasses"
    eye_skin_file = get_skin_files([eye_file])

    skin_color_id = get_skin_color(skin_color)

    eye_bricks = get_bricks_from_files([eye_file])
    eye_skin_bricks = get_bricks_from_files(eye_skin_file, skin_color_id)

    return eye_bricks + eye_skin_bricks

def gen_hands(hands:int, skin_color:int):
    hands_file = f"hands/hands_down_{hands}"
    hands_skin_file = get_skin_files([hands_file])

    skin_color_id = get_skin_color(skin_color)

    hands_bricks = get_bricks_from_files([hands_file])
    hands_skin_bricks = get_bricks_from_files(hands_skin_file, skin_color_id)

    return hands_bricks + hands_skin_bricks

def gen_jaw(jaw, skin_color:int):
    if jaw == 3: #unsupported jaw
        jaw = 0
    jaw_file = f"jaw/jaw_{jaw}"
    jaw_skin_files = get_skin_files([jaw_file])

    skin_color_id = get_skin_color(skin_color)

    jaw_bricks = get_bricks_from_files([jaw_file])
    jaw_skin_bricks = get_bricks_from_files(jaw_skin_files, skin_color_id)

    return jaw_bricks + jaw_skin_bricks


def get_part_files(body, json_data):
    if body == "hair":
        gender = "F" if json_data["gender"] == 0 else "M"
        length = json_data["hair"][1]
        nearest_ldr_color = nearest_color_id(rgb=json_data["hair"][0])

        if sum(json_data["hair"][2]) == 0:  # no bang
            return [
                (f"{gender}-Hair-{length}", nearest_ldr_color),
                ("Hair_no_lh", nearest_ldr_color),
            ]
        else:
            return [
                (f"{gender}-Hair-{length}", nearest_ldr_color),
                (f"{gender}-Hair-{length}_lh", nearest_ldr_color),
            ]

    elif body == "hands":
        return [("08_hands_front_non", None)]

    elif body == "clothes":
        nearest_ldr_color = nearest_color_id(rgb=json_data["clothes"][0])
        return [("clothes", nearest_ldr_color)]

    elif body == "glasses":
        if json_data["glasses"] == -1:
            return [("eyes_0", None)]
        else:
            return [("eyes_glasses", None)]

    elif body == "beard":
        return [("mustache_no" if json_data["beard"] == -1 else "mustache_yes", None)]

    else:
        print("error id:", body_id)
        input()


def get_skin_files(selected_files):
    skin_files = []

    for file in selected_files:
        skined_file = file + "_skin"
        file_path = conf.parts_dir + skined_file + ".ldr"
        if os.path.exists(file_path):
            skin_files.append(skined_file)

    return skin_files


# get the bricks of the indicate files
def get_bricks_from_files(files: List[str], assign_color_id=None):
    total_bricks = []

    for file in files:
        bricks = read_bricks_from_file(
            conf.parts_dir + file + ".ldr", read_fake_bricks=True
        )
        total_bricks += bricks

    if assign_color_id is not None:
        for b in total_bricks:
            b.color = assign_color_id

    return total_bricks