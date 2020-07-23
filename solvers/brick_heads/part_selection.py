from bricks_modeling.database import ldraw_colors
from typing import List
import solvers.brick_heads.config as conf
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
import os
from typing import Tuple

def get_skin_color(skin_color: int):
    skin_color_map = {
        2: 78,  # white
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
    template_head_file = "template_head"
    template_bottom_file = "template_bottom"
    template_head_bricks = get_bricks_from_files([template_head_file])
    template__bottom_bricks = get_bricks_from_files([template_bottom_file])
    return template_head_bricks, template__bottom_bricks


def gen_hair(gender: int, hair: int, hair_color: str, bang: int, skin_color: int):
    gender = "M" if gender == 1 else "F"
    hair_file = f"hair/{gender}-Hair-{hair}"
    hair_lh_file = f"hair/{gender}-Hair-{hair}_lh" if bang > 0 else "hair/Hair_no_lh"
    hair_files = [hair_file, hair_lh_file] if hair_lh_file is not None else [hair_file]
    hair_skin_files = get_skin_files(hair_files)

    hair_color_id = nearest_color_id(hair_color)
    skin_color_id = get_skin_color(skin_color)

    hair_bricks = get_bricks_from_files(hair_files, hair_color_id)
    hair_skin_bricks = get_bricks_from_files(hair_skin_files, skin_color_id)

    return hair_bricks , hair_skin_bricks


def gen_eyes(eye: int, skin_color: int):
    eye_file = "eyes/eyes_0" if eye == 0 else "eyes/eyes_glasses"
    eye_skin_file = get_skin_files([eye_file])

    skin_color_id = get_skin_color(skin_color)

    eye_bricks = get_bricks_from_files([eye_file])
    eye_skin_bricks = get_bricks_from_files(eye_skin_file, skin_color_id)

    return eye_bricks , eye_skin_bricks


def gen_hands(hands: int, skin_color: int, clothes_bg_color: Tuple):
    hands_file = f"hands/hands_down_{hands}"
    hands_skin_file = get_skin_files([hands_file])

    skin_color_id = get_skin_color(skin_color)
    clothes_bg_color = nearest_color_id(clothes_bg_color)

    hands_bricks = get_bricks_from_files([hands_file], clothes_bg_color)
    hands_skin_bricks = get_bricks_from_files(hands_skin_file, skin_color_id)

    return hands_bricks , hands_skin_bricks


def gen_leges(pants_type, clothes_bg_color: Tuple, pants_color, skin_color):
    legs_file = "legs/legs"

    legs_bricks = get_bricks_from_files([legs_file])
    # if no pants color, use cloth color
    pants_color_id = nearest_color_id(pants_color) if pants_color is not None else nearest_color_id(clothes_bg_color)
    skin_color_id = get_skin_color(skin_color)

    if pants_type == 1: # long pants
        for b in legs_bricks[:8]:
            b.color = pants_color_id
    elif pants_type == 2: #shorts
        for idx, b in enumerate(legs_bricks[:8]):
            if idx < 4:
                b.color = pants_color_id
            else:
                b.color = skin_color_id
    else: #stocking
        for idx, b in enumerate(legs_bricks[:8]):
            if idx >= 4:
                b.color = pants_color_id
            else:
                b.color = skin_color_id

    return legs_bricks


def gen_clothes(clothes, clothes_bg_color: Tuple):
    clothes_file = "clothes/clothes" if clothes != 15 else "clothes/skirt"

    clothes_color_id = nearest_color_id(clothes_bg_color)

    clothes_bricks = get_bricks_from_files([clothes_file], assign_color_id=clothes_color_id)

    return clothes_bricks


def gen_jaw(jaw, skin_color: int):
    if jaw == 3:  # unsupported jaw
        jaw = 0
    jaw_file = f"jaw/jaw_{jaw}"
    jaw_skin_files = get_skin_files([jaw_file])

    skin_color_id = get_skin_color(skin_color)

    jaw_bricks = get_bricks_from_files([jaw_file])
    jaw_skin_bricks = get_bricks_from_files(jaw_skin_files, skin_color_id)

    return jaw_bricks , jaw_skin_bricks


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
