from bricks_modeling.database import ldraw_colors
from typing import List
import solvers.brick_heads.config as conf
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
import os
from typing import Tuple
from solvers.brick_heads import texture_to_brick
from solvers.brick_heads.lego_util import get_random_string, add_texture_to_brick
from shutil import copyfile


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
    clothes_bg_color: List,
    logo_imgs: List[str],
    logo_pos: List[Tuple],
    pants_type: int,
    pants_color: List,
):
    print(f"gender:{gender}")
    print(f"hair:{hair}")
    print(f"hair_color:{hair_color}")
    print(f"bang:{bang}")
    print(f"skin_color:{skin_color}")
    print(f"eye:{eye}")
    print(f"jaw:{jaw}")
    print(f"hands:{hands}")
    print(f"clothes_style:{clothes_style}")
    print(f"clothes_bg_color:{clothes_bg_color}")
    print(f"logo_imgs:{logo_imgs}")
    print(f"logo_pos:{logo_pos}")
    print(f"pants_type:{pants_type}")
    print(f"pants_color:{pants_color}")

    ### exception handling
    if len(clothes_bg_color) == 0:
        clothes_bg_color.append((0,0,0))

    if len(pants_color) == 0:
        pants_color.append((0,0,0))

    # Template
    template_head_bricks, template_bottom_bricks = gen_template(skin_color)

    # Hair
    hair_bricks, hair_skin_bricks = gen_hair(gender, hair, hair_color, bang, skin_color)

    # Eye
    eye_bricks, eye_skin_bricks = gen_eyes(eye, skin_color)

    # Jaw
    jaw_bricks, jaw_skin_bricks = gen_jaw(jaw, skin_color)

    # Hands
    hands_bricks, hands_skin_bricks = gen_hands(hands, skin_color, clothes_bg_color)

    # Clothes
    clothes_bricks, texture_brick_strs = gen_clothes(
        clothes_style, clothes_bg_color, logo_imgs, logo_pos
    )

    # Legs
    legs_bricks = gen_leges(
        pants_type, clothes_bg_color, pants_color, skin_color, clothes_style
    )

    return (
        (
            clothes_bricks
            + hands_bricks
            + hands_skin_bricks
            + template_head_bricks
            + eye_skin_bricks
            + eye_bricks
            + jaw_skin_bricks
            + jaw_bricks
            + hair_skin_bricks
            + hair_bricks
            + legs_bricks
            + template_bottom_bricks
        ),
        texture_brick_strs,
        [
            clothes_bricks,
            hands_bricks,
            hands_skin_bricks,
            template_head_bricks,
            eye_skin_bricks,
            eye_bricks,
            jaw_skin_bricks,
            jaw_bricks,
            hair_skin_bricks,
            hair_bricks,
            legs_bricks,
            template_bottom_bricks,
        ],
    )


def get_skin_color(skin_color: int):
    skin_color_map = {
        2: 19,  # white people
        1: 19,  # yellow people
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


def gen_template(skin_color):
    template_head_file = "template_head"
    template_bottom_file = "template_bottom"
    skin_color_id = get_skin_color(skin_color)
    template_head_bricks = get_bricks_from_files([template_head_file], skin_color_id)
    template__bottom_bricks = get_bricks_from_files([template_bottom_file])
    return template_head_bricks, template__bottom_bricks


def gen_hair(gender: int, hair: int, hair_color: str, bang: int, skin_color: int):
    if hair == 6 and gender == 0:
        hair = 2

    if hair == 5 and gender == 0:
        hair = 1

    gender = "M" if gender == 1 else "F"
    hair_file = f"hair/{gender}-Hair-{hair}"
    # special case
    if gender == "M" and hair == 2 and bang > 0:
        hair_file = f"hair/{gender}-Hair-{hair}_2"
    hair_lh_file = f"{hair_file}_lh" if bang > 0 else "hair/Hair_no_lh"
    hair_files = [hair_file, hair_lh_file] if hair_lh_file is not None else [hair_file]
    hair_skin_files = get_skin_files(hair_files)

    hair_color_id = nearest_color_id(hair_color)
    skin_color_id = get_skin_color(skin_color)

    hair_bricks = get_bricks_from_files(hair_files, hair_color_id)
    hair_skin_bricks = get_bricks_from_files(hair_skin_files, skin_color_id)

    return hair_bricks, hair_skin_bricks


def gen_eyes(eye: int, skin_color: int):
    eye_file = "eyes/eyes_0" if eye == 0 else "eyes/eyes_glasses"
    eye_skin_file = get_skin_files([eye_file])

    skin_color_id = get_skin_color(skin_color)

    eye_bricks = get_bricks_from_files([eye_file])
    eye_skin_bricks = get_bricks_from_files(eye_skin_file, skin_color_id)

    return eye_bricks, eye_skin_bricks


def gen_hands(hands: int, skin_color: int, clothes_bg_color: List[Tuple]):
    if hands == 2:
        hands = 2  # because of unsupported brick
    hands_file = f"hands/hands_down_{hands}"
    hands_skin_file = get_skin_files([hands_file])

    skin_color_id = get_skin_color(skin_color)
    clothes_bg_color = (
        nearest_color_id(clothes_bg_color[0]) if len(clothes_bg_color) > 0 else None
    )

    hands_bricks = get_bricks_from_files([hands_file], clothes_bg_color)
    hands_skin_bricks = get_bricks_from_files(hands_skin_file, skin_color_id)

    return hands_bricks, hands_skin_bricks


def gen_leges(
    pants_type, clothes_bg_color: Tuple, pants_color, skin_color, clothes_style
):
    legs_file = "legs/legs"

    legs_bricks = get_bricks_from_files([legs_file])

    # if no pants color, use cloth color
    pants_color_id = (
        nearest_color_id(pants_color[0])
        if len(pants_color) > 0
        else nearest_color_id(clothes_bg_color[0])
    )
    skin_color_id = get_skin_color(skin_color)

    if pants_type == 1:  # long pants
        for b in legs_bricks[:8]:
            b.color = pants_color_id

    if pants_type == 2 or pants_type == 3:  # shorts
        for idx, b in enumerate(legs_bricks[:8]):
            if idx < 4:
                b.color = pants_color_id
            else:
                b.color = skin_color_id

    skirt_bricks = []
    if pants_type == 4 or clothes_style in {15}:
        skirt_bricks = get_bricks_from_files(["legs/skirt"])
        for b in skirt_bricks:
            b.color = pants_color_id
        for b in legs_bricks[:8]:
            b.color = skin_color_id

    return legs_bricks + skirt_bricks


"""
1:西装领带 2:西装无领带 3.领带衬衫 4:夹克衫 5:开怀外套 6:半开怀外套 7:衬衫 8:纯色上衣 9:格子上衣(保留) 10:横条纹上衣(保留) 11:竖条纹上衣(保留) 12:双色上衣(保留) 13:篮球服 14:足球服 15:裙子(待细分)
"""


def gen_clothes(
    clothes, clothes_bg_color: Tuple, logo_imgs: List[str], logo_pos: List[Tuple]
):
    clothes_color_id = (
        nearest_color_id(clothes_bg_color[0]) if len(clothes_bg_color) > 0 else None
    )
    clothes_bricks = []
    texture_brick_strs = []
    front_bricks = []

    if clothes in {1, 2, 3, 4, 5, 6, 7}:
        clothes_files = ["clothes/clothes"]
        front_bricks = get_bricks_from_files(
            ["clothes/clothes_front_1"], assign_color_id=clothes_color_id
        )
        if clothes in {5, 6}:
            front_bricks[0].color = nearest_color_id((0, 0, 0))
        else:
            customize_brick = front_bricks[0]
            texture_content, customized_id = add_texture_to_brick(
                customize_brick.template.id, clothes
            )
            customize_brick.template.id = customized_id
            texture_brick_strs.append((customized_id, texture_content))

    elif clothes in {13, 14}:
        clothes_files = ["clothes/clothes"]
        front_bricks = get_bricks_from_files(
            ["clothes/clothes_front_2"], assign_color_id=clothes_color_id
        )

        customize_brick_1 = front_bricks[0]
        texture_content_1, customized_id1 = add_texture_to_brick(
            customize_brick_1.template.id, f"{clothes}_0"
        )
        customize_brick_1.template.id = customized_id1
        texture_brick_strs.append((customized_id1, texture_content_1))

        customize_brick_2 = front_bricks[1]
        texture_content_2, customized_id2 = add_texture_to_brick(
            customize_brick_2.template.id, f"{clothes}_1"
        )
        customize_brick_2.template.id = customized_id2
        texture_brick_strs.append((customized_id2, texture_content_2))

    elif clothes in {15}:
        clothes_files = ["clothes/skirt"]
    else:
        clothes_files = ["clothes/clothes"]
        front_bricks = get_bricks_from_files(
            ["clothes/clothes_front_1"], assign_color_id=clothes_color_id
        )
        for image in logo_imgs:
            if os.path.exists(os.path.join(conf.parts_dir, "textures", image)):
                customize_brick = front_bricks[0]
                texture_content, customized_id = add_texture_to_brick(
                    customize_brick.template.id, image
                )
                customize_brick.template.id = customized_id
                texture_brick_strs.append((customized_id, texture_content))

    clothes_bricks = get_bricks_from_files(
        clothes_files, assign_color_id=clothes_color_id
    )

    inline_textures = []
    for texture in texture_brick_strs:
        if conf.output_inline_ldr:
            inline_textures.append(texture[1])
        else:
            f = open(os.path.join(conf.customized_parts_dir, f"{texture[0]}.dat"), "w")
            f.write(texture[1])
            f.close()

    return clothes_bricks + front_bricks, inline_textures


def gen_jaw(jaw, skin_color: int):
    if jaw == 3:  # unsupported jaw
        jaw = 0

    jaw_file = f"jaw/jaw_{jaw}"
    jaw_skin_files = get_skin_files([jaw_file])

    skin_color_id = get_skin_color(skin_color)
    if jaw == 2:
        if skin_color == 0:
            jaw_bricks = get_bricks_from_files([jaw_file], assign_color_id=0)
        else:
            jaw_bricks = get_bricks_from_files([jaw_file], assign_color_id=71)
    else:
        jaw_bricks = get_bricks_from_files([jaw_file])

    jaw_skin_bricks = get_bricks_from_files(jaw_skin_files, skin_color_id)

    return jaw_bricks, jaw_skin_bricks


def get_skin_files(selected_files):
    skin_files = []

    for file in selected_files:
        skined_file = file + "_skin"
        file_path = os.path.join(conf.parts_dir, skined_file + ".ldr")
        if os.path.exists(file_path):
            skin_files.append(skined_file)

    return skin_files


# get the bricks of the indicate files
def get_bricks_from_files(files: List[str], assign_color_id=None):
    total_bricks = []

    for file in files:
        bricks = read_bricks_from_file(
            os.path.join(conf.parts_dir, file + ".ldr"), read_fake_bricks=True
        )
        total_bricks += bricks

    if assign_color_id is not None:
        for b in total_bricks:
            b.color = assign_color_id

    return total_bricks

def gen_alart_model(debugger):
    copyfile(os.path.join(conf.parts_dir, "404.ldr"), debugger.file_path(f"complete.ldr"))