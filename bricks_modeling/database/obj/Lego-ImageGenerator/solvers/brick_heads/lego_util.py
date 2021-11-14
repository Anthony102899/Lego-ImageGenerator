import random
import string
from solvers.brick_heads import texture_to_brick
import solvers.brick_heads.config as conf
import os

def get_random_string(length):
    # Random string with the combination of lower and upper case
    letters = string.ascii_letters
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

uv_mapping = {"3245c":(20, 0, -20, 48, -10.1),
              "3010":(-40, 0, 40, 24, 10.2)
              }

def add_texture_to_brick(brick_id:str, image:str):
    customized_id = brick_id + "_"+str(hash(brick_id + str(image)))[:(7-len(brick_id))]

    texture_content = texture_to_brick.image_to_lego_texture(
        os.path.join(conf.parts_dir, "textures", image if str(image).endswith(".png") else f"{image}.png"),
        brick_id, customized_id, *uv_mapping[brick_id])

    return texture_content, customized_id