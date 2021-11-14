from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from util.debugger import MyDebugger
from bricks_modeling.bricks.brick_factory import get_all_brick_templates
from bricks_modeling.bricks.brickinstance import BrickInstance
import numpy as np
from util.geometry_util import get_random_transformation
import os


def read_colors():
    rgb2id = dict()
    id2name = dict()
    color_file = os.path.join(os.path.dirname(__file__), "my_LDConfig.ldr")
    f = open(color_file, "r")

    for line in f.readlines():
        if line.startswith("0 !COLOUR"):
            line_content = line.rstrip().split()
            if len(line_content) <= 9 or (len(line_content) > 9 and line_content[9] != "ALPHA"): # ignore transparent color
                color_id = int(line_content[4])
                color_name = line_content[2]
                color = (int(line_content[6][1+0*2:1+0*2+2], 16),
                         int(line_content[6][1+1*2:1+1*2+2], 16),
                         int(line_content[6][1+2*2:1+2*2+2], 16))
                rgb2id[color] = color_id
                id2name[color_id] = color_name

    return rgb2id, id2name

if __name__ == "__main__":
    print(read_colors())