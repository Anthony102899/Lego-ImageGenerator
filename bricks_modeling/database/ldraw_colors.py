from bricks_modeling.file_IO.model_reader import read_bricks_from_file
from bricks_modeling.file_IO.model_writer import write_bricks_to_file
from bricks_modeling.connectivity_graph import ConnectivityGraph
from util.debugger import MyDebugger
from bricks_modeling.bricks.brick_factory import get_all_brick_templates
from bricks_modeling.bricks.brickinstance import BrickInstance
import numpy as np
from util.geometry_util import get_random_transformation


def read_colors():
    results = dict()
    color_file = "./bricks_modeling/database/ldraw/LDConfig.ldr"
    f = open(color_file, "r")

    for line in f.readlines():
        if line.startswith("0 !COLOUR"):
            line_content = line.rstrip().split()
            color = (int(line_content[6][1+0*2:1+0*2+2], 16),
                     int(line_content[6][1+1*2:1+1*2+2], 16),
                     int(line_content[6][1+2*2:1+2*2+2], 16))
            results[color] = int(line_content[4])

    print(results)

if __name__ == "__main__":
    read_colors()