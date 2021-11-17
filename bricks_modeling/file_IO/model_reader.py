import copy
import math
import numpy as np
from bricks_modeling.bricks.brick_factory import get_all_brick_templates
from bricks_modeling.bricks.brickinstance import BrickInstance
from bricks_modeling.file_IO import model_writer
from util.debugger import MyDebugger
from bricks_modeling.bricks.bricktemplate import BrickTemplate
from bricks_modeling.bricks.model import Model
from bricks_modeling.bricks.brick_group import BrickGroup
from bricks_modeling.file_IO.util import *
import os


def read_model_from_file(file_path, read_fake_bricks=False):
    if not os.path.exists(file_path):
        print("file not exist!")
        exit(7)

    f = open(file_path, "r")
    brick_templates, template_ids = get_all_brick_templates()
    group_names = read_all_subgroup_names(file_path)
    model = Model(group_names)

    current_file = model.get_root_group()

    lines = f.readlines()
    for line in lines:
        line_content = line.rstrip().split(" ")
        if len(line_content) <= 1:
            continue

        if is_file_name_annotation(line_content):
            file_name = get_group_name(line_content).lower()
            print(f"Read a new file {file_name}")
            current_file = model.groups[file_name]
        elif is_a_brick(line_content, group_names):
            current_file.add_brick(
                line_content, brick_templates, template_ids, read_fake_bricks
            )
        elif is_brick_group(line_content, group_names):
            print("parts group: ", line_content)
            current_file.add_a_subgroup(line_content)
        elif is_step_annotation(line_content):
            current_file.add_a_step()
        else:
            print(f"unknown condition for line:{line}")
            pass
    return model

def read_bricks_from_file(file_path, read_fake_bricks=False):
    print(f"reading file: {file_path}...")
    hierarchy_model = read_model_from_file(file_path, read_fake_bricks)
    return hierarchy_model.get_bricks()

if __name__ == "__main__":
    bricks = read_bricks_from_file("../../data/full_models/F-Long-Hair.ldr")
    debugger = MyDebugger("test")
    model_writer.write_bricks_to_file(bricks, debugger.file_path("model.ldr"))
