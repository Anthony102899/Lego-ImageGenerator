from bricks_modeling.file_IO.model_writer import write_bricks_to_file_with_steps, write_model_to_file
from util.debugger import MyDebugger
import os
import csv
import solvers.brick_heads.bach_render_images as render
import solvers.brick_heads.part_selection as p_select
from bricks_modeling.file_IO.model_reader import read_model_from_file, read_bricks_from_file
import numpy as np

'''
We assume the following information is provided:
1) assembly order
2) grouping
3) default camera view
'''

if __name__ == "__main__":
    debugger = MyDebugger("brick_heads")
    file_path = r"data/full_models/hierarchy_test.ldr"
    model = read_model_from_file(file_path, read_fake_bricks=True)
    write_bricks_to_file_with_steps(model.get_bricks(), debugger.file_path(f"complete_bricks.ldr"))
    write_model_to_file(model, debugger.file_path(f"complete_full.ldr"))