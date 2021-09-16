from bricks_modeling.file_IO.model_writer import write_bricks_to_file_with_steps
from util.debugger import MyDebugger
import os
import csv
import solvers.brick_heads.bach_render_images as render
import solvers.brick_heads.part_selection as p_select
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
import numpy as np

if __name__ == "__main__":
    debugger = MyDebugger("brick_heads")
    dir_path = r"/Users/apple/workspace/lego-photo-studio/debug/2020-08-04_11-35-00_brick_heads"

    final_str = ""
    brick_count = {}

    for i in range(1, 108):
        file_path = os.path.join(dir_path, f"complete_{i}.ldr")
        bricks = read_bricks_from_file(file_path, read_fake_bricks=True)
        for b in bricks:
            if b.template.id not in brick_count:
                brick_count[b.template.id] = 1
            else:
                brick_count[b.template.id] += 1

    for key, value in brick_count.items():
        print(f"{key}\t{value}")