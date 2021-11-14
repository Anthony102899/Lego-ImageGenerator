from bricks_modeling.file_IO.model_writer import write_bricks_to_file_with_steps
from util.debugger import MyDebugger
import os
import csv
import solvers.brick_heads.bach_render_images as render
import solvers.brick_heads.part_selection as p_select
from bricks_modeling.file_IO.model_reader import read_bricks_from_file
import numpy as np

def one_line_arrange():
    final_str = ""
    left_trans_vec = np.array([150, 0, 70])
    right_trans_vec = np.array([-150, 0, 70])

    i = 1

    for j in range(1, 50):
        file_path = os.path.join(dir_path, f"complete_{j}.ldr")
        if os.path.exists(file_path):
            bricks = read_bricks_from_file(file_path, read_fake_bricks=True)
            trans_vec = (i // 2) * left_trans_vec if i % 2 == 0 else (i // 2) * right_trans_vec
            for b in bricks:
                b.translate(trans_vec)
                final_str += b.to_ldraw()
                final_str += "\n"
            final_str += "\n0 STEP\n"
            i = i + 1

    return final_str

def line_arrange():
    final_str = ""

    for i in range(80, 104):
        file_path = os.path.join(dir_path, f"complete_{i}.ldr")
        bricks = read_bricks_from_file(file_path, read_fake_bricks=True)
        trans_vec = np.array([180*i, 0, 0])
        for b in bricks:
            b.translate(trans_vec)
            final_str += b.to_ldraw()
            final_str += "\n"
        final_str += "\n0 STEP\n"

    return final_str


def triangle_arrange():
    final_str = ""
    z_shift = 300
    x_shift = 300

    row, row_shift = 0,0

    for j in range(1, 120):
        file_path = os.path.join(dir_path, f"complete_{j}.ldr")
        if os.path.exists(file_path):
            bricks = read_bricks_from_file(file_path, read_fake_bricks=True)
            x = (-row * x_shift)/2 + row_shift * x_shift
            z = row * z_shift
            trans_vec = np.array([x, 0, z])
            for b in bricks:
                b.translate(trans_vec)
                final_str += b.to_ldraw()
                final_str += "\n"
            final_str += "\n0 STEP\n"

            if row_shift + 1 > row:
                row = row + 1
                row_shift = 0
            else:
                row_shift = row_shift + 1

    return final_str

if __name__ == "__main__":
    debugger = MyDebugger("brick_heads")
    dir_path = r"/Users/apple/Dropbox/deecamp/results"

    final_str = line_arrange()

    file = open(debugger.file_path('composed.ldr'), "a")
    file.write(final_str)
    file.close()
